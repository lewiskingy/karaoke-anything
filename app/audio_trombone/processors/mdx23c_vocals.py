"""Finite-window, non-causal MDX23C vocal-reduction processor."""

from array import array
import asyncio
from collections import deque
from collections.abc import Callable
import time
from typing import Any, AsyncIterator

from audio_trombone.kany import KanyPacket, KanyProtocolError
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor, ProcessorCapabilities

InferenceFunction = Callable[[array, int, int], array]
SUPPORTED_SEGMENTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def _raise_first_failure(checks: tuple[tuple[bool, str], ...]) -> None:
    for passed, message in checks:
        if not passed:
            raise ValueError(message)


class MDX23CVocalsProcessor(AudioProcessor):
    name = "mdx23c-vocals"
    description = (
        "Finite-window, non-causal MDX23C instrumental/vocal reduction with "
        "bounded buffering and paced KANY output."
    )
    capabilities = ProcessorCapabilities(False, True, True, True)

    def __init__(self, *, config_path: str = "/models/mdx23c/model_2_stem_full_band_8k.yaml",
                 checkpoint_path: str = "/models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt",
                 device: str = "auto", segment_seconds: float = 1.0,
                 overlap: float = 0.25, batch_size: int = 1,
                 vocal_reduction: float = 1.0, precision: str = "float32",
                 warm_up: bool = False, max_buffered_segments: int = 3,
                 inference_fn: InferenceFunction | None = None) -> None:
        _raise_first_failure((
            (
                segment_seconds in SUPPORTED_SEGMENTS,
                f"segment_seconds must be one of {SUPPORTED_SEGMENTS}",
            ),
            (0 <= overlap < 0.5, "overlap must be between 0.0 and less than 0.5"),
            (batch_size == 1, "batch_size must be 1 for paced interactive processing"),
            (0 <= vocal_reduction <= 1, "vocal_reduction must be between 0.0 and 1.0"),
            (
                precision in {"float32", "float16", "bfloat16"},
                "precision must be float32, float16, or bfloat16",
            ),
        ))
        self.config_path, self.checkpoint_path = config_path, checkpoint_path
        self.requested_device, self.device = device, device
        self.segment_seconds, self.overlap = segment_seconds, overlap
        self.batch_size, self.vocal_reduction = batch_size, vocal_reduction
        self.precision, self.warm_up = precision, warm_up
        self.max_buffered_segments = max_buffered_segments
        self._inference_fn, self._adapter = inference_fn, None
        self._input: deque[KanyPacket] = deque()
        self._input_frames = 0
        self._ready: deque[ProcessedPacket] = deque()
        self._task: asyncio.Task[array] | None = None
        self._active: list[KanyPacket] = []
        self._rate: int | None = None
        self._channels: int | None = None
        self._previous_tail: array | None = None
        self.segments_started = self.segments_completed = self.dropped_packets = 0
        self.underruns = 0
        self.last_inference_seconds: float | None = None
        self.last_real_time_factor: float | None = None
        self.last_error: str | None = None
        self.model_loaded = False

    async def start(self) -> None:
        if self._inference_fn is None:
            from audio_trombone.mdx23c import MDX23CAdapter
            self._adapter = await asyncio.to_thread(
                MDX23CAdapter, self.config_path, self.checkpoint_path,
                device=self.requested_device, precision=self.precision)
            self.device = self._adapter.device
            self.model_loaded = True
            if self.warm_up:
                import torch
                frames = round(self._adapter.sample_rate * self.segment_seconds)
                await asyncio.to_thread(self._adapter.infer, torch.zeros(1, 2, frames), self._adapter.sample_rate)
        else:
            self.model_loaded = True

    async def stop(self) -> None:
        await self.reset(); self._adapter = None; self.model_loaded = False

    async def reset(self) -> None:
        if self._task:
            self._task.cancel()
            try: await self._task
            except (asyncio.CancelledError, Exception): pass
        self._task = None; self._active.clear(); self._input.clear(); self._ready.clear()
        self._input_frames = 0; self._rate = self._channels = None
        self._previous_tail = None; self.last_error = None

    @staticmethod
    def _decode_stereo_packet(payload: bytes) -> KanyPacket:
        try:
            decoded = KanyPacket.decode(payload)
        except KanyProtocolError as exc:
            raise ValueError(f"MDX23C requires KANY v1 f32 PCM: {exc}") from exc
        if decoded.channels != 2:
            raise ValueError("MDX23C requires stereo input")
        return decoded

    async def _track_stream_format(self, decoded: KanyPacket) -> None:
        if self._rate is None:
            self._rate, self._channels = decoded.sample_rate, decoded.channels
            return
        if (decoded.sample_rate, decoded.channels) != (self._rate, self._channels):
            await self.reset()
            raise ValueError("audio format changed; MDX23C state was reset")

    def _buffer_packet(self, decoded: KanyPacket) -> None:
        self._input.append(decoded)
        self._input_frames += decoded.frames
        maximum = round(decoded.sample_rate * self.segment_seconds * self.max_buffered_segments)
        while self._input_frames > maximum and self._input:
            dropped = self._input.popleft()
            self._input_frames -= dropped.frames
            self.dropped_packets += 1

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        await self._harvest()
        decoded = self._decode_stereo_packet(packet.payload)
        await self._track_stream_format(decoded)
        self._buffer_packet(decoded)

        if self._task is None and self._input_frames >= self._target_frames():
            self._launch()

        if self._ready:
            yield self._ready.popleft()
        elif self._task is not None:
            self.underruns += 1

    async def flush(self) -> AsyncIterator[ProcessedPacket]:
        await self._harvest()
        while self._ready: yield self._ready.popleft()

    def _target_frames(self) -> int:
        assert self._rate is not None
        return round(self._rate * self.segment_seconds)

    def _launch(self) -> None:
        packets, frames = [], 0
        while self._input and frames < self._target_frames():
            item = self._input.popleft(); packets.append(item); frames += item.frames; self._input_frames -= item.frames
        samples = array("f"); [samples.extend(item.samples) for item in packets]
        self._active = packets; self.segments_started += 1; started = time.perf_counter()
        async def run() -> array:
            result = await asyncio.to_thread(self._run_inference, samples, packets[0].sample_rate, 2)
            elapsed = time.perf_counter() - started; self.last_inference_seconds = elapsed
            self.last_real_time_factor = elapsed / (frames / packets[0].sample_rate); return result
        self._task = asyncio.create_task(run(), name=f"mdx23c-segment-{self.segments_started}")

    async def _harvest(self) -> None:
        if self._task is None or not self._task.done(): return
        task, self._task = self._task, None
        try: output = await task
        except Exception as exc: self._active.clear(); self.last_error = str(exc); raise RuntimeError(f"MDX23C inference failed: {exc}") from exc
        expected = sum(p.frames * 2 for p in self._active)
        if len(output) != expected: self._active.clear(); raise RuntimeError(f"MDX23C returned {len(output)} samples; expected {expected}")
        self._crossfade(output)
        offset = 0
        for original in self._active:
            count = original.frames * 2
            self._ready.append(ProcessedPacket(payload=original.encode_samples(array("f", output[offset:offset+count])))); offset += count
        self._active.clear(); self.segments_completed += 1; self.last_error = None

    def _crossfade(self, output: array) -> None:
        frames = len(output) // 2
        overlap_frames = min(round(frames * self.overlap), frames // 2)
        samples = overlap_frames * 2
        if samples and self._previous_tail is not None:
            for frame in range(overlap_frames):
                alpha = (frame + 1) / (overlap_frames + 1)
                for channel in range(2):
                    index = frame * 2 + channel
                    output[index] = self._previous_tail[index] * (1-alpha) + output[index] * alpha
        self._previous_tail = array("f", output[-samples:]) if samples else None

    def _run_inference(self, samples: array, sample_rate: int, channels: int) -> array:
        if self._inference_fn: return self._inference_fn(samples, sample_rate, channels)
        if self._adapter is None: raise RuntimeError("MDX23C model is not loaded")
        import torch, torchaudio.functional as AF
        waveform = torch.tensor(samples).view(-1, 2).t().unsqueeze(0)
        original_frames = waveform.shape[-1]
        if sample_rate != self._adapter.sample_rate: waveform = AF.resample(waveform, sample_rate, self._adapter.sample_rate)
        estimates = self._adapter.infer(waveform, self._adapter.sample_rate)
        instrumental = estimates[0, self._adapter.stem_index("instrumental", "other", "accompaniment")]
        original = waveform[0]
        result = instrumental * self.vocal_reduction + original * (1-self.vocal_reduction)
        if sample_rate != self._adapter.sample_rate: result = AF.resample(result, self._adapter.sample_rate, sample_rate)
        result = torch.nn.functional.pad(result, (0, max(0, original_frames-result.shape[-1])))[:, :original_frames].clamp(-1, 1)
        return array("f", result.t().contiguous().cpu().view(-1).tolist())

    def diagnostics(self) -> dict[str, object]:
        rate = self._rate or 0
        return {"device": self.device, "precision": self.precision, "model_loaded": self.model_loaded,
                "segment_seconds": self.segment_seconds, "overlap": self.overlap, "batch_size": self.batch_size,
                "vocal_reduction": self.vocal_reduction, "effective_input_buffering_seconds": self.segment_seconds,
                "estimated_algorithmic_latency_seconds": self.segment_seconds,
                "queued_input_seconds": self._input_frames/rate if rate else 0.0,
                "queued_output_seconds": sum(len(p.payload)-28 for p in self._ready)/(rate*2*4) if rate else 0.0,
                "buffered_input_packets": len(self._input), "ready_output_packets": len(self._ready),
                "dropped_packets": self.dropped_packets, "underruns": self.underruns,
                "inference_running": self._task is not None, "segments_started": self.segments_started,
                "segments_completed": self.segments_completed, "last_inference_seconds": self.last_inference_seconds,
                "last_real_time_factor": self.last_real_time_factor, "last_error": self.last_error,
                "causal": False}
