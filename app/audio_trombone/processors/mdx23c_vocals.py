"""Finite-window, non-causal MDX23C vocal-reduction processor."""

import asyncio
import time
from array import array
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, get_args

from audio_trombone.kany import HEADER_SIZE, KanyPacket, KanyProtocolError
from audio_trombone.mdx23c import MDX23CPrecision
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor, ProcessorCapabilities

InferenceFunction = Callable[[array, int, int], array]
SUPPORTED_SEGMENTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
BYTES_PER_SAMPLE = 4  # sizeof(float32); KANY packets are f32 PCM


def _raise_first_failure(checks: tuple[tuple[bool, str], ...]) -> None:
    for passed, message in checks:
        if not passed:
            raise ValueError(message)


@dataclass(frozen=True)
class MDX23CVocalsConfig:
    """Tuning parameters for `MDX23CVocalsProcessor`, grouped into one object
    so the constructor doesn't take an argument per field."""

    config_path: str = "/models/mdx23c/model_2_stem_full_band_8k.yaml"
    checkpoint_path: str = "/models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt"
    device: str = "auto"
    segment_seconds: float = 1.0
    overlap: float = 0.25
    batch_size: int = 1
    vocal_reduction: float = 1.0
    precision: MDX23CPrecision = "float32"
    warm_up: bool = False
    max_buffered_segments: int = 3

    def __post_init__(self) -> None:
        _raise_first_failure(
            (
                (
                    self.segment_seconds in SUPPORTED_SEGMENTS,
                    f"segment_seconds must be one of {SUPPORTED_SEGMENTS}",
                ),
                (
                    # `_crossfade` only ever blends up to half a segment
                    # (`frames // 2`), so overlap can't reach 0.5.
                    0 <= self.overlap < 0.5,
                    "overlap must be between 0.0 and less than 0.5",
                ),
                (
                    self.batch_size == 1,
                    "batch_size must be 1 for paced interactive processing",
                ),
                (
                    0 <= self.vocal_reduction <= 1,
                    "vocal_reduction must be between 0.0 and 1.0",
                ),
                (
                    self.precision in get_args(MDX23CPrecision),
                    "precision must be float32, float16, or bfloat16",
                ),
            )
        )


class MDX23CVocalsProcessor(AudioProcessor):
    """Deliberately does not extend `SegmentedInferenceProcessor`, even though
    its packet-buffering and segment-harvest machinery mirrors that base
    class closely: this processor also does crossfading and drops the
    oldest buffered input once `max_buffered_segments` is exceeded, neither
    of which the base class supports. A bug in segment bookkeeping likely
    applies to both -- check `segmented_inference.py` too."""

    name = "mdx23c-vocals"
    description = (
        "Finite-window, non-causal MDX23C instrumental/vocal reduction with "
        "bounded buffering and paced KANY output."
    )
    capabilities = ProcessorCapabilities(
        passthrough=False,
        stateful=True,
        can_buffer=True,
        changes_payload=True,
    )

    def __init__(
        self,
        *,
        config: MDX23CVocalsConfig | None = None,
        inference_fn: InferenceFunction | None = None,
    ) -> None:
        config = config or MDX23CVocalsConfig()
        self.config_path = config.config_path
        self.checkpoint_path = config.checkpoint_path
        self.requested_device = config.device
        self.device = config.device
        self.segment_seconds = config.segment_seconds
        self.overlap = config.overlap
        self.batch_size = config.batch_size
        self.vocal_reduction = config.vocal_reduction
        self.precision = config.precision
        self.warm_up = config.warm_up
        self.max_buffered_segments = config.max_buffered_segments
        self._inference_fn = inference_fn
        self._adapter: Any | None = None

        self._input_packets: deque[KanyPacket] = deque()
        self._input_frames = 0
        self._ready_output: deque[ProcessedPacket] = deque()
        self._inference_task: asyncio.Task[array] | None = None
        self._active_packets: list[KanyPacket] = []
        self._stream_sample_rate: int | None = None
        self._stream_channels: int | None = None
        self._previous_tail: array | None = None

        self.segments_started = 0
        self.segments_completed = 0
        self.dropped_packets = 0
        self.underruns = 0
        self.last_inference_seconds: float | None = None
        self.last_real_time_factor: float | None = None
        self.last_error: str | None = None
        self.model_loaded = False

    async def start(self) -> None:
        if self._inference_fn is None:
            from audio_trombone.mdx23c import MDX23CAdapter

            adapter = await asyncio.to_thread(
                MDX23CAdapter,
                self.config_path,
                self.checkpoint_path,
                device=self.requested_device,
                precision=self.precision,
            )
            self._adapter = adapter
            self.device = adapter.device
            self.model_loaded = True
            if self.warm_up:
                import torch

                frames = round(adapter.sample_rate * self.segment_seconds)
                await asyncio.to_thread(
                    adapter.infer,
                    torch.zeros(1, 2, frames),
                    adapter.sample_rate,
                )
        else:
            self.model_loaded = True

    async def stop(self) -> None:
        await self.reset()
        self._adapter = None
        self.model_loaded = False

    async def reset(self) -> None:
        if self._inference_task:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                # Cancelling and awaiting our own task: whatever it raises
                # (cancellation or a real failure) is expected here, and
                # this is reset -- there's no meaningful way to surface it.
                pass
        self._inference_task = None
        self._active_packets.clear()
        self._input_packets.clear()
        self._ready_output.clear()
        self._input_frames = 0
        self._stream_sample_rate = None
        self._stream_channels = None
        self._previous_tail = None
        self.last_error = None

    @staticmethod
    def _decode_and_validate(payload: bytes) -> KanyPacket:
        try:
            decoded = KanyPacket.decode(payload)
        except KanyProtocolError as exc:
            raise ValueError(f"MDX23C requires KANY v1 f32 PCM: {exc}") from exc
        if decoded.channels != 2:
            raise ValueError("MDX23C requires stereo input")
        return decoded

    async def _track_stream_format(self, decoded: KanyPacket) -> None:
        if self._stream_sample_rate is None:
            self._stream_sample_rate, self._stream_channels = (
                decoded.sample_rate,
                decoded.channels,
            )
            return
        if (decoded.sample_rate, decoded.channels) != (
            self._stream_sample_rate,
            self._stream_channels,
        ):
            await self.reset()
            raise ValueError("audio format changed; MDX23C state was reset")

    def _buffer_packet(self, decoded: KanyPacket) -> None:
        self._input_packets.append(decoded)
        self._input_frames += decoded.frames
        maximum = round(
            decoded.sample_rate * self.segment_seconds * self.max_buffered_segments
        )
        while self._input_frames > maximum and self._input_packets:
            dropped = self._input_packets.popleft()
            self._input_frames -= dropped.frames
            self.dropped_packets += 1

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        await self._harvest_inference()
        decoded = self._decode_and_validate(packet.payload)
        await self._track_stream_format(decoded)
        self._buffer_packet(decoded)

        if (
            self._inference_task is None
            and self._input_frames >= self._target_segment_frames()
        ):
            self._launch_segment()

        if self._ready_output:
            yield self._ready_output.popleft()
        elif self._inference_task is not None:
            self.underruns += 1

    async def flush(self) -> AsyncIterator[ProcessedPacket]:
        await self._harvest_inference()
        while self._ready_output:
            yield self._ready_output.popleft()

    def _target_segment_frames(self) -> int:
        assert self._stream_sample_rate is not None
        return round(self._stream_sample_rate * self.segment_seconds)

    def _launch_segment(self) -> None:
        packets: list[KanyPacket] = []
        frames = 0
        while self._input_packets and frames < self._target_segment_frames():
            item = self._input_packets.popleft()
            packets.append(item)
            frames += item.frames
            self._input_frames -= item.frames

        samples = array("f")
        for item in packets:
            samples.extend(item.samples)

        self._active_packets = packets
        self.segments_started += 1
        started = time.perf_counter()

        async def run() -> array:
            result = await asyncio.to_thread(
                self._run_inference, samples, packets[0].sample_rate, 2
            )
            elapsed = time.perf_counter() - started
            self.last_inference_seconds = elapsed
            self.last_real_time_factor = elapsed / (frames / packets[0].sample_rate)
            return result

        self._inference_task = asyncio.create_task(
            run(), name=f"mdx23c-segment-{self.segments_started}"
        )

    async def _harvest_inference(self) -> None:
        if self._inference_task is None or not self._inference_task.done():
            return
        task, self._inference_task = self._inference_task, None

        output = await self._await_segment_result(task)
        self._distribute_segment_output(output)
        self._finish_segment()

    async def _await_segment_result(self, task: "asyncio.Task[array]") -> array:
        try:
            return await task
        except Exception as exc:
            self._active_packets.clear()
            self.last_error = str(exc)
            raise RuntimeError(f"MDX23C inference failed: {exc}") from exc

    def _distribute_segment_output(self, output: array) -> None:
        expected = sum(p.frames * 2 for p in self._active_packets)
        if len(output) != expected:
            self._active_packets.clear()
            raise RuntimeError(
                f"MDX23C returned {len(output)} samples; expected {expected}"
            )

        self._crossfade(output)
        offset = 0
        for original in self._active_packets:
            count = original.frames * 2
            self._ready_output.append(
                ProcessedPacket(
                    payload=original.encode_samples(
                        array("f", output[offset : offset + count])
                    )
                )
            )
            offset += count

    def _finish_segment(self) -> None:
        self._active_packets.clear()
        self.segments_completed += 1
        self.last_error = None

    def _usable_tail(self, samples: int) -> array | None:
        # `_previous_tail` was sized by the *previous* segment's `overlap`;
        # since `overlap` can change at runtime (via /api/settings) between
        # segments, guard against it now covering less than this segment wants.
        tail = self._previous_tail
        if samples <= 0 or tail is None:
            return None
        if len(tail) < samples:
            return None
        return tail

    def _crossfade(self, output: array) -> None:
        frames = len(output) // 2
        overlap_frames = min(round(frames * self.overlap), frames // 2)
        samples = overlap_frames * 2
        tail = self._usable_tail(samples)
        if tail is not None:
            for frame in range(overlap_frames):
                alpha = (frame + 1) / (overlap_frames + 1)
                for channel in range(2):
                    index = frame * 2 + channel
                    output[index] = tail[index] * (1 - alpha) + output[index] * alpha
        self._previous_tail = array("f", output[-samples:]) if samples else None

    def _run_inference(self, samples: array, sample_rate: int, channels: int) -> array:
        if self._inference_fn:
            return self._inference_fn(samples, sample_rate, channels)
        if self._adapter is None:
            raise RuntimeError("MDX23C model is not loaded")

        import torch
        import torchaudio.functional as audio_functional

        waveform = torch.tensor(samples).view(-1, 2).t().unsqueeze(0)
        original_frames = waveform.shape[-1]
        if sample_rate != self._adapter.sample_rate:
            waveform = audio_functional.resample(
                waveform, sample_rate, self._adapter.sample_rate
            )

        estimates = self._adapter.infer(waveform, self._adapter.sample_rate)
        instrumental = estimates[
            0, self._adapter.stem_index("instrumental", "other", "accompaniment")
        ]
        original = waveform[0]
        result = instrumental * self.vocal_reduction + original * (
            1 - self.vocal_reduction
        )
        if sample_rate != self._adapter.sample_rate:
            result = audio_functional.resample(
                result, self._adapter.sample_rate, sample_rate
            )

        result = torch.nn.functional.pad(
            result, (0, max(0, original_frames - result.shape[-1]))
        )[:, :original_frames].clamp(-1, 1)
        return array("f", result.t().contiguous().cpu().view(-1).tolist())

    def diagnostics(self) -> dict[str, object]:
        rate = self._stream_sample_rate or 0
        return {
            "device": self.device,
            "precision": self.precision,
            "model_loaded": self.model_loaded,
            "segment_seconds": self.segment_seconds,
            "overlap": self.overlap,
            "batch_size": self.batch_size,
            "vocal_reduction": self.vocal_reduction,
            "effective_input_buffering_seconds": self.segment_seconds,
            "estimated_algorithmic_latency_seconds": self.segment_seconds,
            "queued_input_seconds": self._input_frames / rate if rate else 0.0,
            "queued_output_seconds": (
                sum(len(p.payload) - HEADER_SIZE for p in self._ready_output)
                / (rate * (self._stream_channels or 2) * BYTES_PER_SAMPLE)
                if rate
                else 0.0
            ),
            "buffered_input_packets": len(self._input_packets),
            "ready_output_packets": len(self._ready_output),
            "dropped_packets": self.dropped_packets,
            "underruns": self.underruns,
            "inference_running": self._inference_task is not None,
            "segments_started": self.segments_started,
            "segments_completed": self.segments_completed,
            "last_inference_seconds": self.last_inference_seconds,
            "last_real_time_factor": self.last_real_time_factor,
            "last_error": self.last_error,
            "causal": False,
        }
