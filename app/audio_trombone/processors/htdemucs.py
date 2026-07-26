from array import array
import asyncio
from collections import deque
from collections.abc import Callable
import logging
import time
from typing import AsyncIterator, Any

from audio_trombone.kany import KanyPacket, KanyProtocolError
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor, ProcessorCapabilities

logger = logging.getLogger(__name__)

InferenceFunction = Callable[[array, int, int], array]


class HTDemucsProcessor(AudioProcessor):
    """Experimental buffered vocals/accompaniment separator.

    Incoming KANY f32 PCM packets are accumulated into a fixed-duration segment.
    Demucs inference runs on a worker thread while the asyncio UDP loop continues
    accepting packets. Once a segment is ready, one processed packet is released
    for each subsequent input packet, preserving the client's natural playback
    pacing rather than bursting an entire separated segment at once.
    """

    name = "htdemucs-vocals"
    description = (
        "Buffered HTDemucs vocal reduction. Adds a fixed multi-second delay and "
        "requires the Demucs GPU image for practical real-time operation."
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
        model_name: str = "htdemucs",
        device: str = "auto",
        segment_seconds: float = 6.0,
        overlap: float = 0.25,
        shifts: int = 0,
        vocal_reduction: float = 1.0,
        inference_fn: InferenceFunction | None = None,
    ) -> None:
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be greater than zero")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between zero and one")
        if not 0 <= vocal_reduction <= 1:
            raise ValueError("vocal_reduction must be between 0.0 and 1.0")

        self.model_name = model_name
        self.requested_device = device
        self.segment_seconds = segment_seconds
        self.overlap = overlap
        self.shifts = shifts
        self.vocal_reduction = vocal_reduction
        self._inference_fn = inference_fn
        self._separator: Any | None = None
        self._device = device

        self._input_packets: deque[KanyPacket] = deque()
        self._input_frames = 0
        self._ready_output: deque[ProcessedPacket] = deque()
        self._inference_task: asyncio.Task[array] | None = None
        self._active_packets: list[KanyPacket] = []
        self._stream_sample_rate: int | None = None
        self._stream_channels: int | None = None

        self.segments_started = 0
        self.segments_completed = 0
        self.last_inference_seconds: float | None = None
        self.last_real_time_factor: float | None = None
        self.last_error: str | None = None

    async def start(self) -> None:
        if self._inference_fn is not None:
            logger.info("HTDemucs processor using injected inference function")
            return
        await asyncio.to_thread(self._load_separator)

    async def stop(self) -> None:
        await self.reset()
        self._separator = None

    async def reset(self) -> None:
        if self._inference_task is not None:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except (asyncio.CancelledError, Exception):
                pass
        self._inference_task = None
        self._active_packets.clear()
        self._input_packets.clear()
        self._input_frames = 0
        self._ready_output.clear()
        self._stream_sample_rate = None
        self._stream_channels = None
        self.last_error = None

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        await self._harvest_inference()

        try:
            decoded = KanyPacket.decode(packet.payload)
        except KanyProtocolError as exc:
            raise ValueError(f"HTDemucs requires KANY v1 f32 PCM packets: {exc}") from exc

        if decoded.channels != 2:
            raise ValueError(
                f"HTDemucs prototype requires stereo input; received {decoded.channels} channels"
            )

        self._accept_stream_format(decoded)
        self._input_packets.append(decoded)
        self._input_frames += decoded.frames

        if self._inference_task is None and self._segment_ready():
            self._launch_segment()

        if self._ready_output:
            yield self._ready_output.popleft()

    async def flush(self) -> AsyncIterator[ProcessedPacket]:
        await self._harvest_inference()
        while self._ready_output:
            yield self._ready_output.popleft()

    def diagnostics(self) -> dict[str, object]:
        segment_frames = None
        if self._stream_sample_rate is not None:
            segment_frames = self._target_segment_frames()
        return {
            "model": self.model_name,
            "device": self._device,
            "segment_seconds": self.segment_seconds,
            "segment_frames": segment_frames,
            "overlap": self.overlap,
            "shifts": self.shifts,
            "vocal_reduction": self.vocal_reduction,
            "vocal_gain": 1.0 - self.vocal_reduction,
            "buffered_input_packets": len(self._input_packets),
            "buffered_input_frames": self._input_frames,
            "ready_output_packets": len(self._ready_output),
            "inference_running": self._inference_task is not None,
            "segments_started": self.segments_started,
            "segments_completed": self.segments_completed,
            "last_inference_seconds": self.last_inference_seconds,
            "last_real_time_factor": self.last_real_time_factor,
            "last_error": self.last_error,
        }

    def _load_separator(self) -> None:
        try:
            import torch
            from demucs.api import Separator
        except ImportError as exc:
            raise RuntimeError(
                "HTDemucs dependencies are not installed; build with Dockerfile.demucs"
            ) from exc

        if self.requested_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.requested_device

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DEMUCS_DEVICE requests CUDA but PyTorch cannot see a GPU")

        logger.info(
            "Loading Demucs model=%s device=%s segment=%.2fs overlap=%.2f shifts=%d vocal_reduction=%.2f",
            self.model_name,
            self._device,
            self.segment_seconds,
            self.overlap,
            self.shifts,
            self.vocal_reduction,
        )
        self._separator = Separator(
            model=self.model_name,
            device=self._device,
            segment=self.segment_seconds,
            shifts=self.shifts,
            split=True,
            overlap=self.overlap,
            progress=False,
        )
        logger.info(
            "Loaded Demucs model=%s samplerate=%s channels=%s vocal_gain=%.2f",
            self.model_name,
            self._separator.samplerate,
            self._separator.audio_channels,
            1.0 - self.vocal_reduction,
        )

    def _accept_stream_format(self, packet: KanyPacket) -> None:
        if self._stream_sample_rate is None:
            self._stream_sample_rate = packet.sample_rate
            self._stream_channels = packet.channels
            return
        if (
            packet.sample_rate != self._stream_sample_rate
            or packet.channels != self._stream_channels
        ):
            raise ValueError(
                "audio format changed without processor reset: "
                f"expected {self._stream_sample_rate}Hz/{self._stream_channels}ch, "
                f"received {packet.sample_rate}Hz/{packet.channels}ch"
            )

    def _target_segment_frames(self) -> int:
        if self._stream_sample_rate is None:
            raise RuntimeError("stream format is not known")
        return max(1, round(self._stream_sample_rate * self.segment_seconds))

    def _segment_ready(self) -> bool:
        return self._input_frames >= self._target_segment_frames()

    def _launch_segment(self) -> None:
        target = self._target_segment_frames()
        packets: list[KanyPacket] = []
        frames = 0
        while self._input_packets and frames < target:
            current = self._input_packets.popleft()
            packets.append(current)
            frames += current.frames
            self._input_frames -= current.frames

        samples = array("f")
        for current in packets:
            samples.extend(current.samples)

        self._active_packets = packets
        self.segments_started += 1
        started = time.perf_counter()

        async def infer() -> array:
            result = await asyncio.to_thread(
                self._run_inference,
                samples,
                packets[0].sample_rate,
                packets[0].channels,
            )
            elapsed = time.perf_counter() - started
            self.last_inference_seconds = elapsed
            duration = frames / packets[0].sample_rate
            self.last_real_time_factor = elapsed / duration
            return result

        self._inference_task = asyncio.create_task(
            infer(), name=f"htdemucs-segment-{self.segments_started}"
        )

    async def _harvest_inference(self) -> None:
        task = self._inference_task
        if task is None or not task.done():
            return

        self._inference_task = None
        try:
            separated = await task
        except asyncio.CancelledError:
            self._active_packets.clear()
            raise
        except Exception as exc:
            self.last_error = str(exc)
            self._active_packets.clear()
            raise RuntimeError(f"HTDemucs inference failed: {exc}") from exc

        expected = sum(packet.frames * packet.channels for packet in self._active_packets)
        if len(separated) != expected:
            self._active_packets.clear()
            raise RuntimeError(
                f"HTDemucs returned {len(separated)} samples; expected {expected}"
            )

        offset = 0
        for original in self._active_packets:
            count = original.frames * original.channels
            packet_samples = array("f", separated[offset : offset + count])
            self._ready_output.append(
                ProcessedPacket(payload=original.encode_samples(packet_samples))
            )
            offset += count

        self._active_packets.clear()
        self.segments_completed += 1
        self.last_error = None
        logger.info(
            "HTDemucs segment complete: inference=%.3fs rtf=%.3f ready_packets=%d",
            self.last_inference_seconds or 0.0,
            self.last_real_time_factor or 0.0,
            len(self._ready_output),
        )

    def _run_inference(self, samples: array, sample_rate: int, channels: int) -> array:
        if self._inference_fn is not None:
            return self._inference_fn(samples, sample_rate, channels)
        if self._separator is None:
            raise RuntimeError("Demucs separator is not loaded")

        import torch
        import torchaudio.functional as audio_functional

        waveform = torch.tensor(samples, dtype=torch.float32)
        waveform = waveform.view(-1, channels).transpose(0, 1).contiguous()

        with torch.inference_mode():
            original, stems = self._separator.separate_tensor(waveform, sr=sample_rate)
            vocals = stems.get("vocals")
            if vocals is None:
                raise RuntimeError("Demucs model did not return a vocals stem")

            # 0.0 leaves the original mix unchanged; 1.0 removes the estimated
            # vocal stem completely; intermediate values retain some guide vocal.
            output = original - (vocals * self.vocal_reduction)

            model_rate = int(self._separator.samplerate)
            if model_rate != sample_rate:
                output = audio_functional.resample(output, model_rate, sample_rate)

            target_frames = len(samples) // channels
            if output.shape[-1] < target_frames:
                output = torch.nn.functional.pad(
                    output, (0, target_frames - output.shape[-1])
                )
            output = output[..., :target_frames]
            output = output.clamp(-1.0, 1.0)
            interleaved = (
                output.transpose(0, 1)
                .contiguous()
                .view(-1)
                .to(device="cpu", dtype=torch.float32)
            )

        return array("f", interleaved.tolist())