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


class ConvTasNetLyricsProcessor(AudioProcessor):
    """Buffered causal Conv-TasNet lyrics/accompaniment separator.

    The model itself is causal, but this first integration deliberately reuses the
    proven packet-buffering and paced-output behaviour of the HTDemucs processor.
    That keeps the network path stable while allowing the segment duration to be
    reduced independently as real-time performance is measured.
    """

    name = "convtasnet-lyrics-causal"
    description = (
        "Experimental causal Conv-TasNet lyric reduction using the Cadenza "
        "pretrained stereo model. Requires the GPU image."
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
        model_path: str = "/models/convtasnet-lyrics-causal",
        device: str = "auto",
        segment_seconds: float = 1.0,
        vocal_reduction: float = 1.0,
        vocal_source_index: int = 0,
        accompaniment_source_index: int = 1,
        inference_fn: InferenceFunction | None = None,
    ) -> None:
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be greater than zero")
        if not 0 <= vocal_reduction <= 1:
            raise ValueError("vocal_reduction must be between 0.0 and 1.0")
        if vocal_source_index < 0 or accompaniment_source_index < 0:
            raise ValueError("source indexes must be zero or greater")

        self.model_path = model_path
        self.requested_device = device
        self.segment_seconds = segment_seconds
        self.vocal_reduction = vocal_reduction
        self.vocal_source_index = vocal_source_index
        self.accompaniment_source_index = accompaniment_source_index
        self._inference_fn = inference_fn
        self._model: Any | None = None
        self._device = device
        self._model_sample_rate = 44100

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
            logger.info("ConvTasNet processor using injected inference function")
            return
        await asyncio.to_thread(self._load_model)

    async def stop(self) -> None:
        await self.reset()
        self._model = None

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
            raise ValueError(
                f"ConvTasNet requires KANY v1 f32 PCM packets: {exc}"
            ) from exc

        if decoded.channels != 2:
            raise ValueError(
                "ConvTasNet lyrics processor requires stereo input; "
                f"received {decoded.channels} channels"
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
            "model_path": self.model_path,
            "device": self._device,
            "model_sample_rate": self._model_sample_rate,
            "segment_seconds": self.segment_seconds,
            "segment_frames": segment_frames,
            "vocal_reduction": self.vocal_reduction,
            "vocal_source_index": self.vocal_source_index,
            "accompaniment_source_index": self.accompaniment_source_index,
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

    def _load_model(self) -> None:
        try:
            import torch
            from audio_trombone.vendor.clarity_tasnet import ConvTasNetStereo
        except ImportError as exc:
            raise RuntimeError(
                "ConvTasNet dependencies are not installed; build with "
                "Dockerfile.demucs"
            ) from exc

        if self.requested_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.requested_device

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CONVTASNET_DEVICE requests CUDA but PyTorch cannot see a GPU"
            )

        logger.info(
            "Loading ConvTasNet model=%s device=%s segment=%.3fs vocal_reduction=%.2f",
            self.model_path,
            self._device,
            self.segment_seconds,
            self.vocal_reduction,
        )
        self._model = ConvTasNetStereo.from_pretrained(
            self.model_path,
            local_files_only=True,
        ).to(self._device)
        self._model.eval()
        self._model_sample_rate = int(getattr(self._model, "samplerate", 44100))
        logger.info(
            "Loaded ConvTasNet sample_rate=%d audio_channels=%s sources=%s",
            self._model_sample_rate,
            getattr(self._model, "audio_channels", None),
            getattr(self._model, "C", None),
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
            infer(), name=f"convtasnet-segment-{self.segments_started}"
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
            raise RuntimeError(f"ConvTasNet inference failed: {exc}") from exc

        expected = sum(packet.frames * packet.channels for packet in self._active_packets)
        if len(separated) != expected:
            self._active_packets.clear()
            raise RuntimeError(
                f"ConvTasNet returned {len(separated)} samples; expected {expected}"
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
            "ConvTasNet segment complete: inference=%.3fs rtf=%.3f ready_packets=%d",
            self.last_inference_seconds or 0.0,
            self.last_real_time_factor or 0.0,
            len(self._ready_output),
        )

    def _run_inference(self, samples: array, sample_rate: int, channels: int) -> array:
        if self._inference_fn is not None:
            return self._inference_fn(samples, sample_rate, channels)
        if self._model is None:
            raise RuntimeError("ConvTasNet model is not loaded")

        import torch
        import torchaudio.functional as audio_functional

        waveform = torch.tensor(samples, dtype=torch.float32)
        waveform = waveform.view(-1, channels).transpose(0, 1).contiguous()
        waveform = waveform.unsqueeze(0)

        if sample_rate != self._model_sample_rate:
            waveform = audio_functional.resample(
                waveform, sample_rate, self._model_sample_rate
            )
        waveform = waveform.to(self._device)

        with torch.inference_mode():
            estimates = self._model(waveform)
            if estimates.ndim != 4:
                raise RuntimeError(
                    "Expected ConvTasNet output [batch, sources, channels, frames], "
                    f"received shape {tuple(estimates.shape)}"
                )
            source_count = estimates.shape[1]
            if max(self.vocal_source_index, self.accompaniment_source_index) >= source_count:
                raise RuntimeError(
                    "Configured ConvTasNet source index exceeds model output count: "
                    f"sources={source_count}"
                )

            vocals = estimates[0, self.vocal_source_index]
            accompaniment = estimates[0, self.accompaniment_source_index]
            output = (
                accompaniment * self.vocal_reduction
                + waveform[0] * (1.0 - self.vocal_reduction)
            )

            if self._model_sample_rate != sample_rate:
                output = audio_functional.resample(
                    output, self._model_sample_rate, sample_rate
                )

            target_frames = len(samples) // channels
            if output.shape[-1] < target_frames:
                output = torch.nn.functional.pad(
                    output, (0, target_frames - output.shape[-1])
                )
            output = output[..., :target_frames].clamp(-1.0, 1.0)
            interleaved = (
                output.transpose(0, 1)
                .contiguous()
                .view(-1)
                .to(device="cpu", dtype=torch.float32)
            )

        return array("f", interleaved.tolist())
