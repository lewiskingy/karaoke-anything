from array import array
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import Any

from audio_trombone.kany import KanyPacket, KanyProtocolError
from audio_trombone.processors.base import ProcessorCapabilities
from audio_trombone.processors.segmented_inference import SegmentedInferenceProcessor

logger = logging.getLogger(__name__)

InferenceFunction = Callable[[array, int, int], array]


@dataclass(frozen=True)
class HTDemucsConfig:
    """Tuning parameters for `HTDemucsProcessor`, grouped into one object so
    the constructor doesn't take an argument per field."""

    model_name: str = "htdemucs"
    device: str = "auto"
    segment_seconds: float = 6.0
    overlap: float = 0.25
    shifts: int = 0
    vocal_reduction: float = 1.0

    def __post_init__(self) -> None:
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be greater than zero")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be between zero and one")
        if not 0 <= self.vocal_reduction <= 1:
            raise ValueError("vocal_reduction must be between 0.0 and 1.0")


class HTDemucsProcessor(SegmentedInferenceProcessor):
    """Experimental buffered vocals/accompaniment separator.

    Incoming KANY f32 PCM packets are accumulated into a fixed-duration segment.
    Demucs inference runs on a worker thread while the asyncio UDP loop continues
    accepting packets. Once a segment is ready, one processed packet is released
    for each subsequent input packet, preserving the client's natural playback
    pacing rather than bursting an entire separated segment at once.
    """

    _model_label = "HTDemucs"
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
        config: HTDemucsConfig | None = None,
        inference_fn: InferenceFunction | None = None,
    ) -> None:
        config = config or HTDemucsConfig()
        super().__init__(segment_seconds=config.segment_seconds, inference_fn=inference_fn)

        self.model_name = config.model_name
        self.requested_device = config.device
        self.overlap = config.overlap
        self.shifts = config.shifts
        self.vocal_reduction = config.vocal_reduction
        self._separator: Any | None = None
        self._device = config.device

    async def start(self) -> None:
        if self._inference_fn is not None:
            logger.info("HTDemucs processor using injected inference function")
            return
        await asyncio.to_thread(self._load_separator)

    async def stop(self) -> None:
        await self.reset()
        self._separator = None

    def _decode_and_validate(self, payload: bytes) -> KanyPacket:
        try:
            decoded = KanyPacket.decode(payload)
        except KanyProtocolError as exc:
            raise ValueError(f"HTDemucs requires KANY v1 f32 PCM packets: {exc}") from exc

        if decoded.channels != 2:
            raise ValueError(
                f"HTDemucs prototype requires stereo input; received {decoded.channels} channels"
            )
        return decoded

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

    def _to_channel_first_waveform(self, samples: array, channels: int) -> Any:
        """Reshape flat interleaved samples into a [channels, frames] tensor."""
        import torch

        waveform = torch.tensor(samples, dtype=torch.float32)
        return waveform.view(-1, channels).transpose(0, 1).contiguous()

    def _reduce_vocals(self, waveform: Any, sample_rate: int) -> Any:
        original, stems = self._separator.separate_tensor(waveform, sr=sample_rate)
        vocals = stems.get("vocals")
        if vocals is None:
            raise RuntimeError("Demucs model did not return a vocals stem")

        # 0.0 leaves the original mix unchanged; 1.0 removes the estimated
        # vocal stem completely; intermediate values retain some guide vocal.
        return original - (vocals * self.vocal_reduction)

    def _finalize_output(self, output: Any, sample_rate: int, target_frames: int) -> Any:
        import torch
        import torchaudio.functional as audio_functional

        model_rate = int(self._separator.samplerate)
        if model_rate != sample_rate:
            output = audio_functional.resample(output, model_rate, sample_rate)

        if output.shape[-1] < target_frames:
            output = torch.nn.functional.pad(
                output, (0, target_frames - output.shape[-1])
            )
        output = output[..., :target_frames].clamp(-1.0, 1.0)
        return (
            output.transpose(0, 1)
            .contiguous()
            .view(-1)
            .to(device="cpu", dtype=torch.float32)
        )

    def _run_inference(self, samples: array, sample_rate: int, channels: int) -> array:
        if self._inference_fn is not None:
            return self._inference_fn(samples, sample_rate, channels)
        if self._separator is None:
            raise RuntimeError("Demucs separator is not loaded")

        import torch

        waveform = self._to_channel_first_waveform(samples, channels)
        target_frames = len(samples) // channels
        with torch.inference_mode():
            output = self._reduce_vocals(waveform, sample_rate)
            interleaved = self._finalize_output(output, sample_rate, target_frames)

        return array("f", interleaved.tolist())