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
class ConvTasNetLyricsConfig:
    """Tuning parameters for `ConvTasNetLyricsProcessor`, grouped into one
    object so the constructor doesn't take an argument per field."""

    model_path: str = "/models/convtasnet-lyrics-causal"
    device: str = "auto"
    segment_seconds: float = 1.0
    vocal_reduction: float = 1.0
    vocal_source_index: int = 0
    accompaniment_source_index: int = 1

    def __post_init__(self) -> None:
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be greater than zero")
        if not 0 <= self.vocal_reduction <= 1:
            raise ValueError("vocal_reduction must be between 0.0 and 1.0")
        if self.vocal_source_index < 0 or self.accompaniment_source_index < 0:
            raise ValueError("source indexes must be zero or greater")


class ConvTasNetLyricsProcessor(SegmentedInferenceProcessor):
    """Buffered causal Conv-TasNet lyrics/accompaniment separator.

    The model itself is causal, but this first integration deliberately reuses the
    proven packet-buffering and paced-output behaviour of the HTDemucs processor.
    That keeps the network path stable while allowing the segment duration to be
    reduced independently as real-time performance is measured.
    """

    _log_name = "ConvTasNet"
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
        config: ConvTasNetLyricsConfig | None = None,
        inference_fn: InferenceFunction | None = None,
    ) -> None:
        config = config or ConvTasNetLyricsConfig()
        super().__init__(segment_seconds=config.segment_seconds, inference_fn=inference_fn)

        self.model_path = config.model_path
        self.requested_device = config.device
        self.vocal_reduction = config.vocal_reduction
        self.vocal_source_index = config.vocal_source_index
        self.accompaniment_source_index = config.accompaniment_source_index
        self._model: Any | None = None
        self._device = config.device
        self._model_sample_rate = 44100

    async def start(self) -> None:
        if self._inference_fn is not None:
            logger.info("ConvTasNet processor using injected inference function")
            return
        await asyncio.to_thread(self._load_model)

    async def stop(self) -> None:
        await self.reset()
        self._model = None

    def _decode_and_validate(self, payload: bytes) -> KanyPacket:
        try:
            decoded = KanyPacket.decode(payload)
        except KanyProtocolError as exc:
            raise ValueError(
                f"ConvTasNet requires KANY v1 f32 PCM packets: {exc}"
            ) from exc

        if decoded.channels != 2:
            raise ValueError(
                "ConvTasNet lyrics processor requires stereo input; "
                f"received {decoded.channels} channels"
            )
        return decoded

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

    def _prepare_waveform(self, samples: array, sample_rate: int, channels: int) -> Any:
        import torch
        import torchaudio.functional as audio_functional

        waveform = torch.tensor(samples, dtype=torch.float32)
        waveform = waveform.view(-1, channels).transpose(0, 1).contiguous().unsqueeze(0)
        if sample_rate != self._model_sample_rate:
            waveform = audio_functional.resample(
                waveform, sample_rate, self._model_sample_rate
            )
        return waveform.to(self._device)

    def _separate_accompaniment(self, waveform: Any) -> Any:
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

        accompaniment = estimates[0, self.accompaniment_source_index]
        return accompaniment * self.vocal_reduction + waveform[0] * (1.0 - self.vocal_reduction)

    def _finalize_output(self, output: Any, sample_rate: int, target_frames: int) -> Any:
        import torch
        import torchaudio.functional as audio_functional

        if self._model_sample_rate != sample_rate:
            output = audio_functional.resample(
                output, self._model_sample_rate, sample_rate
            )

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
        if self._model is None:
            raise RuntimeError("ConvTasNet model is not loaded")

        import torch

        waveform = self._prepare_waveform(samples, sample_rate, channels)
        target_frames = len(samples) // channels
        with torch.inference_mode():
            output = self._separate_accompaniment(waveform)
            interleaved = self._finalize_output(output, sample_rate, target_frames)

        return array("f", interleaved.tolist())
