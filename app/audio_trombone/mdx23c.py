"""Repository-owned loading and inference boundary for the pinned MDX23C model."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, get_args

from audio_trombone.mdx23c_model_yaml import DEFAULT_CONFIG, load_config
from audio_trombone.tools.validate_mdx23c import (
    DEFAULT_CHECKPOINT,
    normalise_state_dict,
)

MDX23CPrecision = Literal["float32", "float16", "bfloat16"]
_SUPPORTED_PRECISIONS = set(get_args(MDX23CPrecision))


def _resolve_device(requested_device: str, precision: MDX23CPrecision) -> str:
    """Applies the "auto" device default and the CUDA-only precision rule."""
    import torch

    device = requested_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MDX23C_DEVICE requests CUDA but PyTorch cannot see a GPU")
    if precision != "float32" and not device.startswith("cuda"):
        raise ValueError("reduced MDX23C precision is supported only on CUDA")
    return device


def _load_model(config: Any, checkpoint_path: Path, device: str) -> Any:
    """Constructs the pinned architecture and strictly loads the checkpoint onto it."""
    import torch
    from audio_trombone.vendor.mdx23c_tfc_tdf_v3 import TFC_TDF_net

    model = TFC_TDF_net(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(normalise_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


def _autocast_context(precision: MDX23CPrecision):
    import torch

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _has_valid_stem_shape(result: Any, waveform: Any) -> bool:
    """True when `result` is `[batch, stems, 2, frames]` for the given input batch."""
    return (
        result.ndim == 4
        and result.shape[0] == waveform.shape[0]
        and result.shape[2] == 2
    )


def _validate_waveform_input(
    waveform: Any, sample_rate: int, expected_rate: int
) -> None:
    if sample_rate != expected_rate:
        raise ValueError(
            f"MDX23C requires {expected_rate} Hz input; received {sample_rate} Hz"
        )
    if waveform.ndim != 3 or waveform.shape[1] != 2:
        raise ValueError(
            "MDX23C input must have shape [batch, 2, frames]; "
            f"received {tuple(waveform.shape)}"
        )


def _normalise_output(result: Any, waveform: Any, frames: int, stem_count: int) -> Any:
    import torch

    if result.ndim == 3:
        result = result.unsqueeze(1)
    if not _has_valid_stem_shape(result, waveform):
        raise RuntimeError(
            "MDX23C output must be [batch, stems, 2, frames]; "
            f"received {tuple(result.shape)}"
        )
    if result.shape[1] != stem_count:
        raise RuntimeError(
            f"MDX23C returned {result.shape[1]} stems; YAML declares {stem_count}"
        )
    if result.shape[-1] < frames:
        result = torch.nn.functional.pad(result, (0, frames - result.shape[-1]))
    return result[..., :frames].to(dtype=torch.float32)


class MDX23CAdapter:
    """Strict, offline MDX23C loader with an explicit tensor contract.

    Input is ``[batch, channel, frame]`` stereo PCM at the YAML sample rate.
    Output is normalised to ``[batch, stem, channel, frame]`` and trimmed to
    the input length.  Channel order is never exchanged.
    """

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        *,
        device: str = "auto",
        precision: MDX23CPrecision = "float32",
    ) -> None:
        if precision not in _SUPPORTED_PRECISIONS:
            raise ValueError("precision must be float32, float16, or bfloat16")

        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.config = load_config(self.config_path)
        self.sample_rate = int(self.config.audio.sample_rate)
        self.stems = tuple(
            str(item).lower() for item in self.config.training.instruments
        )
        self.device = _resolve_device(device, precision)
        self.precision = precision
        self.model = _load_model(self.config, self.checkpoint_path, self.device)

    def infer(self, waveform: Any, sample_rate: int) -> Any:
        import torch

        _validate_waveform_input(waveform, sample_rate, self.sample_rate)
        frames = waveform.shape[-1]
        waveform = waveform.to(self.device, dtype=torch.float32)

        with torch.inference_mode(), _autocast_context(self.precision):
            result = self.model(waveform)

        return _normalise_output(result, waveform, frames, len(self.stems))

    def stem_index(self, *names: str) -> int:
        for name in names:
            if name.lower() in self.stems:
                return self.stems.index(name.lower())
        raise RuntimeError(
            f"none of {names!r} appears in configured stems {self.stems!r}"
        )
