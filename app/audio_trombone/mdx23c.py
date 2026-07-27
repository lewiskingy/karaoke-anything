"""Repository-owned loading and inference boundary for the pinned MDX23C model."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from audio_trombone.tools.validate_mdx23c import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_config,
    normalise_state_dict,
)


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
        precision: str = "float32",
    ) -> None:
        if precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be float32, float16, or bfloat16")
        import torch

        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.config = load_config(self.config_path)
        self.sample_rate = int(self.config.audio.sample_rate)
        self.stems = tuple(str(item).lower() for item in self.config.training.instruments)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("MDX23C_DEVICE requests CUDA but PyTorch cannot see a GPU")
        if precision != "float32" and not device.startswith("cuda"):
            raise ValueError("reduced MDX23C precision is supported only on CUDA")
        self.device = device
        self.precision = precision

        from audio_trombone.vendor.mdx23c_tfc_tdf_v3 import TFC_TDF_net

        self.model = TFC_TDF_net(self.config)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(normalise_state_dict(checkpoint), strict=True)
        self.model.to(self.device).eval()

    def infer(self, waveform: Any, sample_rate: int) -> Any:
        import torch

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"MDX23C requires {self.sample_rate} Hz input; received {sample_rate} Hz"
            )
        if waveform.ndim != 3 or waveform.shape[1] != 2:
            raise ValueError(
                "MDX23C input must have shape [batch, 2, frames]; "
                f"received {tuple(waveform.shape)}"
            )
        frames = waveform.shape[-1]
        waveform = waveform.to(self.device, dtype=torch.float32)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            self.precision
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if dtype is not None
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            result = self.model(waveform)
        if result.ndim == 3:
            result = result.unsqueeze(1)
        if result.ndim != 4 or result.shape[0] != waveform.shape[0] or result.shape[2] != 2:
            raise RuntimeError(
                "MDX23C output must be [batch, stems, 2, frames]; "
                f"received {tuple(result.shape)}"
            )
        if result.shape[1] != len(self.stems):
            raise RuntimeError(
                f"MDX23C returned {result.shape[1]} stems; YAML declares {len(self.stems)}"
            )
        if result.shape[-1] < frames:
            result = torch.nn.functional.pad(result, (0, frames - result.shape[-1]))
        return result[..., :frames].to(dtype=torch.float32)

    def stem_index(self, *names: str) -> int:
        for name in names:
            if name.lower() in self.stems:
                return self.stems.index(name.lower())
        raise RuntimeError(f"none of {names!r} appears in configured stems {self.stems!r}")
