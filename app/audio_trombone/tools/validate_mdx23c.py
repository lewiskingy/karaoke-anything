"""Construct an MDX23C model and strictly validate its checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("/models/mdx23c/model_2_stem_full_band_8k.yaml")
DEFAULT_CHECKPOINT = Path("/models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt")
SUPPORTED_WRAPPERS = ("state_dict", "model")
SUPPORTED_PREFIX = "module."


class Config(dict):
    """Recursive mapping with the attribute access expected by upstream MDX23C."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config({key: _config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_config(item) for item in value]
    return value


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError("MDX23C config must be a YAML mapping")
    return _config(document)


def normalise_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Accept only documented wrappers and a uniform DataParallel prefix."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a state dictionary mapping")

    wrapper_keys = [key for key in SUPPORTED_WRAPPERS if key in checkpoint]
    if len(wrapper_keys) > 1:
        raise ValueError(f"ambiguous checkpoint wrappers: {wrapper_keys}")
    if wrapper_keys:
        state = checkpoint[wrapper_keys[0]]
    else:
        state = checkpoint
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint state dictionary must be a non-empty mapping")
    if not all(isinstance(key, str) for key in state):
        raise TypeError("checkpoint state dictionary keys must be strings")

    prefixed = [key.startswith(SUPPORTED_PREFIX) for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("mixed 'module.' prefixes are not supported")
    if all(prefixed):
        state = {key.removeprefix(SUPPORTED_PREFIX): value for key, value in state.items()}
    return dict(state)


def validate(config_path: Path, checkpoint_path: Path) -> None:
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    import torch

    config = load_config(config_path)
    # This single upstream file is copied into the package by Dockerfile.demucs.
    from audio_trombone.vendor.mdx23c_tfc_tdf_v3 import TFC_TDF_net

    model = TFC_TDF_net(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = normalise_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    targets = config.training.instruments
    sample_rate = config.audio.sample_rate
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("MDX23C checkpoint validation")
    print(f"config: {config_path.resolve()}")
    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(f"model: {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"parameters: {parameters}")
    print(f"sample_rate: {sample_rate}")
    print(f"targets: {', '.join(targets)}")
    print("strict checkpoint load: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    validate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
