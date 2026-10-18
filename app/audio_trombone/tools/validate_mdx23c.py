"""Construct an MDX23C model and strictly validate its checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from audio_trombone.mdx23c_model_yaml import DEFAULT_CONFIG, load_config

DEFAULT_CHECKPOINT = Path("/models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt")
SUPPORTED_WRAPPERS = ("state_dict", "model")
SUPPORTED_PREFIX = "module."


def _unwrap_checkpoint(checkpoint: Mapping) -> Any:
    wrapper_keys = [key for key in SUPPORTED_WRAPPERS if key in checkpoint]
    if len(wrapper_keys) > 1:
        raise ValueError(f"ambiguous checkpoint wrappers: {wrapper_keys}")
    if wrapper_keys:
        return checkpoint[wrapper_keys[0]]
    return checkpoint


def _validate_state_mapping(state: Any) -> None:
    is_non_empty_mapping = isinstance(state, Mapping) and state
    if not is_non_empty_mapping:
        raise ValueError("checkpoint state dictionary must be a non-empty mapping")
    if not all(isinstance(key, str) for key in state):
        raise TypeError("checkpoint state dictionary keys must be strings")


def _strip_module_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    prefix_variants = {key.startswith(SUPPORTED_PREFIX) for key in state}
    if prefix_variants == {True, False}:
        raise ValueError("mixed 'module.' prefixes are not supported")
    if prefix_variants == {True}:
        return {
            key.removeprefix(SUPPORTED_PREFIX): value for key, value in state.items()
        }
    return dict(state)


def normalise_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Accept only documented wrappers and a uniform DataParallel prefix."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a state dictionary mapping")

    state = _unwrap_checkpoint(checkpoint)
    _validate_state_mapping(state)
    return _strip_module_prefix(state)


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
