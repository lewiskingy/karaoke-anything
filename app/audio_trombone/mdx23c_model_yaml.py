"""Load MDX23C YAML configs into attribute-accessible mappings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("/models/mdx23c/model_2_stem_full_band_8k.yaml")


class YamlConfig(dict):
    """Recursive mapping with the attribute access expected by upstream MDX23C."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return YamlConfig({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def load_config(path: Path) -> YamlConfig:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError("MDX23C config must be a YAML mapping")
    return _wrap(document)
