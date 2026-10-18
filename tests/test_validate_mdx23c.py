import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from audio_trombone.mdx23c_model_yaml import YamlConfig, _wrap, load_config
from audio_trombone.tools import validate_mdx23c
from audio_trombone.tools.validate_mdx23c import main, normalise_state_dict, validate


def test_accepts_raw_state_dict():
    tensor = object()
    assert normalise_state_dict({"weight": tensor}) == {"weight": tensor}


@pytest.mark.parametrize("wrapper", ["state_dict", "model"])
def test_accepts_documented_wrapper_and_uniform_module_prefix(wrapper):
    tensor = object()
    result = normalise_state_dict({wrapper: {"module.weight": tensor}})
    assert result == {"weight": tensor}


def test_rejects_ambiguous_wrappers():
    with pytest.raises(ValueError, match="ambiguous"):
        normalise_state_dict({"state_dict": {"a": 1}, "model": {"a": 1}})


def test_rejects_mixed_module_prefixes():
    with pytest.raises(ValueError, match="mixed"):
        normalise_state_dict({"module.weight": 1, "bias": 2})


def test_rejects_empty_state_dict():
    with pytest.raises(ValueError, match="non-empty"):
        normalise_state_dict({})


def test_rejects_non_mapping_checkpoint():
    with pytest.raises(TypeError, match="state dictionary mapping"):
        normalise_state_dict(["not", "a", "mapping"])


def test_rejects_non_string_keys():
    with pytest.raises(TypeError, match="must be strings"):
        normalise_state_dict({1: "value"})


def test_config_getattr_returns_present_key():
    config = YamlConfig({"training": {"instruments": ["vocals"]}})
    assert config["training"] == {"instruments": ["vocals"]}
    assert config.training == {"instruments": ["vocals"]}


def test_config_getattr_raises_attribute_error_for_missing_key():
    config = YamlConfig({})
    with pytest.raises(AttributeError, match="missing"):
        config.missing  # noqa: B018 -- the attribute access is the thing under test


def test_config_recursively_wraps_nested_mappings_and_lists():
    wrapped = _wrap(
        {
            "training": {"instruments": ["vocals", "instrumental"]},
            "layers": [{"kind": "conv"}, {"kind": "norm"}],
            "sample_rate": 44_100,
        }
    )

    assert isinstance(wrapped, YamlConfig)
    assert isinstance(wrapped.training, YamlConfig)
    assert wrapped.training.instruments == ["vocals", "instrumental"]
    assert isinstance(wrapped.layers, list)
    assert all(isinstance(layer, YamlConfig) for layer in wrapped.layers)
    assert wrapped.layers[0].kind == "conv"
    assert wrapped.sample_rate == 44_100


def test_load_config_parses_yaml_mapping(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "training:\n  instruments: [vocals, instrumental]\naudio:\n  sample_rate: 44100\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, YamlConfig)
    assert config.training.instruments == ["vocals", "instrumental"]
    assert config.audio.sample_rate == 44_100


def test_load_config_rejects_non_mapping_document(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)


class _FakeModel:
    instances: ClassVar[list["_FakeModel"]] = []

    def __init__(self, config) -> None:
        self.config = config
        self.loaded_state: dict | None = None
        self.strict: bool | None = None
        _FakeModel.instances.append(self)

    def load_state_dict(self, state, strict: bool = True) -> None:
        self.loaded_state = state
        self.strict = strict

    def eval(self) -> None:
        pass

    def parameters(self):
        return [SimpleNamespace(numel=lambda: 5), SimpleNamespace(numel=lambda: 7)]


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "training:\n  instruments: [vocals, instrumental]\naudio:\n  sample_rate: 44100\n",
        encoding="utf-8",
    )
    return config_path


def _stub_vendor_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(TFC_TDF_net=_FakeModel)
    monkeypatch.setitem(
        sys.modules, "audio_trombone.vendor.mdx23c_tfc_tdf_v3", fake_module
    )


def test_validate_rejects_missing_config(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="config not found"):
        validate(tmp_path / "missing.yaml", checkpoint_path)


def test_validate_rejects_missing_checkpoint(tmp_path: Path):
    config_path = _write_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        validate(config_path, tmp_path / "missing.ckpt")


def test_validate_loads_checkpoint_strictly_and_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    config_path = _write_config(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.write_bytes(b"")

    _stub_vendor_module(monkeypatch)

    import torch

    monkeypatch.setattr(torch, "load", lambda *a, **k: {"weight": object()})

    _FakeModel.instances.clear()
    validate(config_path, checkpoint_path)

    assert _FakeModel.instances[-1].strict is True

    output = capsys.readouterr().out
    assert "MDX23C checkpoint validation" in output
    assert "parameters: 12" in output
    assert "sample_rate: 44100" in output
    assert "targets: vocals, instrumental" in output
    assert "strict checkpoint load: OK" in output


def test_main_invokes_validate_with_parsed_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = _write_config(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.write_bytes(b"")

    captured = {}

    def fake_validate(config_arg: Path, checkpoint_arg: Path) -> None:
        captured["config"] = config_arg
        captured["checkpoint"] = checkpoint_arg

    monkeypatch.setattr(validate_mdx23c, "validate", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_mdx23c",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ],
    )

    main()

    assert captured["config"] == config_path
    assert captured["checkpoint"] == checkpoint_path


def test_module_runs_as_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_config(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.write_bytes(b"")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_mdx23c",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
        ],
    )
    monkeypatch.delitem(
        sys.modules, "audio_trombone.tools.validate_mdx23c", raising=False
    )
    _stub_vendor_module(monkeypatch)

    import torch

    monkeypatch.setattr(torch, "load", lambda *a, **k: {"weight": object()})

    _FakeModel.instances.clear()
    runpy.run_module("audio_trombone.tools.validate_mdx23c", run_name="__main__")

    assert _FakeModel.instances[-1].strict is True
