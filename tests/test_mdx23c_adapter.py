import sys
import types
from pathlib import Path

import pytest

from audio_trombone.mdx23c import MDX23CAdapter, _autocast_context


class _FakeModel:
    def __init__(self, config):
        self.config = config
        self.loaded_state = None
        self.device = None
        self.forward_fn = None

    def load_state_dict(self, state, strict=True):
        self.loaded_state = state

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, waveform):
        return self.forward_fn(waveform)


def _install_fake_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("audio_trombone.vendor.mdx23c_tfc_tdf_v3")
    fake_module.TFC_TDF_net = _FakeModel
    monkeypatch.setitem(
        sys.modules, "audio_trombone.vendor.mdx23c_tfc_tdf_v3", fake_module
    )


def _write_config(tmp_path: Path, instruments: list[str]) -> Path:
    config_path = tmp_path / "config.yaml"
    joined = ", ".join(instruments)
    config_path.write_text(
        f"training:\n  instruments: [{joined}]\naudio:\n  sample_rate: 8000\n",
        encoding="utf-8",
    )
    return config_path


def _write_checkpoint(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.write_bytes(b"")
    return checkpoint_path


def _build_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    instruments: list[str] | None = None,
    device: str = "cpu",
) -> MDX23CAdapter:
    import torch

    _install_fake_vendor(monkeypatch)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"weight": object()})
    config_path = _write_config(tmp_path, instruments or ["vocals", "instrumental"])
    checkpoint_path = _write_checkpoint(tmp_path)
    return MDX23CAdapter(config_path, checkpoint_path, device=device, precision="float32")


def test_rejects_unsupported_precision() -> None:
    with pytest.raises(ValueError, match="precision must be"):
        MDX23CAdapter(precision="int8")


def test_rejects_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config_path = _write_config(tmp_path, ["vocals", "instrumental"])
    with pytest.raises(RuntimeError, match="cannot see a GPU"):
        MDX23CAdapter(config_path, tmp_path / "missing.ckpt", device="cuda")


def test_rejects_reduced_precision_without_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, ["vocals", "instrumental"])
    with pytest.raises(ValueError, match="supported only on CUDA"):
        MDX23CAdapter(
            config_path, tmp_path / "missing.ckpt", device="cpu", precision="float16"
        )


def test_constructor_auto_selects_cpu_and_loads_checkpoint_strictly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    adapter = _build_adapter(monkeypatch, tmp_path, device="auto")

    assert adapter.device == "cpu"
    assert adapter.sample_rate == 8000
    assert adapter.stems == ("vocals", "instrumental")
    assert list(adapter.model.loaded_state.keys()) == ["weight"]
    assert adapter.model.device == "cpu"


def test_infer_rejects_sample_rate_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    adapter = _build_adapter(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="requires 8000 Hz input"):
        adapter.infer(torch.zeros(1, 2, 10), 44_100)


def test_infer_rejects_wrong_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    adapter = _build_adapter(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="must have shape"):
        adapter.infer(torch.zeros(2, 10), 8000)


def test_infer_promotes_single_stem_output_and_returns_full_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    adapter = _build_adapter(monkeypatch, tmp_path, instruments=["vocals"])
    adapter.model.forward_fn = lambda waveform: waveform.clone()

    output = adapter.infer(torch.zeros(1, 2, 20), 8000)

    assert output.shape == (1, 1, 2, 20)


def test_infer_rejects_wrong_output_rank_or_channels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    adapter = _build_adapter(monkeypatch, tmp_path)
    adapter.model.forward_fn = lambda waveform: torch.zeros(1, 2, 3, waveform.shape[-1])

    with pytest.raises(RuntimeError, match="must be \\[batch, stems, 2, frames\\]"):
        adapter.infer(torch.zeros(1, 2, 20), 8000)


def test_infer_rejects_stem_count_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    adapter = _build_adapter(monkeypatch, tmp_path)  # 2 stems declared
    adapter.model.forward_fn = lambda waveform: torch.zeros(
        1, 3, 2, waveform.shape[-1]
    )

    with pytest.raises(RuntimeError, match="returned 3 stems"):
        adapter.infer(torch.zeros(1, 2, 20), 8000)


def test_infer_pads_short_output_to_input_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    adapter = _build_adapter(monkeypatch, tmp_path)
    adapter.model.forward_fn = lambda waveform: torch.ones(
        1, 2, 2, waveform.shape[-1] - 3
    )

    output = adapter.infer(torch.zeros(1, 2, 20), 8000)

    assert output.shape == (1, 2, 2, 20)
    assert output.dtype == torch.float32


def test_stem_index_finds_matching_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _build_adapter(monkeypatch, tmp_path, instruments=["vocals", "other"])
    assert adapter.stem_index("instrumental", "other") == 1


def test_stem_index_raises_when_nothing_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _build_adapter(monkeypatch, tmp_path, instruments=["vocals", "other"])
    with pytest.raises(RuntimeError, match="none of"):
        adapter.stem_index("instrumental", "accompaniment")


def test_autocast_context_is_null_for_float32() -> None:
    import contextlib

    assert isinstance(_autocast_context("float32"), contextlib.nullcontext)


def test_autocast_context_selects_dtype_for_reduced_precision() -> None:
    import torch

    context = _autocast_context("float16")

    assert isinstance(context, torch.autocast)
