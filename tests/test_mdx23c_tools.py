import runpy
import struct
import sys
import wave
from pathlib import Path

import pytest


def _write_wav(path: Path, *, channels: int = 2) -> None:
    frames, rate = 10, 8_000
    sample_count = frames * channels
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(struct.pack(f"<{sample_count}h", *([1_000] * sample_count)))


class _FakeAdapterForInfer:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self.stems = ("vocals", "instrumental")

    def infer(self, waveform, rate):
        return waveform.unsqueeze(1).repeat(1, 2, 1, 1)


def test_infer_writes_stem_wavs_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    input_wav = tmp_path / "input.wav"
    _write_wav(input_wav)
    output_dir = tmp_path / "out"

    monkeypatch.setattr("audio_trombone.tools.infer_mdx23c.MDX23CAdapter", _FakeAdapterForInfer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["infer_mdx23c", str(input_wav), "--output-dir", str(output_dir), "--device", "cpu"],
    )

    from audio_trombone.tools import infer_mdx23c

    infer_mdx23c.main()

    assert (output_dir / "vocals.wav").exists()
    assert (output_dir / "instrumental.wav").exists()
    output = capsys.readouterr().out
    assert "input:" in output
    assert "output:" in output


def test_infer_rejects_non_stereo_16bit_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mono_wav = tmp_path / "mono.wav"
    _write_wav(mono_wav, channels=1)

    monkeypatch.setattr("audio_trombone.tools.infer_mdx23c.MDX23CAdapter", _FakeAdapterForInfer)
    monkeypatch.setattr(sys, "argv", ["infer_mdx23c", str(mono_wav)])

    from audio_trombone.tools import infer_mdx23c

    with pytest.raises(ValueError, match="stereo 16-bit"):
        infer_mdx23c.main()


def test_infer_module_runs_as_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    input_wav = tmp_path / "input.wav"
    _write_wav(input_wav)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["infer_mdx23c", str(input_wav), "--output-dir", str(output_dir), "--device", "cpu"],
    )
    monkeypatch.delitem(sys.modules, "audio_trombone.tools.infer_mdx23c", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "audio_trombone.mdx23c",
        type(sys)("audio_trombone.mdx23c"),
    )
    sys.modules["audio_trombone.mdx23c"].MDX23CAdapter = _FakeAdapterForInfer

    runpy.run_module("audio_trombone.tools.infer_mdx23c", run_name="__main__")

    assert (output_dir / "vocals.wav").exists()


class _FakeAdapterForBenchmark:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self.sample_rate = 8_000
        self.precision = "float32"

    def infer(self, waveform, sample_rate):
        return waveform


def _patch_cuda_benchmark_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr("audio_trombone.mdx23c.MDX23CAdapter", _FakeAdapterForBenchmark)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1_234)
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_mdx23c", "--device", "cuda", "--segments", "0.25", "--overlaps", "0"],
    )


def test_benchmark_prints_report_with_cuda_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _patch_cuda_benchmark_env(monkeypatch)

    from audio_trombone.tools import benchmark_mdx23c

    benchmark_mdx23c.main()

    output = capsys.readouterr().out
    assert '"device": "cuda"' in output
    assert '"peak_cuda_bytes": 1234' in output
    assert "segment=0.25s" in output


def test_benchmark_module_runs_as_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _patch_cuda_benchmark_env(monkeypatch)
    monkeypatch.delitem(sys.modules, "audio_trombone.tools.benchmark_mdx23c", raising=False)

    runpy.run_module("audio_trombone.tools.benchmark_mdx23c", run_name="__main__")

    output = capsys.readouterr().out
    assert '"device": "cuda"' in output
