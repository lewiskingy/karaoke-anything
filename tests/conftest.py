import sys
import types
from collections.abc import Callable

import pytest


def install_fake_torchaudio(
    monkeypatch: pytest.MonkeyPatch, resample: Callable
) -> None:
    """Inject a fake ``torchaudio.functional`` module exposing only ``resample``.

    ``torchaudio`` isn't installed in this environment; model-backed processor
    tests use this to exercise the real resample-call sites in ``_run_inference``
    without depending on it.
    """
    fake_functional = types.ModuleType("torchaudio.functional")
    fake_functional.resample = resample
    fake_torchaudio = types.ModuleType("torchaudio")
    fake_torchaudio.functional = fake_functional
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", fake_functional)
