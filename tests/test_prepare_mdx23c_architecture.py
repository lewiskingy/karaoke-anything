import ast
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from audio_trombone.tools.prepare_mdx23c_architecture import (
    EXPECTED_IMPORT,
    main,
    prepare,
)


def _prepared_helper(tmp_path: Path):
    architecture = tmp_path / "mdx23c_tfc_tdf_v3.py"
    architecture.write_text(
        f"{EXPECTED_IMPORT}\nclass Model:\n    pass\n", encoding="utf-8"
    )
    prepare(architecture)

    prepared = architecture.read_text(encoding="utf-8")
    assert EXPECTED_IMPORT not in prepared
    tree = ast.parse(prepared)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prefer_target_instrument"
    )
    namespace = {}
    # Executing the just-extracted helper function's own AST is the only way
    # to unit test the source `prepare()` embeds into the architecture file,
    # short of writing and re-importing a temporary module.
    exec(  # noqa: S102
        compile(ast.Module(body=[helper], type_ignores=[]), str(architecture), "exec"),
        namespace,
    )
    return namespace["prefer_target_instrument"]


def test_prefer_target_instrument_uses_present_target(tmp_path):
    helper = _prepared_helper(tmp_path)
    config = SimpleNamespace(
        training=SimpleNamespace(
            target_instrument="Vocals", instruments=["Vocals", "Instrumental"]
        )
    )
    assert helper(config) == ["Vocals"]


def test_prefer_target_instrument_uses_instruments_when_absent(tmp_path):
    helper = _prepared_helper(tmp_path)
    instruments = ["Vocals", "Instrumental"]
    config = SimpleNamespace(training=SimpleNamespace(instruments=instruments))
    assert helper(config) is instruments


@pytest.mark.parametrize("target_instrument", ["", None, False])
def test_prefer_target_instrument_uses_instruments_when_target_is_falsey(
    tmp_path, target_instrument
):
    helper = _prepared_helper(tmp_path)
    instruments = ["Vocals", "Instrumental"]
    config = SimpleNamespace(
        training=SimpleNamespace(
            target_instrument=target_instrument, instruments=instruments
        )
    )
    assert helper(config) is instruments


def test_prepare_rejects_unexpected_upstream_source(tmp_path):
    architecture = tmp_path / "mdx23c_tfc_tdf_v3.py"
    architecture.write_text("class Model:\n    pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="found 0"):
        prepare(architecture)


def test_main_parses_argv_and_prepares_architecture(tmp_path, monkeypatch):
    architecture = tmp_path / "mdx23c_tfc_tdf_v3.py"
    architecture.write_text(
        f"{EXPECTED_IMPORT}\nclass Model:\n    pass\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["prepare_mdx23c_architecture", str(architecture)])

    main()

    assert EXPECTED_IMPORT not in architecture.read_text(encoding="utf-8")


def test_module_runs_as_script(tmp_path, monkeypatch):
    architecture = tmp_path / "mdx23c_tfc_tdf_v3.py"
    architecture.write_text(
        f"{EXPECTED_IMPORT}\nclass Model:\n    pass\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["prepare_mdx23c_architecture", str(architecture)])
    monkeypatch.delitem(
        sys.modules, "audio_trombone.tools.prepare_mdx23c_architecture", raising=False
    )

    runpy.run_module(
        "audio_trombone.tools.prepare_mdx23c_architecture", run_name="__main__"
    )

    assert EXPECTED_IMPORT not in architecture.read_text(encoding="utf-8")
