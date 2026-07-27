"""Make the pinned upstream MDX23C architecture minimally self-contained."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_IMPORT = "from utils.model_utils import prefer_target_instrument\n"
LOCAL_HELPER = '''def prefer_target_instrument(config):
    if getattr(config.training, "target_instrument", None):
        return [config.training.target_instrument]
    return config.training.instruments
'''


def prepare(path: Path) -> None:
    """Replace the single expected upstream utility import with its exact semantics."""
    source = path.read_text(encoding="utf-8")
    occurrences = source.count(EXPECTED_IMPORT)
    if occurrences != 1:
        raise RuntimeError(
            "expected exactly one MDX23C prefer_target_instrument import, "
            f"found {occurrences}"
        )
    path.write_text(
        source.replace(EXPECTED_IMPORT, f"{LOCAL_HELPER}\n", 1), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", type=Path)
    args = parser.parse_args()
    prepare(args.architecture)


if __name__ == "__main__":
    main()
