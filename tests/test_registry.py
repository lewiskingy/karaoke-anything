import pytest

from audio_trombone.processors.registry import ProcessorRegistry


def test_registry_creates_passthrough() -> None:
    registry = ProcessorRegistry()
    processor = registry.create("passthrough")
    assert processor.name == "passthrough"


def test_registry_rejects_unknown_processor() -> None:
    registry = ProcessorRegistry()
    with pytest.raises(ValueError):
        registry.create("missing")
