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

def test_registry_describes_mdx23c() -> None:
    assert "mdx23c-vocals" in {item["name"] for item in ProcessorRegistry().describe()["processors"]}
