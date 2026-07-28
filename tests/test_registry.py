import pytest

from audio_trombone.config import Settings
from audio_trombone.processors.registry import ProcessorRegistry


def test_registry_creates_passthrough() -> None:
    registry = ProcessorRegistry()
    processor = registry.create("passthrough")
    assert processor.name == "passthrough"


def test_registry_rejects_unknown_processor() -> None:
    registry = ProcessorRegistry()
    with pytest.raises(ValueError):
        registry.create("missing")


def test_registry_uses_default_settings_when_none_provided() -> None:
    registry = ProcessorRegistry(settings=None)
    processor = registry.create("stereo-centre-reduction")
    assert processor.centre_reduction == Settings().centre_reduction


@pytest.mark.parametrize(
    "name",
    [
        "passthrough",
        "delay-passthrough",
        "null",
        "stereo-centre-reduction",
        "htdemucs-vocals",
        "convtasnet-lyrics-causal",
    ],
)
def test_registry_creates_every_registered_processor(name: str) -> None:
    registry = ProcessorRegistry()
    processor = registry.create(name)
    assert processor.name


def test_registry_describe_lists_all_processors_sorted_with_capabilities() -> None:
    registry = ProcessorRegistry()

    description = registry.describe()

    names = [entry["name"] for entry in description["processors"]]
    assert names == sorted(names)
    assert len(names) == 6

    passthrough_entry = next(
        entry for entry in description["processors"] if entry["name"] == "passthrough"
    )
    assert passthrough_entry["description"]
    assert passthrough_entry["capabilities"] == {
        "passthrough": True,
        "stateful": False,
        "can_buffer": False,
        "changes_payload": False,
    }
