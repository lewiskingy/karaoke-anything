from collections.abc import AsyncIterator

import pytest
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor, ProcessorCapabilities


class ConcreteProcessor(AudioProcessor):
    name = "concrete"
    description = "minimal concrete processor for exercising base defaults"
    capabilities = ProcessorCapabilities(
        passthrough=False,
        stateful=False,
        can_buffer=False,
        changes_payload=False,
    )

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        yield ProcessedPacket(payload=packet.payload)


def make_packet() -> MediaPacket:
    return MediaPacket.received(payload=b"abc", sender_host="127.0.0.1", sender_port=1)


@pytest.mark.asyncio
async def test_default_start_stop_reset_return_none() -> None:
    processor = ConcreteProcessor()
    assert await processor.start() is None
    assert await processor.stop() is None
    assert await processor.reset() is None


def test_default_diagnostics_returns_empty_dict() -> None:
    processor = ConcreteProcessor()
    assert processor.diagnostics() == {}


@pytest.mark.asyncio
async def test_default_flush_yields_nothing() -> None:
    processor = ConcreteProcessor()
    results = [item async for item in processor.flush()]
    assert results == []


@pytest.mark.asyncio
async def test_base_process_is_not_implemented() -> None:
    processor = ConcreteProcessor()
    with pytest.raises(NotImplementedError):
        await AudioProcessor.process(processor, make_packet())
