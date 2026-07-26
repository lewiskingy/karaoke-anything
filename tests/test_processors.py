import pytest

from audio_trombone.models import MediaPacket
from audio_trombone.processors.delay import DelayPassthroughProcessor
from audio_trombone.processors.null import NullProcessor
from audio_trombone.processors.passthrough import PassthroughProcessor


def make_packet(payload: bytes = b"test") -> MediaPacket:
    return MediaPacket.received(
        payload=payload,
        sender_host="127.0.0.1",
        sender_port=40000,
    )


@pytest.mark.asyncio
async def test_passthrough_returns_identical_payload() -> None:
    processor = PassthroughProcessor()
    outputs = [output async for output in processor.process(make_packet(b"abc"))]
    assert len(outputs) == 1
    assert outputs[0].payload == b"abc"


@pytest.mark.asyncio
async def test_delay_passthrough_returns_identical_payload() -> None:
    processor = DelayPassthroughProcessor(delay_ms=0)
    outputs = [output async for output in processor.process(make_packet(b"abc"))]
    assert len(outputs) == 1
    assert outputs[0].payload == b"abc"


@pytest.mark.asyncio
async def test_null_processor_emits_nothing() -> None:
    processor = NullProcessor()
    outputs = [output async for output in processor.process(make_packet())]
    assert outputs == []
