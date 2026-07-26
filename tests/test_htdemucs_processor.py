from array import array
import asyncio

import pytest

from audio_trombone.kany import HEADER_SIZE, KanyPacket
from audio_trombone.models import MediaPacket
from audio_trombone.processors.htdemucs import HTDemucsProcessor


def make_media_packet(sequence: int, samples: list[float]) -> MediaPacket:
    channels = 2
    frames = len(samples) // channels
    header = bytearray(HEADER_SIZE)
    header[0:4] = b"KANY"
    header[4] = 1
    header[6] = channels
    header[7] = 1
    header[8:12] = (1_000).to_bytes(4, "big")
    header[12:16] = sequence.to_bytes(4, "big")
    header[16:24] = (sequence * 2_000).to_bytes(8, "big")
    header[24:26] = frames.to_bytes(2, "big")
    payload = bytes(header) + array("f", samples).tobytes()
    return MediaPacket.received(payload, "127.0.0.1", 40_000)


async def collect(processor: HTDemucsProcessor, packet: MediaPacket):
    return [output async for output in processor.process(packet)]


def test_vocal_reduction_is_reported_in_diagnostics() -> None:
    processor = HTDemucsProcessor(vocal_reduction=0.5, inference_fn=lambda s, _r, _c: s)

    diagnostics = processor.diagnostics()

    assert diagnostics["vocal_reduction"] == pytest.approx(0.5)
    assert diagnostics["vocal_gain"] == pytest.approx(0.5)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_vocal_reduction_must_be_between_zero_and_one(value: float) -> None:
    with pytest.raises(ValueError, match="vocal_reduction"):
        HTDemucsProcessor(vocal_reduction=value)


@pytest.mark.asyncio
async def test_buffers_inference_and_releases_one_packet_per_input() -> None:
    def remove_everything(samples: array, sample_rate: int, channels: int) -> array:
        assert sample_rate == 1_000
        assert channels == 2
        return array("f", [0.0] * len(samples))

    processor = HTDemucsProcessor(
        segment_seconds=0.004,
        inference_fn=remove_everything,
    )
    await processor.start()

    assert await collect(processor, make_media_packet(0, [0.5, 0.5, 0.5, 0.5])) == []
    assert await collect(processor, make_media_packet(1, [0.5, 0.5, 0.5, 0.5])) == []

    for _ in range(50):
        await asyncio.sleep(0.002)
        if processor.diagnostics()["inference_running"] is False:
            break

    first = await collect(processor, make_media_packet(2, [0.5, 0.5, 0.5, 0.5]))
    second = await collect(processor, make_media_packet(3, [0.5, 0.5, 0.5, 0.5]))

    assert len(first) == 1
    assert len(second) == 1
    assert KanyPacket.decode(first[0].payload).sequence == 0
    assert KanyPacket.decode(second[0].payload).sequence == 1
    assert list(KanyPacket.decode(first[0].payload).samples) == pytest.approx([0.0] * 4)
    assert processor.diagnostics()["segments_completed"] == 1

    await processor.stop()