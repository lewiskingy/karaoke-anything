from array import array

import pytest
from audio_trombone.kany import HEADER_SIZE, KanyPacket
from audio_trombone.models import MediaPacket
from audio_trombone.processors.centre_reduction import StereoCentreReductionProcessor


def make_media_packet(samples: list[float], channels: int = 2) -> MediaPacket:
    frames = len(samples) // channels
    header = bytearray(HEADER_SIZE)
    header[0:4] = b"KANY"
    header[4] = 1
    header[6] = channels
    header[7] = 1
    header[8:12] = (48_000).to_bytes(4, "big")
    header[12:16] = (1).to_bytes(4, "big")
    header[16:24] = (0).to_bytes(8, "big")
    header[24:26] = frames.to_bytes(2, "big")
    payload = bytes(header) + array("f", samples).tobytes()
    return MediaPacket.received(payload, "127.0.0.1", 40_000)


async def collect(processor: StereoCentreReductionProcessor, packet: MediaPacket):
    return [output async for output in processor.process(packet)]


@pytest.mark.asyncio
async def test_full_reduction_removes_identical_centre_signal() -> None:
    processor = StereoCentreReductionProcessor(centre_reduction=1.0)
    outputs = await collect(processor, make_media_packet([0.5, 0.5, -0.25, -0.25]))

    decoded = KanyPacket.decode(outputs[0].payload)
    assert list(decoded.samples) == pytest.approx([0.0, 0.0, 0.0, 0.0])


@pytest.mark.asyncio
async def test_partial_reduction_retains_requested_centre_gain() -> None:
    processor = StereoCentreReductionProcessor(centre_reduction=0.5)
    outputs = await collect(processor, make_media_packet([0.8, 0.8]))

    decoded = KanyPacket.decode(outputs[0].payload)
    assert list(decoded.samples) == pytest.approx([0.4, 0.4])


@pytest.mark.asyncio
async def test_side_signal_is_preserved() -> None:
    processor = StereoCentreReductionProcessor(centre_reduction=1.0)
    outputs = await collect(processor, make_media_packet([0.6, -0.6]))

    decoded = KanyPacket.decode(outputs[0].payload)
    assert list(decoded.samples) == pytest.approx([0.6, -0.6])


def test_reduction_must_be_bounded() -> None:
    with pytest.raises(ValueError):
        StereoCentreReductionProcessor(centre_reduction=1.1)


@pytest.mark.asyncio
async def test_rejects_non_kany_payload() -> None:
    processor = StereoCentreReductionProcessor()
    packet = MediaPacket.received(b"not-a-kany-packet", "127.0.0.1", 40_000)

    with pytest.raises(ValueError, match="requires KANY v1 f32 PCM packets"):
        await collect(processor, packet)


@pytest.mark.asyncio
async def test_rejects_non_stereo_input() -> None:
    processor = StereoCentreReductionProcessor()
    packet = make_media_packet([0.5, 0.25], channels=1)

    with pytest.raises(ValueError, match="requires stereo input"):
        await collect(processor, packet)


def test_diagnostics_reports_reduction_and_gain() -> None:
    processor = StereoCentreReductionProcessor(centre_reduction=0.3)

    assert processor.diagnostics() == {
        "centre_reduction": 0.3,
        "centre_gain": pytest.approx(0.7),
        "algorithmic_lookahead_ms": 0.0,
    }
