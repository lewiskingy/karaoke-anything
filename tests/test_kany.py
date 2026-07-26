from array import array

import pytest

from audio_trombone.kany import HEADER_SIZE, KanyPacket, KanyProtocolError


def make_payload(samples: array, *, channels: int = 2, sample_rate: int = 48_000) -> bytes:
    frames = len(samples) // channels
    header = bytearray(HEADER_SIZE)
    header[0:4] = b"KANY"
    header[4] = 1
    header[6] = channels
    header[7] = 1
    header[8:12] = sample_rate.to_bytes(4, "big")
    header[12:16] = (42).to_bytes(4, "big")
    header[16:24] = (123_456).to_bytes(8, "big")
    header[24:26] = frames.to_bytes(2, "big")
    return bytes(header) + samples.tobytes()


def test_decode_and_reencode_samples() -> None:
    original = array("f", [0.0, 0.25, -0.5, 1.0])
    packet = KanyPacket.decode(make_payload(original))

    assert packet.channels == 2
    assert packet.sample_rate == 48_000
    assert packet.sequence == 42
    assert packet.timestamp_us == 123_456
    assert packet.frames == 2
    assert list(packet.samples) == pytest.approx(list(original))

    replacement = array("f", [0.1, 0.2, 0.3, 0.4])
    encoded = packet.encode_samples(replacement)
    decoded = KanyPacket.decode(encoded)
    assert list(decoded.samples) == pytest.approx(list(replacement))
    assert decoded.sequence == packet.sequence
    assert decoded.timestamp_us == packet.timestamp_us


def test_rejects_non_kany_payload() -> None:
    with pytest.raises(KanyProtocolError, match="incorrect packet magic"):
        KanyPacket.decode(b"NOPE" + bytes(HEADER_SIZE - 4))
