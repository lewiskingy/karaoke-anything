from array import array
from types import SimpleNamespace

import pytest

from audio_trombone import kany
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


def test_rejects_payload_shorter_than_header() -> None:
    with pytest.raises(KanyProtocolError, match="shorter than the"):
        KanyPacket.decode(b"\x00" * (HEADER_SIZE - 1))


def test_rejects_unsupported_version() -> None:
    payload = bytearray(make_payload(array("f", [0.0, 0.0])))
    payload[4] = 2
    with pytest.raises(KanyProtocolError, match="unsupported protocol version 2"):
        KanyPacket.decode(bytes(payload))


def test_rejects_unsupported_sample_format() -> None:
    payload = bytearray(make_payload(array("f", [0.0, 0.0])))
    payload[7] = 9
    with pytest.raises(KanyProtocolError, match="unsupported sample format 9"):
        KanyPacket.decode(bytes(payload))


def test_rejects_zero_channels_frames_or_sample_rate() -> None:
    payload = bytearray(make_payload(array("f", [0.0, 0.0])))
    payload[6] = 0
    with pytest.raises(KanyProtocolError, match="must be non-zero"):
        KanyPacket.decode(bytes(payload))


def test_rejects_payload_length_mismatch() -> None:
    payload = make_payload(array("f", [0.0, 0.25, -0.5, 1.0])) + b"\x00\x00\x00\x00"
    with pytest.raises(KanyProtocolError, match="payload length mismatch"):
        KanyPacket.decode(payload)


def test_decode_byteswaps_on_big_endian_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    original = array("f", [0.0, 0.25, -0.5, 1.0])
    expected = array("f", original)
    expected.byteswap()

    monkeypatch.setattr(kany, "sys", SimpleNamespace(byteorder="big"))
    packet = KanyPacket.decode(make_payload(original))

    assert packet.samples.tobytes() == expected.tobytes()


def test_encode_samples_rejects_wrong_typecode() -> None:
    packet = KanyPacket.decode(make_payload(array("f", [0.0, 0.25, -0.5, 1.0])))
    with pytest.raises(KanyProtocolError, match="array\\('f'\\)"):
        packet.encode_samples(array("d", [0.0, 0.25, -0.5, 1.0]))


def test_encode_samples_rejects_sample_count_mismatch() -> None:
    packet = KanyPacket.decode(make_payload(array("f", [0.0, 0.25, -0.5, 1.0])))
    with pytest.raises(KanyProtocolError, match="sample count mismatch"):
        packet.encode_samples(array("f", [0.0, 0.25]))


def test_encode_samples_byteswaps_on_big_endian_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    packet = KanyPacket.decode(make_payload(array("f", [0.0, 0.25, -0.5, 1.0])))
    replacement = array("f", [0.0, 0.25, -0.5, 1.0])
    expected = array("f", replacement)
    expected.byteswap()

    monkeypatch.setattr(kany, "sys", SimpleNamespace(byteorder="big"))
    encoded = packet.encode_samples(replacement)

    assert encoded == packet.raw_header + expected.tobytes()
