import sys
from array import array
from dataclasses import dataclass

# KANY packet layout (big-endian header, little-endian f32 samples):
#   0..4   magic ("KANY")
#   4      version
#   5      reserved
#   6      channels
#   7      sample_format
#   8..12  sample_rate
#   12..16 sequence
#   16..24 timestamp_us
#   24..26 frames
#   26..28 reserved
#   28..   samples (f32, little-endian, interleaved)
# Mirrored independently in client/src/protocol.rs -- a layout change needs
# both files updated to match.
MAGIC = b"KANY"
VERSION = 1
HEADER_SIZE = 28
SAMPLE_FORMAT_F32_LE = 1


class KanyProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class KanyPacket:
    raw_header: bytes
    channels: int
    sample_rate: int
    sequence: int
    timestamp_us: int
    frames: int
    samples: array

    @classmethod
    def decode(cls, payload: bytes) -> "KanyPacket":
        cls._validate_header(payload)
        channels, sample_rate, sequence, timestamp_us, frames = (
            cls._parse_header_fields(payload)
        )
        cls._validate_body_length(payload, channels, frames)

        return cls(
            raw_header=payload[:HEADER_SIZE],
            channels=channels,
            sample_rate=sample_rate,
            sequence=sequence,
            timestamp_us=timestamp_us,
            frames=frames,
            samples=cls._read_samples(payload),
        )

    @classmethod
    def _validate_header(cls, payload: bytes) -> None:
        cls._validate_preamble(payload)
        cls._validate_version_and_format(payload)

    @staticmethod
    def _validate_preamble(payload: bytes) -> None:
        if len(payload) < HEADER_SIZE:
            raise KanyProtocolError(
                f"packet is shorter than the {HEADER_SIZE}-byte header"
            )
        if payload[0:4] != MAGIC:
            raise KanyProtocolError("incorrect packet magic")

    @staticmethod
    def _validate_version_and_format(payload: bytes) -> None:
        if payload[4] != VERSION:
            raise KanyProtocolError(f"unsupported protocol version {payload[4]}")
        if payload[7] != SAMPLE_FORMAT_F32_LE:
            raise KanyProtocolError(f"unsupported sample format {payload[7]}")

    @staticmethod
    def _parse_header_fields(payload: bytes) -> tuple[int, int, int, int, int]:
        channels = payload[6]
        sample_rate = int.from_bytes(payload[8:12], "big")
        sequence = int.from_bytes(payload[12:16], "big")
        timestamp_us = int.from_bytes(payload[16:24], "big")
        frames = int.from_bytes(payload[24:26], "big")
        if any(value == 0 for value in (channels, sample_rate, frames)):
            raise KanyProtocolError("channels, sample rate and frames must be non-zero")
        return channels, sample_rate, sequence, timestamp_us, frames

    @staticmethod
    def _validate_body_length(payload: bytes, channels: int, frames: int) -> None:
        expected_size = HEADER_SIZE + channels * frames * 4
        if len(payload) != expected_size:
            raise KanyProtocolError(
                f"payload length mismatch: expected {expected_size}, received {len(payload)}"
            )

    @staticmethod
    def _read_samples(payload: bytes) -> array:
        samples = array("f")
        samples.frombytes(payload[HEADER_SIZE:])
        if sys.byteorder != "little":
            samples.byteswap()
        return samples

    def encode_samples(self, samples: array) -> bytes:
        if samples.typecode != "f":
            raise KanyProtocolError("samples must be an array('f')")
        expected_samples = self.channels * self.frames
        if len(samples) != expected_samples:
            raise KanyProtocolError(
                f"sample count mismatch: expected {expected_samples}, received {len(samples)}"
            )

        encoded = array("f", samples)
        if sys.byteorder != "little":
            encoded.byteswap()
        return self.raw_header + encoded.tobytes()
