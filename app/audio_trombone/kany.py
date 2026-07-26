from array import array
from dataclasses import dataclass
import sys

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
        if len(payload) < HEADER_SIZE:
            raise KanyProtocolError(f"packet is shorter than the {HEADER_SIZE}-byte header")
        if payload[0:4] != MAGIC:
            raise KanyProtocolError("incorrect packet magic")
        if payload[4] != VERSION:
            raise KanyProtocolError(f"unsupported protocol version {payload[4]}")
        if payload[7] != SAMPLE_FORMAT_F32_LE:
            raise KanyProtocolError(f"unsupported sample format {payload[7]}")

        channels = payload[6]
        sample_rate = int.from_bytes(payload[8:12], "big")
        sequence = int.from_bytes(payload[12:16], "big")
        timestamp_us = int.from_bytes(payload[16:24], "big")
        frames = int.from_bytes(payload[24:26], "big")
        if channels == 0 or frames == 0 or sample_rate == 0:
            raise KanyProtocolError("channels, sample rate and frames must be non-zero")

        expected_samples = channels * frames
        expected_size = HEADER_SIZE + expected_samples * 4
        if len(payload) != expected_size:
            raise KanyProtocolError(
                f"payload length mismatch: expected {expected_size}, received {len(payload)}"
            )

        samples = array("f")
        samples.frombytes(payload[HEADER_SIZE:])
        if sys.byteorder != "little":
            samples.byteswap()

        return cls(
            raw_header=payload[:HEADER_SIZE],
            channels=channels,
            sample_rate=sample_rate,
            sequence=sequence,
            timestamp_us=timestamp_us,
            frames=frames,
            samples=samples,
        )

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
