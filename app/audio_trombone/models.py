from dataclasses import asdict, dataclass
import time


@dataclass(frozen=True)
class MediaPacket:
    payload: bytes
    sender_host: str
    sender_port: int
    received_at: float

    @classmethod
    def received(
        cls,
        payload: bytes,
        sender_host: str,
        sender_port: int,
    ) -> "MediaPacket":
        return cls(
            payload=payload,
            sender_host=sender_host,
            sender_port=sender_port,
            received_at=time.time(),
        )


@dataclass(frozen=True)
class ProcessedPacket:
    payload: bytes
    destination_host: str | None = None
    destination_port: int | None = None


@dataclass
class Metrics:
    started_at: float

    packets_received: int = 0
    packets_enqueued: int = 0
    packets_dropped: int = 0
    packets_processed: int = 0
    packets_emitted: int = 0
    packets_forwarded: int = 0

    bytes_received: int = 0
    bytes_forwarded: int = 0

    processor_errors: int = 0
    forwarding_errors: int = 0

    queue_high_watermark: int = 0

    last_packet_at: float | None = None
    last_sender_host: str | None = None
    last_sender_port: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)
