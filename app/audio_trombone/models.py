import time
from dataclasses import asdict, dataclass


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

    packets_received: int = 0  # UDP datagrams accepted by the ingress socket
    packets_enqueued: int = 0  # of those, successfully placed on the input queue
    packets_dropped: int = 0  # oversized, or the input queue was full
    packets_processed: int = 0  # input packets the processor finished handling
    packets_emitted: int = 0  # outputs the processor produced (process() + flush())
    packets_forwarded: int = 0  # of those, successfully sent back out over UDP

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
