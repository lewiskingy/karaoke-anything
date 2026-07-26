import asyncio
import logging

from audio_trombone.models import MediaPacket, Metrics


logger = logging.getLogger(__name__)


class UdpIngressProtocol(asyncio.DatagramProtocol):
    """Receive UDP quickly and enqueue packets for asynchronous processing."""

    def __init__(
        self,
        queue: asyncio.Queue[MediaPacket],
        metrics: Metrics,
        max_packet_size: int,
    ) -> None:
        self.queue = queue
        self.metrics = metrics
        self.max_packet_size = max_packet_size
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]
        logger.info("UDP ingress ready on %s", transport.get_extra_info("sockname"))

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if len(data) > self.max_packet_size:
            self.metrics.packets_dropped += 1
            logger.warning("Dropping oversized packet of %d bytes from %s", len(data), addr)
            return

        host, port = addr
        packet = MediaPacket.received(data, host, port)
        self.metrics.packets_received += 1
        self.metrics.bytes_received += len(data)
        self.metrics.last_packet_at = packet.received_at
        self.metrics.last_sender_host = host
        self.metrics.last_sender_port = port

        try:
            self.queue.put_nowait(packet)
            self.metrics.packets_enqueued += 1
            self.metrics.queue_high_watermark = max(
                self.metrics.queue_high_watermark,
                self.queue.qsize(),
            )
        except asyncio.QueueFull:
            self.metrics.packets_dropped += 1
            logger.warning("Dropping packet from %s:%d because input queue is full", host, port)

    def error_received(self, exc: Exception) -> None:
        logger.warning("UDP ingress error: %s", exc)

    def connection_lost(self, exc: Exception | None) -> None:
        if exc:
            logger.error("UDP ingress stopped with error: %s", exc)
        else:
            logger.info("UDP ingress stopped")
