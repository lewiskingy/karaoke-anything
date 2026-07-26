import asyncio
import logging
import time
from typing import Any

from audio_trombone.config import Settings
from audio_trombone.models import MediaPacket, Metrics, ProcessedPacket
from audio_trombone.processors import MediaProcessor, ProcessorRegistry
from audio_trombone.transport import UdpIngressProtocol

logger = logging.getLogger(__name__)


class TromboneService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metrics = Metrics(started_at=time.time())
        self.input_queue: asyncio.Queue[MediaPacket] = asyncio.Queue(
            maxsize=settings.input_queue_size
        )
        self.processor_registry = ProcessorRegistry()
        self.processor: MediaProcessor = self.processor_registry.create(
            settings.processor
        )
        self.transport: asyncio.DatagramTransport | None = None
        self.protocol: UdpIngressProtocol | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self._processor_lock = asyncio.Lock()
        self._running = False

    async def start(self) -> None:
        if self._running:
            return

        await self.processor.start()
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: UdpIngressProtocol(
                queue=self.input_queue,
                metrics=self.metrics,
                max_packet_size=self.settings.max_packet_size,
            ),
            local_addr=(self.settings.listen_host, self.settings.input_port),
        )
        self.transport = transport  # type: ignore[assignment]
        self.protocol = protocol  # type: ignore[assignment]
        self.worker_task = asyncio.create_task(
            self._processing_loop(),
            name="media-processing-loop",
        )
        self._running = True
        logger.info(
            "Karaoke Anything started: ingress=%s:%d, output=%s:%d, processor=%s",
            self.settings.listen_host,
            self.settings.input_port,
            self.settings.return_host or "<sender-ip>",
            self.settings.output_port,
            self.processor.name,
        )

    async def stop(self) -> None:
        if not self._running:
            return

        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None

        async with self._processor_lock:
            await self._flush_processor()
            await self.processor.stop()

        if self.transport is not None:
            self.transport.close()
            self.transport = None

        self._running = False
        logger.info("Karaoke Anything stopped")

    async def select_processor(self, name: str) -> None:
        replacement = self.processor_registry.create(name)
        await replacement.start()

        async with self._processor_lock:
            old = self.processor
            await self._flush_processor()
            self.processor = replacement
            await old.stop()

        logger.info("Processor changed from %s to %s", old.name, replacement.name)

    async def reset_processor(self) -> None:
        async with self._processor_lock:
            await self.processor.reset()

    async def _processing_loop(self) -> None:
        while True:
            packet = await self.input_queue.get()
            try:
                async with self._processor_lock:
                    async for output in self.processor.process(packet):
                        self.metrics.packets_emitted += 1
                        self._send_output(packet, output)
                self.metrics.packets_processed += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.metrics.processor_errors += 1
                logger.exception(
                    "Processor %s failed for packet from %s:%d",
                    self.processor.name,
                    packet.sender_host,
                    packet.sender_port,
                )
            finally:
                self.input_queue.task_done()

    def _send_output(self, source_packet: MediaPacket, output: ProcessedPacket) -> None:
        if self.transport is None:
            self.metrics.forwarding_errors += 1
            logger.error("Cannot forward packet: UDP transport is unavailable")
            return

        destination_host = (
            output.destination_host
            or self.settings.return_host
            or source_packet.sender_host
        )
        destination_port = output.destination_port or self.settings.output_port

        try:
            self.transport.sendto(output.payload, (destination_host, destination_port))
            self.metrics.packets_forwarded += 1
            self.metrics.bytes_forwarded += len(output.payload)
        except Exception:
            self.metrics.forwarding_errors += 1
            logger.exception(
                "Failed to forward packet to %s:%d",
                destination_host,
                destination_port,
            )

    async def _flush_processor(self) -> None:
        async for output in self.processor.flush():
            sender_host = self.settings.return_host or self.metrics.last_sender_host
            if sender_host is None:
                logger.warning("Discarding flushed packet because no destination is known")
                continue

            synthetic_source = MediaPacket(
                payload=b"",
                sender_host=sender_host,
                sender_port=self.metrics.last_sender_port or 0,
                received_at=time.time(),
            )
            self.metrics.packets_emitted += 1
            self._send_output(synthetic_source, output)

    def health(self) -> dict:
        return {
            "status": "ok" if self._running else "starting",
            "processor": self.processor.name,
        }

    def status(self) -> dict[str, Any]:
        now = time.time()
        last_packet_age_ms = None
        if self.metrics.last_packet_at is not None:
            last_packet_age_ms = round((now - self.metrics.last_packet_at) * 1000, 1)

        capabilities = self.processor.capabilities
        return {
            "status": "ok" if self._running else "starting",
            "pipeline": [
                "udp-ingress",
                "bounded-input-queue",
                self.processor.name,
                "udp-egress",
            ],
            "listen": {
                "host": self.settings.listen_host,
                "port": self.settings.input_port,
            },
            "forward": {
                "host": self.settings.return_host or "sender-ip",
                "port": self.settings.output_port,
            },
            "processor": {
                "name": self.processor.name,
                "description": self.processor.description,
                "capabilities": {
                    "passthrough": capabilities.passthrough,
                    "stateful": capabilities.stateful,
                    "can_buffer": capabilities.can_buffer,
                    "changes_payload": capabilities.changes_payload,
                },
            },
            "queue": {
                "size": self.input_queue.qsize(),
                "capacity": self.settings.input_queue_size,
            },
            "uptime_seconds": round(now - self.metrics.started_at, 3),
            "last_packet_age_ms": last_packet_age_ms,
            "metrics": self.metrics.as_dict(),
        }

    def prometheus_metrics(self) -> str:
        now = time.time()
        last_packet_age = -1.0
        if self.metrics.last_packet_at is not None:
            last_packet_age = now - self.metrics.last_packet_at

        values = {
            "karaoke_anything_uptime_seconds": now - self.metrics.started_at,
            "karaoke_anything_queue_size": self.input_queue.qsize(),
            "karaoke_anything_queue_capacity": self.settings.input_queue_size,
            "karaoke_anything_queue_high_watermark": self.metrics.queue_high_watermark,
            "karaoke_anything_packets_received_total": self.metrics.packets_received,
            "karaoke_anything_packets_enqueued_total": self.metrics.packets_enqueued,
            "karaoke_anything_packets_dropped_total": self.metrics.packets_dropped,
            "karaoke_anything_packets_processed_total": self.metrics.packets_processed,
            "karaoke_anything_packets_emitted_total": self.metrics.packets_emitted,
            "karaoke_anything_packets_forwarded_total": self.metrics.packets_forwarded,
            "karaoke_anything_bytes_received_total": self.metrics.bytes_received,
            "karaoke_anything_bytes_forwarded_total": self.metrics.bytes_forwarded,
            "karaoke_anything_processor_errors_total": self.metrics.processor_errors,
            "karaoke_anything_forwarding_errors_total": self.metrics.forwarding_errors,
            "karaoke_anything_last_packet_age_seconds": last_packet_age,
        }

        lines = []
        for name, value in values.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        lines.append("")
        return "\n".join(lines)
