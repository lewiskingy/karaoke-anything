import asyncio
from dataclasses import asdict, replace
import logging
import time
from typing import Any

from audio_trombone.config import Settings
from audio_trombone.models import MediaPacket, Metrics, ProcessedPacket
from audio_trombone.processors import AudioProcessor, ProcessorRegistry
from audio_trombone.transport import UdpIngressProtocol

logger = logging.getLogger(__name__)


class TromboneService:
    def __init__(self, settings: Settings) -> None:
        self.startup_settings = settings
        self.settings = settings
        self.metrics = Metrics(started_at=time.time())
        self.input_queue: asyncio.Queue[MediaPacket] = asyncio.Queue(
            maxsize=settings.input_queue_size
        )
        self.processor_registry = ProcessorRegistry(settings)
        self.processor: AudioProcessor = self.processor_registry.create(settings.processor)
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
            self._processing_loop(), name="audio-processing-loop"
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
        self.settings = replace(self.settings, processor=name)
        logger.info("Processor changed from %s to %s", old.name, replacement.name)

    async def reset_processor(self) -> None:
        async with self._processor_lock:
            await self.processor.reset()

    def runtime_settings(self) -> dict[str, object]:
        return {
            "processor": self.settings.processor,
            "demucs": {
                "model": self.settings.demucs_model,
                "device": self.settings.demucs_device,
                "segment_seconds": self.settings.demucs_segment_seconds,
                "overlap": self.settings.demucs_overlap,
                "shifts": self.settings.demucs_shifts,
                "vocal_reduction": self.settings.demucs_vocal_reduction,
            },
            "centre_reduction": {
                "reduction": self.settings.centre_reduction,
            },
            "startup_defaults": {
                "processor": self.startup_settings.processor,
                "demucs": {
                    "model": self.startup_settings.demucs_model,
                    "device": self.startup_settings.demucs_device,
                    "segment_seconds": self.startup_settings.demucs_segment_seconds,
                    "overlap": self.startup_settings.demucs_overlap,
                    "shifts": self.startup_settings.demucs_shifts,
                    "vocal_reduction": self.startup_settings.demucs_vocal_reduction,
                },
                "centre_reduction": {
                    "reduction": self.startup_settings.centre_reduction,
                },
            },
        }

    async def update_runtime_settings(self, updates: dict[str, object]) -> dict[str, object]:
        allowed = {
            "processor",
            "demucs_model",
            "demucs_device",
            "demucs_segment_seconds",
            "demucs_overlap",
            "demucs_shifts",
            "demucs_vocal_reduction",
            "centre_reduction",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unknown runtime settings: {', '.join(sorted(unknown))}")

        candidate = replace(self.settings, **updates)
        self._validate_runtime_settings(candidate)

        if self._can_apply_live(updates):
            if "demucs_vocal_reduction" in updates:
                self.processor.vocal_reduction = candidate.demucs_vocal_reduction  # type: ignore[attr-defined]
                applies_from = "next segment"
                logger.info(
                    "Updated live vocal reduction to %.2f",
                    candidate.demucs_vocal_reduction,
                )
            else:
                self.processor.centre_reduction = candidate.centre_reduction  # type: ignore[attr-defined]
                applies_from = "next packet"
                logger.info(
                    "Updated live centre reduction to %.2f",
                    candidate.centre_reduction,
                )
            self.settings = candidate
            return {"processor_restarted": False, "applies_from": applies_from}

        registry = ProcessorRegistry(candidate)
        replacement = registry.create(candidate.processor)
        await replacement.start()
        async with self._processor_lock:
            old = self.processor
            await self._flush_processor()
            self.processor = replacement
            self.processor_registry = registry
            self.settings = candidate
            await old.stop()
        logger.info("Runtime settings applied; processor reinitialised")
        return {"processor_restarted": True, "applies_from": "immediately"}

    def _can_apply_live(self, updates: dict[str, object]) -> bool:
        keys = set(updates)
        return (
            keys == {"demucs_vocal_reduction"}
            and self.processor.name == "htdemucs-vocals"
            and hasattr(self.processor, "vocal_reduction")
        ) or (
            keys == {"centre_reduction"}
            and self.processor.name == "stereo-centre-reduction"
            and hasattr(self.processor, "centre_reduction")
        )

    async def restore_startup_settings(self) -> dict[str, object]:
        startup = asdict(self.startup_settings)
        updates = {
            key: startup[key]
            for key in (
                "processor",
                "demucs_model",
                "demucs_device",
                "demucs_segment_seconds",
                "demucs_overlap",
                "demucs_shifts",
                "demucs_vocal_reduction",
                "centre_reduction",
            )
        }
        return await self.update_runtime_settings(updates)

    @staticmethod
    def _validate_runtime_settings(settings: Settings) -> None:
        if not settings.processor:
            raise ValueError("processor must not be empty")
        if not settings.demucs_model.strip():
            raise ValueError("demucs model must not be empty")
        if settings.demucs_segment_seconds <= 0:
            raise ValueError("segment_seconds must be greater than zero")
        if not 0 <= settings.demucs_overlap < 1:
            raise ValueError("overlap must be between 0.0 and less than 1.0")
        if settings.demucs_shifts < 0:
            raise ValueError("shifts must be zero or greater")
        if not 0 <= settings.demucs_vocal_reduction <= 1:
            raise ValueError("vocal_reduction must be between 0.0 and 1.0")
        if not 0 <= settings.centre_reduction <= 1:
            raise ValueError("centre_reduction must be between 0.0 and 1.0")

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
                "diagnostics": self.processor.diagnostics(),
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
