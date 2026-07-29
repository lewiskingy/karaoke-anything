import asyncio
import logging
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

from audio_trombone.config import Settings
from audio_trombone.models import MediaPacket, Metrics, ProcessedPacket
from audio_trombone.processors import AudioProcessor, ProcessorRegistry
from audio_trombone.processors.mdx23c_vocals import SUPPORTED_SEGMENTS
from audio_trombone.transport import UdpIngressProtocol

logger = logging.getLogger(__name__)

# Single source of truth for the runtime-settings API: each group name maps
# the JSON field a client sends to the `Settings` attribute it updates.
# `runtime_settings()`, `update_runtime_settings()`, `restore_startup_settings()`,
# and main.py's request-flattening all derive from this instead of repeating
# the field list.
RUNTIME_SETTING_GROUPS: dict[str, dict[str, str]] = {
    "demucs": {
        "model": "demucs_model",
        "device": "demucs_device",
        "segment_seconds": "demucs_segment_seconds",
        "overlap": "demucs_overlap",
        "shifts": "demucs_shifts",
        "vocal_reduction": "demucs_vocal_reduction",
    },
    "convtasnet": {
        "model_path": "convtasnet_model_path",
        "device": "convtasnet_device",
        "segment_seconds": "convtasnet_segment_seconds",
        "vocal_reduction": "convtasnet_vocal_reduction",
        "vocal_source_index": "convtasnet_vocal_source_index",
        "accompaniment_source_index": "convtasnet_accompaniment_source_index",
    },
    "mdx23c": {
        "device": "mdx23c_device",
        "segment_seconds": "mdx23c_segment_seconds",
        "overlap": "mdx23c_overlap",
        "batch_size": "mdx23c_batch_size",
        "vocal_reduction": "mdx23c_vocal_reduction",
        "precision": "mdx23c_precision",
    },
    "centre_reduction": {
        "reduction": "centre_reduction",
    },
}

_ALL_SETTING_FIELDS: frozenset[str] = frozenset(
    {"processor"}
    | {
        attribute
        for group in RUNTIME_SETTING_GROUPS.values()
        for attribute in group.values()
    }
)


def _settings_snapshot(settings: Settings) -> dict[str, object]:
    """The `runtime_settings()` shape for one `Settings` instance: `processor`
    plus one dict per group, each mapping the group's JSON field names to
    their current values."""
    snapshot: dict[str, object] = {"processor": settings.processor}
    for group_name, fields in RUNTIME_SETTING_GROUPS.items():
        snapshot[group_name] = {
            field_name: getattr(settings, attribute)
            for field_name, attribute in fields.items()
        }
    return snapshot


@dataclass(frozen=True)
class _LiveApplyRule:
    """A single runtime setting that can be pushed onto the active processor
    without restarting it, provided that processor is the one currently running."""

    setting_key: str
    processor_name: str
    attribute: str
    applies_from: str
    log_message: str

    def matches(self, key: str, processor: AudioProcessor) -> bool:
        return (
            key == self.setting_key
            and processor.name == self.processor_name
            and hasattr(processor, self.attribute)
        )


class TromboneService:
    def __init__(self, settings: Settings) -> None:
        self.startup_settings = settings
        self.settings = settings
        self.metrics = Metrics(started_at=time.time())
        self.input_queue: asyncio.Queue[MediaPacket] = asyncio.Queue(
            maxsize=settings.input_queue_size
        )
        self.processor_registry = ProcessorRegistry(settings)
        self.processor: AudioProcessor = self.processor_registry.create(
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
        snapshot = _settings_snapshot(self.settings)
        snapshot["startup_defaults"] = _settings_snapshot(self.startup_settings)
        return snapshot

    async def update_runtime_settings(
        self, updates: dict[str, object]
    ) -> dict[str, object]:
        unknown = set(updates) - _ALL_SETTING_FIELDS
        if unknown:
            raise ValueError(f"Unknown runtime settings: {', '.join(sorted(unknown))}")

        candidate = replace(self.settings, **updates)
        self._validate_runtime_settings(candidate)

        rule = self._find_live_apply_rule(updates)
        if rule is not None:
            new_value = getattr(candidate, rule.setting_key)
            setattr(self.processor, rule.attribute, new_value)
            logger.info(rule.log_message, new_value)
            self.settings = candidate
            return {"processor_restarted": False, "applies_from": rule.applies_from}

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

    _LIVE_APPLY_RULES = (
        _LiveApplyRule(
            "demucs_vocal_reduction",
            "htdemucs-vocals",
            "vocal_reduction",
            "next segment",
            "Updated live Demucs vocal reduction to %.2f",
        ),
        _LiveApplyRule(
            "mdx23c_vocal_reduction",
            "mdx23c-vocals",
            "vocal_reduction",
            "next segment",
            "Updated live MDX23C vocal reduction to %.2f",
        ),
        _LiveApplyRule(
            "convtasnet_vocal_reduction",
            "convtasnet-lyrics-causal",
            "vocal_reduction",
            "next segment",
            "Updated live ConvTasNet vocal reduction to %.2f",
        ),
        _LiveApplyRule(
            "centre_reduction",
            "stereo-centre-reduction",
            "centre_reduction",
            "next packet",
            "Updated live centre reduction to %.2f",
        ),
    )

    def _find_live_apply_rule(
        self, updates: dict[str, object]
    ) -> "_LiveApplyRule | None":
        keys = set(updates)
        if len(keys) != 1:
            return None
        (key,) = keys
        for rule in self._LIVE_APPLY_RULES:
            if rule.matches(key, self.processor):
                return rule
        return None

    def _can_apply_live(self, updates: dict[str, object]) -> bool:
        return self._find_live_apply_rule(updates) is not None

    async def restore_startup_settings(self) -> dict[str, object]:
        startup = asdict(self.startup_settings)
        updates = {key: startup[key] for key in _ALL_SETTING_FIELDS}
        return await self.update_runtime_settings(updates)

    @staticmethod
    def _validate_runtime_settings(settings: Settings) -> None:
        if not settings.processor:
            raise ValueError("processor must not be empty")
        TromboneService._validate_demucs_settings(settings)
        TromboneService._validate_convtasnet_settings(settings)
        TromboneService._validate_mdx23c_settings(settings)
        if not 0 <= settings.centre_reduction <= 1:
            raise ValueError("centre_reduction must be between 0.0 and 1.0")

    @staticmethod
    def _raise_first_failure(checks: tuple[tuple[bool, str], ...]) -> None:
        for passed, message in checks:
            if not passed:
                raise ValueError(message)

    @staticmethod
    def _validate_demucs_settings(settings: Settings) -> None:
        TromboneService._raise_first_failure(
            (
                (bool(settings.demucs_model.strip()), "demucs model must not be empty"),
                (
                    settings.demucs_segment_seconds > 0,
                    "Demucs segment_seconds must be greater than zero",
                ),
                (
                    0 <= settings.demucs_overlap < 1,
                    "Demucs overlap must be between 0.0 and less than 1.0",
                ),
                (settings.demucs_shifts >= 0, "Demucs shifts must be zero or greater"),
                (
                    0 <= settings.demucs_vocal_reduction <= 1,
                    "Demucs vocal_reduction must be between 0.0 and 1.0",
                ),
            )
        )

    @staticmethod
    def _validate_convtasnet_settings(settings: Settings) -> None:
        TromboneService._raise_first_failure(
            (
                (
                    bool(settings.convtasnet_model_path.strip()),
                    "ConvTasNet model_path must not be empty",
                ),
                (
                    settings.convtasnet_segment_seconds > 0,
                    "ConvTasNet segment_seconds must be greater than zero",
                ),
                (
                    0 <= settings.convtasnet_vocal_reduction <= 1,
                    "ConvTasNet vocal_reduction must be between 0.0 and 1.0",
                ),
                (
                    settings.convtasnet_vocal_source_index >= 0,
                    "ConvTasNet vocal_source_index must be zero or greater",
                ),
                (
                    settings.convtasnet_accompaniment_source_index >= 0,
                    "ConvTasNet accompaniment_source_index must be zero or greater",
                ),
                (
                    settings.convtasnet_vocal_source_index
                    != settings.convtasnet_accompaniment_source_index,
                    "ConvTasNet vocal and accompaniment source indexes must differ",
                ),
            )
        )

    @staticmethod
    def _validate_mdx23c_settings(settings: Settings) -> None:
        TromboneService._raise_first_failure(
            (
                (
                    settings.mdx23c_segment_seconds in SUPPORTED_SEGMENTS,
                    (
                        "MDX23C segment_seconds must be one of "
                        f"{', '.join(str(value) for value in SUPPORTED_SEGMENTS)}"
                    ),
                ),
                (
                    0 <= settings.mdx23c_overlap < 0.5,
                    "MDX23C overlap must be between 0.0 and less than 0.5",
                ),
                (settings.mdx23c_batch_size == 1, "MDX23C batch_size must be 1"),
                (
                    0 <= settings.mdx23c_vocal_reduction <= 1,
                    "MDX23C vocal_reduction must be between 0.0 and 1.0",
                ),
                (
                    settings.mdx23c_precision in {"float32", "float16", "bfloat16"},
                    "MDX23C precision must be float32, float16, or bfloat16",
                ),
            )
        )

    async def _processing_loop(self) -> None:
        while True:
            packet = await self.input_queue.get()
            try:
                async with self._processor_lock:
                    async for output in self.processor.process(packet):
                        self.metrics.packets_emitted += 1
                        self._send_output(output, packet.sender_host)
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

    def _send_output(self, output: ProcessedPacket, fallback_host: str) -> None:
        """Forward `output`, preferring its own destination, then the
        configured return host, then `fallback_host` (the sender of the
        input packet this output was produced from, or -- for flushed
        output with no single input packet -- the last known sender)."""
        if self.transport is None:
            self.metrics.forwarding_errors += 1
            logger.error("Cannot forward packet: UDP transport is unavailable")
            return
        destination_host = (
            output.destination_host or self.settings.return_host or fallback_host
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
            fallback_host = self.settings.return_host or self.metrics.last_sender_host
            if fallback_host is None:
                logger.warning(
                    "Discarding flushed packet because no destination is known"
                )
                continue
            self.metrics.packets_emitted += 1
            self._send_output(output, fallback_host)

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
