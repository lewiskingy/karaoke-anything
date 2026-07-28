import asyncio
import time

import pytest

from audio_trombone.config import Settings
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor, ProcessorCapabilities
from audio_trombone.service import TromboneService


def make_service(**overrides) -> TromboneService:
    defaults = dict(
        listen_host="127.0.0.1",
        input_port=0,
        output_port=59_999,
        input_queue_size=8,
        processor="passthrough",
    )
    defaults.update(overrides)
    return TromboneService(Settings(**defaults))


class RecordingTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        self.sent.append((data, addr))


class BrokenTransport:
    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        raise OSError("boom")


class BufferedFakeProcessor(AudioProcessor):
    name = "fake-buffered"
    description = "buffers nothing; used to test flush wiring"
    capabilities = ProcessorCapabilities(
        passthrough=False, stateful=True, can_buffer=True, changes_payload=True
    )

    def __init__(self, outputs: list[ProcessedPacket]) -> None:
        self._outputs = outputs

    async def process(self, packet: MediaPacket):
        if False:
            yield ProcessedPacket(payload=b"")

    async def flush(self):
        for item in self._outputs:
            yield item


class FailingProcessor(AudioProcessor):
    name = "failing"
    description = "always raises during process()"
    capabilities = ProcessorCapabilities(
        passthrough=False, stateful=False, can_buffer=False, changes_payload=False
    )

    async def process(self, packet: MediaPacket):
        raise RuntimeError("boom")
        yield ProcessedPacket(payload=b"")  # pragma: no cover


class SlowProcessor(AudioProcessor):
    name = "slow"
    description = "awaits indefinitely so cancellation can be exercised mid-process"
    capabilities = ProcessorCapabilities(
        passthrough=False, stateful=False, can_buffer=False, changes_payload=False
    )

    async def process(self, packet: MediaPacket):
        await asyncio.sleep(10)
        yield ProcessedPacket(payload=packet.payload)  # pragma: no cover


def make_packet(payload: bytes = b"abcd", host: str = "127.0.0.1", port: int = 4000) -> MediaPacket:
    return MediaPacket.received(payload=payload, sender_host=host, sender_port=port)


# --- lifecycle -------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_is_idempotent_and_stop_closes_transport() -> None:
    service = make_service()

    await service.start()
    assert service._running is True
    assert service.transport is not None

    await service.start()  # second call is a no-op

    await service.stop()
    assert service._running is False
    assert service.transport is None

    await service.stop()  # idempotent when already stopped


@pytest.mark.asyncio
async def test_processing_loop_processes_and_forwards_packets() -> None:
    service = make_service(return_host="10.0.0.1")
    await service.start()
    try:
        await service.input_queue.put(make_packet())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if service.metrics.packets_processed >= 1:
                break
        assert service.metrics.packets_processed == 1
        assert service.metrics.packets_emitted == 1
        assert service.metrics.packets_forwarded == 1
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_processing_loop_records_processor_errors_and_continues() -> None:
    service = make_service()
    await service.start()
    try:
        service.processor = FailingProcessor()
        await service.input_queue.put(make_packet())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if service.metrics.processor_errors >= 1:
                break
        assert service.metrics.processor_errors == 1
        assert service.metrics.packets_processed == 0
        assert service.input_queue.qsize() == 0
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_processing_loop_reraises_cancelled_error_mid_processing() -> None:
    service = make_service()
    await service.start()
    service.processor = SlowProcessor()
    await service.input_queue.put(make_packet())
    await asyncio.sleep(0.05)

    service.worker_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await service.worker_task
    service.worker_task = None

    await service.stop()


# --- processor selection -----------------------------------------------------


@pytest.mark.asyncio
async def test_select_processor_switches_active_processor() -> None:
    service = make_service(processor="passthrough")

    await service.select_processor("null")

    assert service.processor.name == "null"
    assert service.settings.processor == "null"


@pytest.mark.asyncio
async def test_reset_processor_does_not_raise() -> None:
    service = make_service(processor="passthrough")
    await service.reset_processor()


# --- runtime settings ---------------------------------------------------------


def test_runtime_settings_returns_nested_structure() -> None:
    service = make_service()

    settings_dict = service.runtime_settings()

    assert settings_dict["processor"] == service.settings.processor
    assert settings_dict["demucs"]["model"] == service.settings.demucs_model
    assert settings_dict["convtasnet"]["model_path"] == service.settings.convtasnet_model_path
    assert settings_dict["centre_reduction"]["reduction"] == service.settings.centre_reduction
    assert settings_dict["startup_defaults"]["processor"] == service.startup_settings.processor
    assert (
        settings_dict["startup_defaults"]["convtasnet"]["model_path"]
        == service.startup_settings.convtasnet_model_path
    )


@pytest.mark.asyncio
async def test_update_runtime_settings_rejects_unknown_key() -> None:
    service = make_service()
    with pytest.raises(ValueError, match="Unknown runtime settings"):
        await service.update_runtime_settings({"bogus": 1})


@pytest.mark.asyncio
async def test_update_runtime_settings_applies_live_centre_reduction() -> None:
    service = make_service(processor="stereo-centre-reduction")

    result = await service.update_runtime_settings({"centre_reduction": 0.3})

    assert result == {"processor_restarted": False, "applies_from": "next packet"}
    assert service.processor.centre_reduction == 0.3
    assert service.settings.centre_reduction == 0.3


@pytest.mark.asyncio
async def test_update_runtime_settings_applies_live_demucs_vocal_reduction() -> None:
    service = make_service(processor="htdemucs-vocals")

    result = await service.update_runtime_settings({"demucs_vocal_reduction": 0.4})

    assert result == {"processor_restarted": False, "applies_from": "next segment"}
    assert service.processor.vocal_reduction == 0.4
    assert service.settings.demucs_vocal_reduction == 0.4


@pytest.mark.asyncio
async def test_update_runtime_settings_applies_live_convtasnet_vocal_reduction() -> None:
    service = make_service(processor="convtasnet-lyrics-causal")

    result = await service.update_runtime_settings({"convtasnet_vocal_reduction": 0.4})

    assert result == {"processor_restarted": False, "applies_from": "next segment"}
    assert service.processor.vocal_reduction == 0.4
    assert service.settings.convtasnet_vocal_reduction == 0.4


@pytest.mark.asyncio
async def test_update_runtime_settings_restarts_processor_when_not_live_applicable() -> None:
    service = make_service(processor="passthrough")

    result = await service.update_runtime_settings({"processor": "null"})

    assert result == {"processor_restarted": True, "applies_from": "immediately"}
    assert service.processor.name == "null"
    assert service.settings.processor == "null"


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"processor": ""}, "processor must not be empty"),
        ({"demucs_model": "  "}, "demucs model must not be empty"),
        ({"demucs_segment_seconds": 0}, "Demucs segment_seconds"),
        ({"demucs_overlap": 1.0}, "Demucs overlap"),
        ({"demucs_shifts": -1}, "Demucs shifts"),
        ({"demucs_vocal_reduction": 1.5}, "Demucs vocal_reduction"),
        ({"convtasnet_model_path": " "}, "ConvTasNet model_path"),
        ({"convtasnet_segment_seconds": -1}, "ConvTasNet segment_seconds"),
        ({"convtasnet_vocal_reduction": -0.1}, "ConvTasNet vocal_reduction"),
        ({"convtasnet_vocal_source_index": -1}, "ConvTasNet vocal_source_index"),
        ({"convtasnet_accompaniment_source_index": -1}, "ConvTasNet accompaniment_source_index"),
        (
            {"convtasnet_vocal_source_index": 1, "convtasnet_accompaniment_source_index": 1},
            "must differ",
        ),
        ({"centre_reduction": 2.0}, "centre_reduction"),
    ],
)
@pytest.mark.asyncio
async def test_update_runtime_settings_rejects_invalid_values(updates, match) -> None:
    service = make_service()
    with pytest.raises(ValueError, match=match):
        await service.update_runtime_settings(updates)


@pytest.mark.asyncio
async def test_restore_startup_settings_reverts_changes() -> None:
    service = make_service(processor="stereo-centre-reduction", centre_reduction=0.7)
    await service.update_runtime_settings({"centre_reduction": 0.2})
    assert service.settings.centre_reduction == 0.2

    result = await service.restore_startup_settings()

    assert service.settings.centre_reduction == 0.7
    assert result["processor_restarted"] is True


# --- _can_apply_live ----------------------------------------------------------


def test_can_apply_live_true_for_demucs_vocal_reduction() -> None:
    service = make_service(processor="htdemucs-vocals")
    assert service._can_apply_live({"demucs_vocal_reduction": 0.3}) is True


def test_can_apply_live_true_for_convtasnet_vocal_reduction() -> None:
    service = make_service(processor="convtasnet-lyrics-causal")
    assert service._can_apply_live({"convtasnet_vocal_reduction": 0.3}) is True


def test_can_apply_live_true_for_centre_reduction() -> None:
    service = make_service(processor="stereo-centre-reduction")
    assert service._can_apply_live({"centre_reduction": 0.3}) is True


def test_can_apply_live_false_for_mismatched_processor() -> None:
    service = make_service(processor="passthrough")
    assert service._can_apply_live({"centre_reduction": 0.3}) is False


def test_can_apply_live_false_for_multiple_keys() -> None:
    service = make_service(processor="stereo-centre-reduction")
    assert service._can_apply_live({"centre_reduction": 0.3, "processor": "null"}) is False


# --- _send_output --------------------------------------------------------------


def test_send_output_uses_output_destination_when_provided() -> None:
    service = make_service(return_host="10.0.0.1", output_port=6000)
    service.transport = RecordingTransport()
    processed = ProcessedPacket(payload=b"y", destination_host="1.2.3.4", destination_port=7000)

    service._send_output(make_packet(host="192.168.1.5"), processed)

    assert service.transport.sent == [(b"y", ("1.2.3.4", 7000))]
    assert service.metrics.packets_forwarded == 1
    assert service.metrics.bytes_forwarded == 1


def test_send_output_falls_back_to_return_host_and_output_port() -> None:
    service = make_service(return_host="10.0.0.1", output_port=6000)
    service.transport = RecordingTransport()
    processed = ProcessedPacket(payload=b"y")

    service._send_output(make_packet(host="192.168.1.5"), processed)

    assert service.transport.sent == [(b"y", ("10.0.0.1", 6000))]


def test_send_output_falls_back_to_sender_host_when_no_return_host() -> None:
    service = make_service(return_host=None, output_port=6000)
    service.transport = RecordingTransport()
    processed = ProcessedPacket(payload=b"y")

    service._send_output(make_packet(host="192.168.1.5"), processed)

    assert service.transport.sent == [(b"y", ("192.168.1.5", 6000))]


def test_send_output_records_error_when_transport_missing() -> None:
    service = make_service()
    processed = ProcessedPacket(payload=b"y")

    service._send_output(make_packet(), processed)

    assert service.metrics.forwarding_errors == 1


def test_send_output_records_error_when_sendto_raises() -> None:
    service = make_service(return_host="10.0.0.1")
    service.transport = BrokenTransport()
    processed = ProcessedPacket(payload=b"y")

    service._send_output(make_packet(), processed)

    assert service.metrics.forwarding_errors == 1


# --- _flush_processor ----------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_processor_discards_when_no_destination_known() -> None:
    service = make_service(return_host=None)
    service.processor = BufferedFakeProcessor([ProcessedPacket(payload=b"z")])
    service.transport = RecordingTransport()

    await service._flush_processor()

    assert service.transport.sent == []
    assert service.metrics.packets_emitted == 0


@pytest.mark.asyncio
async def test_flush_processor_sends_output_when_destination_known() -> None:
    service = make_service(return_host="10.0.0.1", output_port=6000)
    service.processor = BufferedFakeProcessor([ProcessedPacket(payload=b"z")])
    service.transport = RecordingTransport()

    await service._flush_processor()

    assert service.transport.sent == [(b"z", ("10.0.0.1", 6000))]
    assert service.metrics.packets_emitted == 1


# --- health / status / prometheus ----------------------------------------------


def test_health_reports_starting_before_start() -> None:
    service = make_service()
    assert service.health() == {"status": "starting", "processor": service.processor.name}


@pytest.mark.asyncio
async def test_health_reports_ok_after_start() -> None:
    service = make_service()
    await service.start()
    try:
        assert service.health()["status"] == "ok"
    finally:
        await service.stop()


def test_status_reports_expected_shape_before_start() -> None:
    service = make_service()

    status = service.status()

    assert status["status"] == "starting"
    assert status["last_packet_age_ms"] is None
    assert status["processor"]["name"] == service.processor.name
    assert status["processor"]["capabilities"]["passthrough"] is True
    assert status["queue"]["capacity"] == service.settings.input_queue_size


def test_status_reports_last_packet_age_when_known() -> None:
    service = make_service()
    service.metrics.last_packet_at = time.time() - 1

    status = service.status()

    assert status["last_packet_age_ms"] is not None
    assert status["last_packet_age_ms"] >= 900


def test_prometheus_metrics_reports_negative_one_when_no_packet_seen() -> None:
    service = make_service()

    output = service.prometheus_metrics()

    assert "karaoke_anything_last_packet_age_seconds -1.0" in output
    assert output.endswith("\n")
    assert "# TYPE karaoke_anything_packets_received_total gauge" in output


def test_prometheus_metrics_reports_age_when_packet_seen() -> None:
    service = make_service()
    service.metrics.last_packet_at = time.time() - 5

    output = service.prometheus_metrics()

    line = next(
        line for line in output.splitlines() if line.startswith("karaoke_anything_last_packet_age_seconds")
    )
    value = float(line.split()[1])
    assert value >= 4.9
