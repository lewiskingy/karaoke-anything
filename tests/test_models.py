import time

from audio_trombone.models import MediaPacket, Metrics, ProcessedPacket


def test_media_packet_received_stamps_current_time() -> None:
    before = time.time()
    packet = MediaPacket.received(
        payload=b"abc", sender_host="127.0.0.1", sender_port=1234
    )
    after = time.time()

    assert packet.payload == b"abc"
    assert packet.sender_host == "127.0.0.1"
    assert packet.sender_port == 1234
    assert before <= packet.received_at <= after


def test_processed_packet_defaults_to_no_destination() -> None:
    packet = ProcessedPacket(payload=b"xyz")

    assert packet.payload == b"xyz"
    assert packet.destination_host is None
    assert packet.destination_port is None


def test_metrics_as_dict_returns_all_fields() -> None:
    metrics = Metrics(started_at=100.0, packets_received=3)

    result = metrics.as_dict()

    assert result == {
        "started_at": 100.0,
        "packets_received": 3,
        "packets_enqueued": 0,
        "packets_dropped": 0,
        "packets_processed": 0,
        "packets_emitted": 0,
        "packets_forwarded": 0,
        "bytes_received": 0,
        "bytes_forwarded": 0,
        "processor_errors": 0,
        "forwarding_errors": 0,
        "queue_high_watermark": 0,
        "last_packet_at": None,
        "last_sender_host": None,
        "last_sender_port": None,
    }
