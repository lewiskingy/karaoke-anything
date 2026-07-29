import asyncio
import time
from types import SimpleNamespace

import pytest

from audio_trombone.models import Metrics
from audio_trombone.transport import UdpIngressProtocol


def make_protocol(
    queue_size: int = 10, max_packet_size: int = 65535
) -> UdpIngressProtocol:
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    metrics = Metrics(started_at=time.time())
    return UdpIngressProtocol(
        queue=queue, metrics=metrics, max_packet_size=max_packet_size
    )


@pytest.mark.asyncio
async def test_connection_made_stores_transport() -> None:
    protocol = make_protocol()
    transport = SimpleNamespace(get_extra_info=lambda key: ("0.0.0.0", 5004))

    protocol.connection_made(transport)

    assert protocol.transport is transport


@pytest.mark.asyncio
async def test_datagram_received_drops_oversized_packets() -> None:
    protocol = make_protocol(max_packet_size=4)

    protocol.datagram_received(b"12345", ("127.0.0.1", 40_000))

    assert protocol.metrics.packets_dropped == 1
    assert protocol.metrics.packets_received == 0
    assert protocol.queue.qsize() == 0


@pytest.mark.asyncio
async def test_datagram_received_enqueues_and_updates_metrics() -> None:
    protocol = make_protocol()

    protocol.datagram_received(b"abcd", ("127.0.0.1", 40_000))

    assert protocol.metrics.packets_received == 1
    assert protocol.metrics.bytes_received == 4
    assert protocol.metrics.packets_enqueued == 1
    assert protocol.metrics.queue_high_watermark == 1
    assert protocol.metrics.last_sender_host == "127.0.0.1"
    assert protocol.metrics.last_sender_port == 40_000
    assert protocol.metrics.last_packet_at is not None

    queued = protocol.queue.get_nowait()
    assert queued.payload == b"abcd"
    assert queued.sender_host == "127.0.0.1"
    assert queued.sender_port == 40_000


@pytest.mark.asyncio
async def test_datagram_received_drops_when_queue_is_full() -> None:
    protocol = make_protocol(queue_size=1)
    protocol.datagram_received(b"first", ("127.0.0.1", 40_000))

    protocol.datagram_received(b"second", ("127.0.0.1", 40_001))

    assert protocol.metrics.packets_enqueued == 1
    assert protocol.metrics.packets_dropped == 1
    assert protocol.queue.qsize() == 1


@pytest.mark.asyncio
async def test_error_received_does_not_raise() -> None:
    protocol = make_protocol()
    protocol.error_received(OSError("boom"))


@pytest.mark.asyncio
async def test_connection_lost_with_and_without_error() -> None:
    protocol = make_protocol()
    protocol.connection_lost(None)
    protocol.connection_lost(OSError("boom"))
