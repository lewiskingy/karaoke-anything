import asyncio
from typing import AsyncIterator

from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import MediaProcessor, ProcessorCapabilities


class DelayPassthroughProcessor(MediaProcessor):
    name = "delay-passthrough"
    description = "Returns payloads unchanged after a configurable delay."
    capabilities = ProcessorCapabilities(
        passthrough=True,
        stateful=False,
        can_buffer=False,
        changes_payload=False,
    )

    def __init__(self, delay_ms: float = 5.0) -> None:
        self.delay_seconds = delay_ms / 1000.0

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        await asyncio.sleep(self.delay_seconds)
        yield ProcessedPacket(payload=packet.payload)
