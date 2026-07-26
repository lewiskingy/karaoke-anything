from typing import AsyncIterator

from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import MediaProcessor, ProcessorCapabilities


class NullProcessor(MediaProcessor):
    name = "null"
    description = "Consumes packets and emits no media."
    capabilities = ProcessorCapabilities(
        passthrough=False,
        stateful=False,
        can_buffer=True,
        changes_payload=True,
    )

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        if False:
            yield ProcessedPacket(payload=packet.payload)
