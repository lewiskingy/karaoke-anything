from typing import AsyncIterator

from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor, ProcessorCapabilities


class PassthroughProcessor(AudioProcessor):
    name = "passthrough"
    description = "Returns each incoming UDP payload byte-for-byte."
    capabilities = ProcessorCapabilities(
        passthrough=True,
        stateful=False,
        can_buffer=False,
        changes_payload=False,
    )

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        yield ProcessedPacket(payload=packet.payload)
