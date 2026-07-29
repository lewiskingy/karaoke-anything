from collections.abc import AsyncIterator

from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor, ProcessorCapabilities


class NullProcessor(AudioProcessor):
    name = "null"
    description = "Consumes packets and emits no media."
    capabilities = ProcessorCapabilities(
        passthrough=False,
        stateful=False,
        can_buffer=True,
        changes_payload=True,
    )

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        # `yield` in an unreachable branch makes this an async generator
        # that produces nothing -- this processor discards every packet.
        if False:
            yield ProcessedPacket(payload=packet.payload)
