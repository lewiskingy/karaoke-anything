from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from audio_trombone.models import MediaPacket, ProcessedPacket


@dataclass(frozen=True)
class ProcessorCapabilities:
    passthrough: bool
    stateful: bool
    can_buffer: bool
    changes_payload: bool


class AudioProcessor(ABC):
    """Audio-processing extension point for the server pipeline."""

    name: str
    description: str
    capabilities: ProcessorCapabilities

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def reset(self) -> None:
        return None

    def diagnostics(self) -> dict[str, object]:
        return {}

    @abstractmethod
    def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        # Not `async def`: every real implementation is an async generator
        # (uses `yield`), which calling code iterates directly with
        # `async for` rather than awaiting first. A coroutine-shaped
        # declaration here would mismatch every override's actual type.
        raise NotImplementedError

    async def flush(self) -> AsyncIterator[ProcessedPacket]:
        # `yield` in an unreachable branch makes this an async generator
        # that produces nothing, matching the return type of overrides
        # that do buffer output.
        if False:
            yield ProcessedPacket(payload=b"")
