from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from audio_trombone.models import MediaPacket, ProcessedPacket


@dataclass(frozen=True)
class ProcessorCapabilities:
    passthrough: bool
    stateful: bool
    can_buffer: bool
    changes_payload: bool


class MediaProcessor(ABC):
    name: str
    description: str
    capabilities: ProcessorCapabilities

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def reset(self) -> None:
        return None

    @abstractmethod
    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        raise NotImplementedError

    async def flush(self) -> AsyncIterator[ProcessedPacket]:
        if False:
            yield ProcessedPacket(payload=b"")
