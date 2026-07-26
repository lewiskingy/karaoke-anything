from collections.abc import Callable

from audio_trombone.processors.base import MediaProcessor
from audio_trombone.processors.delay import DelayPassthroughProcessor
from audio_trombone.processors.null import NullProcessor
from audio_trombone.processors.passthrough import PassthroughProcessor

ProcessorFactory = Callable[[], MediaProcessor]


class ProcessorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProcessorFactory] = {
            "passthrough": PassthroughProcessor,
            "delay-passthrough": lambda: DelayPassthroughProcessor(delay_ms=5.0),
            "null": NullProcessor,
        }

    def create(self, name: str) -> MediaProcessor:
        factory = self._factories.get(name)
        if factory is None:
            available = ", ".join(sorted(self._factories))
            raise ValueError(f"Unknown processor '{name}'. Available: {available}")
        return factory()

    def describe(self) -> dict:
        processors = []
        for name in sorted(self._factories):
            processor = self._factories[name]()
            processors.append(
                {
                    "name": processor.name,
                    "description": processor.description,
                    "capabilities": {
                        "passthrough": processor.capabilities.passthrough,
                        "stateful": processor.capabilities.stateful,
                        "can_buffer": processor.capabilities.can_buffer,
                        "changes_payload": processor.capabilities.changes_payload,
                    },
                }
            )
        return {"processors": processors}
