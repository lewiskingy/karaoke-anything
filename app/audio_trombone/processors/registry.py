from collections.abc import Callable

from audio_trombone.config import Settings
from audio_trombone.processors.base import AudioProcessor
from audio_trombone.processors.centre_reduction import StereoCentreReductionProcessor
from audio_trombone.processors.convtasnet_lyrics import ConvTasNetLyricsProcessor
from audio_trombone.processors.delay import DelayPassthroughProcessor
from audio_trombone.processors.htdemucs import HTDemucsProcessor
from audio_trombone.processors.null import NullProcessor
from audio_trombone.processors.passthrough import PassthroughProcessor

ProcessorFactory = Callable[[], AudioProcessor]


class ProcessorRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or Settings.from_environment()
        self._factories: dict[str, ProcessorFactory] = {
            "passthrough": PassthroughProcessor,
            "delay-passthrough": lambda: DelayPassthroughProcessor(delay_ms=5.0),
            "null": NullProcessor,
            "stereo-centre-reduction": lambda: StereoCentreReductionProcessor(
                centre_reduction=settings.centre_reduction,
            ),
            "htdemucs-vocals": lambda: HTDemucsProcessor(
                model_name=settings.demucs_model,
                device=settings.demucs_device,
                segment_seconds=settings.demucs_segment_seconds,
                overlap=settings.demucs_overlap,
                shifts=settings.demucs_shifts,
                vocal_reduction=settings.demucs_vocal_reduction,
            ),
            "convtasnet-lyrics-causal": lambda: ConvTasNetLyricsProcessor(
                model_path=settings.convtasnet_model_path,
                device=settings.convtasnet_device,
                segment_seconds=settings.convtasnet_segment_seconds,
                vocal_reduction=settings.convtasnet_vocal_reduction,
                vocal_source_index=settings.convtasnet_vocal_source_index,
                accompaniment_source_index=(
                    settings.convtasnet_accompaniment_source_index
                ),
            ),
        }

    def create(self, name: str) -> AudioProcessor:
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
