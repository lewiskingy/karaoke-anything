from array import array
from collections.abc import AsyncIterator

from audio_trombone.kany import KanyPacket, KanyProtocolError
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor, ProcessorCapabilities


class StereoCentreReductionProcessor(AudioProcessor):
    name = "stereo-centre-reduction"
    description = (
        "Reduces stereo-centred content using instantaneous mid/side processing "
        "with no lookahead or buffering."
    )
    capabilities = ProcessorCapabilities(
        passthrough=False,
        stateful=False,
        can_buffer=False,
        changes_payload=True,
    )

    def __init__(self, centre_reduction: float = 0.7) -> None:
        if not 0 <= centre_reduction <= 1:
            raise ValueError("centre_reduction must be between 0.0 and 1.0")
        self.centre_reduction = centre_reduction

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        try:
            decoded = KanyPacket.decode(packet.payload)
        except KanyProtocolError as exc:
            raise ValueError(
                f"Stereo centre reduction requires KANY v1 f32 PCM packets: {exc}"
            ) from exc

        if decoded.channels != 2:
            raise ValueError(
                "Stereo centre reduction requires stereo input; "
                f"received {decoded.channels} channels"
            )

        reduction = self.centre_reduction
        output = array("f", decoded.samples)
        for index in range(0, len(output), 2):
            left = output[index]
            right = output[index + 1]
            centre = (left + right) * 0.5
            output[index] = max(-1.0, min(1.0, left - reduction * centre))
            output[index + 1] = max(-1.0, min(1.0, right - reduction * centre))

        yield ProcessedPacket(payload=decoded.encode_samples(output))

    def diagnostics(self) -> dict[str, object]:
        return {
            "centre_reduction": self.centre_reduction,
            "centre_gain": 1.0 - self.centre_reduction,
            "algorithmic_lookahead_ms": 0.0,
        }
