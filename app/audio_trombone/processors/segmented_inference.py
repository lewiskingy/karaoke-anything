"""Shared buffering/pacing machinery for segment-based separators.

`ConvTasNetLyricsProcessor` and `HTDemucsProcessor` both accumulate KANY
packets into a fixed-duration segment, run inference off-thread, and release
one processed packet per subsequent input packet so playback stays paced
rather than bursting an entire segment at once. That buffering, launching,
and harvesting logic is identical between them; only KANY-decode error text,
model loading, and the inference call itself differ per model.

`MDX23CVocalsProcessor` (processors/mdx23c_vocals.py) implements the same
buffer/launch/harvest shape independently rather than inheriting from this
class, because it also crossfades between segments and drops the oldest
buffered input when overfull -- neither of which this base class supports.
A change here to segment bookkeeping is likely needed there too.
"""

from array import array
import asyncio
from collections import deque
import logging
import time
from typing import AsyncIterator

from audio_trombone.kany import KanyPacket
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.base import AudioProcessor

logger = logging.getLogger(__name__)


class SegmentedInferenceProcessor(AudioProcessor):
    """Base class for buffered, segment-at-a-time separators.

    Subclasses must set `_model_label` (used in log/error text and the
    background task name) and implement `_decode_and_validate` and
    `_run_inference`.
    """

    _model_label: str

    def __init__(
        self,
        *,
        segment_seconds: float,
        inference_fn,
    ) -> None:
        self.segment_seconds = segment_seconds
        self._inference_fn = inference_fn

        self._input_packets: deque[KanyPacket] = deque()
        self._input_frames = 0
        self._ready_output: deque[ProcessedPacket] = deque()
        self._inference_task: "asyncio.Task[array] | None" = None
        self._active_packets: list[KanyPacket] = []
        self._stream_sample_rate: int | None = None
        self._stream_channels: int | None = None

        self.segments_started = 0
        self.segments_completed = 0
        self.last_inference_seconds: float | None = None
        self.last_real_time_factor: float | None = None
        self.last_error: str | None = None

    def _decode_and_validate(self, payload: bytes) -> KanyPacket:
        """Decode a packet and enforce this model's input requirements,
        raising `ValueError` with model-specific text on failure."""
        raise NotImplementedError

    def _run_inference(self, samples: array, sample_rate: int, channels: int) -> array:
        raise NotImplementedError

    async def reset(self) -> None:
        if self._inference_task is not None:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except (asyncio.CancelledError, Exception):
                pass
        self._inference_task = None
        self._active_packets.clear()
        self._input_packets.clear()
        self._input_frames = 0
        self._ready_output.clear()
        self._stream_sample_rate = None
        self._stream_channels = None
        self.last_error = None

    async def process(self, packet: MediaPacket) -> AsyncIterator[ProcessedPacket]:
        await self._harvest_inference()

        decoded = self._decode_and_validate(packet.payload)
        self._accept_stream_format(decoded)
        self._input_packets.append(decoded)
        self._input_frames += decoded.frames

        if self._inference_task is None and self._segment_ready():
            self._launch_segment()

        if self._ready_output:
            yield self._ready_output.popleft()

    async def flush(self) -> AsyncIterator[ProcessedPacket]:
        await self._harvest_inference()
        while self._ready_output:
            yield self._ready_output.popleft()

    def _accept_stream_format(self, packet: KanyPacket) -> None:
        if self._stream_sample_rate is None:
            self._stream_sample_rate = packet.sample_rate
            self._stream_channels = packet.channels
            return
        if (
            packet.sample_rate != self._stream_sample_rate
            or packet.channels != self._stream_channels
        ):
            raise ValueError(
                "audio format changed without processor reset: "
                f"expected {self._stream_sample_rate}Hz/{self._stream_channels}ch, "
                f"received {packet.sample_rate}Hz/{packet.channels}ch"
            )

    def _target_segment_frames(self) -> int:
        if self._stream_sample_rate is None:
            raise RuntimeError("stream format is not known")
        return max(1, round(self._stream_sample_rate * self.segment_seconds))

    def _segment_ready(self) -> bool:
        return self._input_frames >= self._target_segment_frames()

    def _launch_segment(self) -> None:
        target = self._target_segment_frames()
        packets: list[KanyPacket] = []
        frames = 0
        while self._input_packets and frames < target:
            current = self._input_packets.popleft()
            packets.append(current)
            frames += current.frames
            self._input_frames -= current.frames

        samples = array("f")
        for current in packets:
            samples.extend(current.samples)

        self._active_packets = packets
        self.segments_started += 1
        started = time.perf_counter()

        async def infer() -> array:
            result = await asyncio.to_thread(
                self._run_inference,
                samples,
                packets[0].sample_rate,
                packets[0].channels,
            )
            elapsed = time.perf_counter() - started
            self.last_inference_seconds = elapsed
            duration = frames / packets[0].sample_rate
            self.last_real_time_factor = elapsed / duration
            return result

        self._inference_task = asyncio.create_task(
            infer(), name=f"{self._model_label.lower()}-segment-{self.segments_started}"
        )

    async def _harvest_inference(self) -> None:
        task = self._inference_task
        if task is None or not task.done():
            return
        self._inference_task = None

        separated = await self._await_segment_result(task)
        self._distribute_segment_output(separated)
        self._finish_segment()

    async def _await_segment_result(self, task: "asyncio.Task[array]") -> array:
        try:
            return await task
        except asyncio.CancelledError:
            self._active_packets.clear()
            raise
        except Exception as exc:
            self.last_error = str(exc)
            self._active_packets.clear()
            raise RuntimeError(f"{self._model_label} inference failed: {exc}") from exc

    def _distribute_segment_output(self, separated: array) -> None:
        expected = sum(
            packet.frames * packet.channels for packet in self._active_packets
        )
        if len(separated) != expected:
            self._active_packets.clear()
            raise RuntimeError(
                f"{self._model_label} returned {len(separated)} samples; expected {expected}"
            )

        offset = 0
        for original in self._active_packets:
            count = original.frames * original.channels
            packet_samples = array("f", separated[offset : offset + count])
            self._ready_output.append(
                ProcessedPacket(payload=original.encode_samples(packet_samples))
            )
            offset += count

    def _finish_segment(self) -> None:
        self._active_packets.clear()
        self.segments_completed += 1
        self.last_error = None
        logger.info(
            "%s segment complete: inference=%.3fs rtf=%.3f ready_packets=%d",
            self._model_label,
            self.last_inference_seconds or 0.0,
            self.last_real_time_factor or 0.0,
            len(self._ready_output),
        )
