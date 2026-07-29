from array import array
import asyncio
import time
import pytest
from audio_trombone.kany import HEADER_SIZE, KanyPacket
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.mdx23c_vocals import (
    MDX23CVocalsConfig,
    MDX23CVocalsProcessor,
)
from conftest import install_fake_torchaudio


def make_processor(**kwargs) -> MDX23CVocalsProcessor:
    inference_fn = kwargs.pop("inference_fn", None)
    return MDX23CVocalsProcessor(
        config=MDX23CVocalsConfig(**kwargs), inference_fn=inference_fn
    )


def packet(sequence: int, value: float = 1.0, sample_rate: int = 100) -> MediaPacket:
    header = bytearray(HEADER_SIZE)
    header[:4] = b"KANY"
    header[4] = 1
    header[6] = 2
    header[7] = 1
    header[8:12] = sample_rate.to_bytes(4, "big")
    header[12:16] = sequence.to_bytes(4, "big")
    header[24:26] = (25).to_bytes(2, "big")
    return MediaPacket.received(
        bytes(header) + array("f", [value] * 50).tobytes(), "127.0.0.1", 40000
    )


async def collect(processor: MDX23CVocalsProcessor, media_packet: MediaPacket):
    return [output async for output in processor.process(media_packet)]


@pytest.mark.parametrize("seconds", [0.1, 0.3, 3.0])
def test_rejects_unvalidated_segment_sizes(seconds):
    with pytest.raises(ValueError, match="one of"):
        make_processor(segment_seconds=seconds, inference_fn=lambda s, r, c: s)


@pytest.mark.asyncio
async def test_exact_frames_bounded_buffer_and_reset():
    processor = make_processor(
        segment_seconds=0.25,
        max_buffered_segments=1,
        inference_fn=lambda s, r, c: array("f", [0] * len(s)),
    )
    await processor.start()
    assert [x async for x in processor.process(packet(0))] == []
    for _ in range(30):
        await asyncio.sleep(0.002)
        outputs = [x async for x in processor.process(packet(1))]
        if outputs:
            break
    decoded = KanyPacket.decode(outputs[0].payload)
    assert decoded.sequence == 0
    assert decoded.frames == 25
    await processor.reset()
    diagnostics = processor.diagnostics()
    assert (
        diagnostics["buffered_input_packets"] == 0
        and diagnostics["ready_output_packets"] == 0
    )
    assert processor._previous_tail is None


def test_overlap_crossfade():
    processor = make_processor(
        segment_seconds=0.25, overlap=0.25, inference_fn=lambda s, r, c: s
    )
    processor._previous_tail = array("f", [0.0] * 4)
    output = array("f", [1.0] * 8)
    processor._crossfade(output)
    assert 0 < output[0] < 1 and output[-1] == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"overlap": 0.6}, "overlap"),
        ({"batch_size": 2}, "batch_size"),
        ({"vocal_reduction": 1.5}, "vocal_reduction"),
        ({"precision": "int8"}, "precision"),
    ],
)
def test_rejects_invalid_construction_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        make_processor(segment_seconds=0.25, inference_fn=lambda s, r, c: s, **kwargs)


class _FakeAdapter:
    def __init__(self, *args, **kwargs) -> None:
        self.device = "cpu"
        self.sample_rate = 8_000
        self.infer_calls: list[tuple[tuple[int, ...], int]] = []

    def infer(self, waveform, sample_rate):
        self.infer_calls.append((tuple(waveform.shape), sample_rate))
        return waveform


@pytest.mark.asyncio
async def test_start_loads_adapter_and_sets_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("audio_trombone.mdx23c.MDX23CAdapter", _FakeAdapter)
    processor = make_processor(segment_seconds=0.25)

    await processor.start()

    assert processor.model_loaded is True
    assert processor.device == "cpu"
    assert processor._adapter is not None


@pytest.mark.asyncio
async def test_start_with_warm_up_runs_one_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("audio_trombone.mdx23c.MDX23CAdapter", _FakeAdapter)
    processor = make_processor(segment_seconds=0.25, warm_up=True)

    await processor.start()

    assert len(processor._adapter.infer_calls) == 1


@pytest.mark.asyncio
async def test_stop_unloads_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audio_trombone.mdx23c.MDX23CAdapter", _FakeAdapter)
    processor = make_processor(segment_seconds=0.25)
    await processor.start()

    await processor.stop()

    assert processor.model_loaded is False
    assert processor._adapter is None


@pytest.mark.asyncio
async def test_process_rejects_non_kany_payload() -> None:
    processor = make_processor(segment_seconds=0.25, inference_fn=lambda s, r, c: s)
    bad = MediaPacket.received(b"not-a-kany-packet", "127.0.0.1", 1)

    with pytest.raises(ValueError, match="MDX23C requires KANY v1 f32 PCM"):
        await collect(processor, bad)


@pytest.mark.asyncio
async def test_process_rejects_non_stereo_input() -> None:
    processor = make_processor(segment_seconds=0.25, inference_fn=lambda s, r, c: s)
    header = bytearray(HEADER_SIZE)
    header[:4] = b"KANY"
    header[4] = 1
    header[6] = 1
    header[7] = 1
    header[8:12] = (100).to_bytes(4, "big")
    header[24:26] = (1).to_bytes(2, "big")
    bad = MediaPacket.received(
        bytes(header) + array("f", [0.0]).tobytes(), "127.0.0.1", 1
    )

    with pytest.raises(ValueError, match="requires stereo input"):
        await collect(processor, bad)


@pytest.mark.asyncio
async def test_process_resets_and_raises_on_format_change() -> None:
    processor = make_processor(segment_seconds=0.25, inference_fn=lambda s, r, c: s)
    await collect(processor, packet(0, sample_rate=100))

    with pytest.raises(ValueError, match="audio format changed"):
        await collect(processor, packet(1, sample_rate=200))

    assert processor._stream_sample_rate is None


@pytest.mark.asyncio
async def test_drops_oldest_packets_when_input_buffer_overflows() -> None:
    def slow(samples: array, rate: int, channels: int) -> array:
        time.sleep(0.05)
        return array("f", [0.0] * len(samples))

    processor = make_processor(
        segment_seconds=0.25, max_buffered_segments=1, inference_fn=slow
    )
    await processor.start()

    await collect(processor, packet(0))  # launches the (slow) first segment
    await collect(processor, packet(1))  # buffers behind the running segment
    await collect(processor, packet(2))  # overflows the 1-segment buffer

    assert processor.dropped_packets == 1
    await processor.reset()


@pytest.mark.asyncio
async def test_flush_drains_ready_output() -> None:
    processor = make_processor(segment_seconds=0.25, inference_fn=lambda s, r, c: s)
    decoded = KanyPacket.decode(packet(0).payload)
    processor._ready_output.append(
        ProcessedPacket(payload=decoded.encode_samples(decoded.samples))
    )

    flushed = [output async for output in processor.flush()]

    assert len(flushed) == 1


@pytest.mark.asyncio
async def test_harvest_wraps_inference_failure() -> None:
    processor = make_processor(segment_seconds=0.25, inference_fn=lambda s, r, c: s)
    decoded = KanyPacket.decode(packet(0).payload)
    processor._active_packets = [decoded]

    async def boom() -> array:
        raise ValueError("boom")

    task = asyncio.create_task(boom())
    await asyncio.sleep(0)
    processor._inference_task = task

    with pytest.raises(RuntimeError, match="MDX23C inference failed"):
        await processor._harvest_inference()

    assert processor.last_error == "boom"
    assert processor._active_packets == []


@pytest.mark.asyncio
async def test_harvest_rejects_sample_count_mismatch() -> None:
    processor = make_processor(segment_seconds=0.25, inference_fn=lambda s, r, c: s)
    decoded = KanyPacket.decode(packet(0).payload)
    processor._active_packets = [decoded]

    async def wrong_length() -> array:
        return array("f", [0.0, 0.0])

    task = asyncio.create_task(wrong_length())
    await asyncio.sleep(0)
    processor._inference_task = task

    with pytest.raises(RuntimeError, match="MDX23C returned"):
        await processor._harvest_inference()

    assert processor._active_packets == []


def test_run_inference_without_adapter_raises() -> None:
    processor = make_processor(segment_seconds=0.25)
    with pytest.raises(RuntimeError, match="MDX23C model is not loaded"):
        processor._run_inference(array("f", [0.0, 0.0]), 100, 2)


def test_run_inference_runs_real_torch_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAdapter:
        sample_rate = 2_000

        def stem_index(self, *names: str) -> int:
            return 0

        def infer(self, waveform, sample_rate):
            return waveform.unsqueeze(1)

    install_fake_torchaudio(
        monkeypatch, resample=lambda tensor, orig_freq, new_freq: tensor[..., :-1]
    )

    processor = make_processor(segment_seconds=0.25, vocal_reduction=1.0)
    processor._adapter = FakeAdapter()

    samples = array("f", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    result = processor._run_inference(samples, 1_000, 2)

    assert len(result) == len(samples)
    assert all(-1.0 <= value <= 1.0 for value in result)
