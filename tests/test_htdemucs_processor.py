from array import array
import asyncio
import sys
import types

import pytest

from audio_trombone.kany import HEADER_SIZE, KanyPacket
from audio_trombone.models import MediaPacket
from audio_trombone.processors.htdemucs import HTDemucsConfig, HTDemucsProcessor
from conftest import install_fake_torchaudio


def make_processor(**kwargs) -> HTDemucsProcessor:
    inference_fn = kwargs.pop("inference_fn", None)
    return HTDemucsProcessor(config=HTDemucsConfig(**kwargs), inference_fn=inference_fn)


def make_media_packet(sequence: int, samples: list[float]) -> MediaPacket:
    channels = 2
    frames = len(samples) // channels
    header = bytearray(HEADER_SIZE)
    header[0:4] = b"KANY"
    header[4] = 1
    header[6] = channels
    header[7] = 1
    header[8:12] = (1_000).to_bytes(4, "big")
    header[12:16] = sequence.to_bytes(4, "big")
    header[16:24] = (sequence * 2_000).to_bytes(8, "big")
    header[24:26] = frames.to_bytes(2, "big")
    payload = bytes(header) + array("f", samples).tobytes()
    return MediaPacket.received(payload, "127.0.0.1", 40_000)


async def collect(processor: HTDemucsProcessor, packet: MediaPacket):
    return [output async for output in processor.process(packet)]


def test_vocal_reduction_is_reported_in_diagnostics() -> None:
    processor = make_processor(vocal_reduction=0.5, inference_fn=lambda s, _r, _c: s)

    diagnostics = processor.diagnostics()

    assert diagnostics["vocal_reduction"] == pytest.approx(0.5)
    assert diagnostics["vocal_gain"] == pytest.approx(0.5)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_vocal_reduction_must_be_between_zero_and_one(value: float) -> None:
    with pytest.raises(ValueError, match="vocal_reduction"):
        make_processor(vocal_reduction=value)


@pytest.mark.asyncio
async def test_buffers_inference_and_releases_one_packet_per_input() -> None:
    def remove_everything(samples: array, sample_rate: int, channels: int) -> array:
        assert sample_rate == 1_000
        assert channels == 2
        return array("f", [0.0] * len(samples))

    processor = make_processor(
        segment_seconds=0.004,
        inference_fn=remove_everything,
    )
    await processor.start()

    assert await collect(processor, make_media_packet(0, [0.5, 0.5, 0.5, 0.5])) == []
    assert await collect(processor, make_media_packet(1, [0.5, 0.5, 0.5, 0.5])) == []

    for _ in range(50):
        await asyncio.sleep(0.002)
        if processor.diagnostics()["inference_running"] is False:
            break

    first = await collect(processor, make_media_packet(2, [0.5, 0.5, 0.5, 0.5]))
    second = await collect(processor, make_media_packet(3, [0.5, 0.5, 0.5, 0.5]))

    assert len(first) == 1
    assert len(second) == 1
    assert KanyPacket.decode(first[0].payload).sequence == 0
    assert KanyPacket.decode(second[0].payload).sequence == 1
    assert list(KanyPacket.decode(first[0].payload).samples) == pytest.approx([0.0] * 4)
    assert processor.diagnostics()["segments_completed"] == 1

    await processor.stop()


def test_segment_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="segment_seconds"):
        make_processor(segment_seconds=0)


@pytest.mark.parametrize("value", [-0.01, 1.0])
def test_overlap_must_be_between_zero_and_one(value: float) -> None:
    with pytest.raises(ValueError, match="overlap"):
        make_processor(overlap=value)


@pytest.mark.asyncio
async def test_start_loads_separator_when_no_inference_fn_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = make_processor()
    calls = []
    monkeypatch.setattr(processor, "_load_separator", lambda: calls.append(True))

    await processor.start()

    assert calls == [True]


@pytest.mark.asyncio
async def test_process_rejects_non_kany_payload() -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    packet = MediaPacket.received(b"not-a-kany-packet", "127.0.0.1", 40_000)

    with pytest.raises(ValueError, match="requires KANY v1 f32 PCM packets"):
        await collect(processor, packet)


@pytest.mark.asyncio
async def test_process_rejects_non_stereo_input() -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    header = bytearray(HEADER_SIZE)
    header[0:4] = b"KANY"
    header[4] = 1
    header[6] = 1
    header[7] = 1
    header[8:12] = (1_000).to_bytes(4, "big")
    header[12:16] = (0).to_bytes(4, "big")
    header[16:24] = (0).to_bytes(8, "big")
    header[24:26] = (2).to_bytes(2, "big")
    payload = bytes(header) + array("f", [0.1, 0.2]).tobytes()
    packet = MediaPacket.received(payload, "127.0.0.1", 40_000)

    with pytest.raises(ValueError, match="requires stereo input"):
        await collect(processor, packet)


@pytest.mark.asyncio
async def test_flush_drains_ready_output() -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    decoded = KanyPacket.decode(make_media_packet(0, [0.1, 0.2, 0.3, 0.4]).payload)
    from audio_trombone.models import ProcessedPacket

    processor._ready_output.append(
        ProcessedPacket(payload=decoded.encode_samples(decoded.samples))
    )

    flushed = [item async for item in processor.flush()]

    assert len(flushed) == 1
    assert len(processor._ready_output) == 0


def test_accept_stream_format_rejects_change_without_reset() -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    first = KanyPacket.decode(make_media_packet(0, [0.1, 0.2, 0.3, 0.4]).payload)
    processor._accept_stream_format(first)

    changed_header = bytearray(HEADER_SIZE)
    changed_header[0:4] = b"KANY"
    changed_header[4] = 1
    changed_header[6] = 2
    changed_header[7] = 1
    changed_header[8:12] = (2_000).to_bytes(4, "big")
    changed_header[12:16] = (0).to_bytes(4, "big")
    changed_header[16:24] = (0).to_bytes(8, "big")
    changed_header[24:26] = (2).to_bytes(2, "big")
    changed_payload = bytes(changed_header) + array("f", [0.1, 0.2, 0.3, 0.4]).tobytes()
    changed = KanyPacket.decode(changed_payload)

    with pytest.raises(
        ValueError, match="audio format changed without processor reset"
    ):
        processor._accept_stream_format(changed)


def test_target_segment_frames_requires_known_stream_format() -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    with pytest.raises(RuntimeError, match="stream format is not known"):
        processor._target_segment_frames()


@pytest.mark.asyncio
async def test_harvest_inference_reraises_cancelled_error() -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    decoded = KanyPacket.decode(make_media_packet(0, [0.1, 0.2, 0.3, 0.4]).payload)
    processor._active_packets = [decoded]

    async def hang() -> array:
        await asyncio.sleep(10)
        return array("f")

    task = asyncio.create_task(hang())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    processor._inference_task = task

    with pytest.raises(asyncio.CancelledError):
        await processor._harvest_inference()

    assert processor._active_packets == []
    assert processor._inference_task is None


@pytest.mark.asyncio
async def test_harvest_inference_wraps_inference_failure() -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    decoded = KanyPacket.decode(make_media_packet(0, [0.1, 0.2, 0.3, 0.4]).payload)
    processor._active_packets = [decoded]

    async def boom() -> array:
        raise ValueError("boom")

    task = asyncio.create_task(boom())
    await asyncio.sleep(0)
    processor._inference_task = task

    with pytest.raises(RuntimeError, match="HTDemucs inference failed"):
        await processor._harvest_inference()

    assert processor.last_error == "boom"
    assert processor._active_packets == []


@pytest.mark.asyncio
async def test_harvest_inference_rejects_sample_count_mismatch() -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    decoded = KanyPacket.decode(make_media_packet(0, [0.1, 0.2, 0.3, 0.4]).payload)
    processor._active_packets = [decoded]

    async def wrong_length() -> array:
        return array("f", [0.0, 0.0, 0.0])

    task = asyncio.create_task(wrong_length())
    await asyncio.sleep(0)
    processor._inference_task = task

    with pytest.raises(RuntimeError, match="HTDemucs returned"):
        await processor._harvest_inference()

    assert processor._active_packets == []


def _install_fake_demucs(monkeypatch: pytest.MonkeyPatch, instances: list) -> None:
    class FakeSeparator:
        def __init__(self, *, model, device, segment, shifts, split, overlap, progress):
            self.model = model
            self.device = device
            self.segment = segment
            self.shifts = shifts
            self.split = split
            self.overlap = overlap
            self.progress = progress
            self.samplerate = 44_100
            self.audio_channels = 2
            instances.append(self)

    fake_api = types.ModuleType("demucs.api")
    fake_api.Separator = FakeSeparator
    fake_pkg = types.ModuleType("demucs")
    fake_pkg.api = fake_api

    monkeypatch.setitem(sys.modules, "demucs", fake_pkg)
    monkeypatch.setitem(sys.modules, "demucs.api", fake_api)


def test_load_separator_raises_when_demucs_dependencies_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "demucs", raising=False)
    monkeypatch.delitem(sys.modules, "demucs.api", raising=False)
    processor = make_processor()

    with pytest.raises(RuntimeError, match="HTDemucs dependencies are not installed"):
        processor._load_separator()


def test_load_separator_auto_selects_cpu_when_no_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    instances: list = []
    _install_fake_demucs(monkeypatch, instances)
    processor = make_processor(model_name="htdemucs", device="auto")

    processor._load_separator()

    assert processor._device == "cpu"
    assert len(instances) == 1
    assert instances[0].model == "htdemucs"


def test_load_separator_honours_explicit_cpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list = []
    _install_fake_demucs(monkeypatch, instances)
    processor = make_processor(device="cpu")

    processor._load_separator()

    assert processor._device == "cpu"


def test_load_separator_rejects_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    instances: list = []
    _install_fake_demucs(monkeypatch, instances)
    processor = make_processor(device="cuda")

    with pytest.raises(RuntimeError, match="cannot see a GPU"):
        processor._load_separator()


def test_run_inference_without_separator_raises() -> None:
    processor = make_processor()
    with pytest.raises(RuntimeError, match="Demucs separator is not loaded"):
        processor._run_inference(array("f", [0.0, 0.0]), 1_000, 2)


def test_run_inference_runs_real_torch_path_with_resample_and_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class FakeSeparator:
        samplerate = 2_000

        def separate_tensor(self, waveform, *, sr):
            vocals = torch.zeros_like(waveform)
            return waveform, {"vocals": vocals}

    install_fake_torchaudio(
        monkeypatch, resample=lambda tensor, orig_freq, new_freq: tensor[..., :-1]
    )

    processor = make_processor(vocal_reduction=1.0)
    processor._separator = FakeSeparator()

    samples = array("f", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    result = processor._run_inference(samples, sample_rate=1_000, channels=2)

    assert len(result) == len(samples)
    assert all(-1.0 <= value <= 1.0 for value in result)


def test_run_inference_skips_resample_when_rates_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class FakeSeparator:
        samplerate = 1_000

        def separate_tensor(self, waveform, *, sr):
            vocals = torch.zeros_like(waveform)
            return waveform, {"vocals": vocals}

    def unexpected_resample(*args, **kwargs):
        raise AssertionError("resample should not be called when rates match")

    install_fake_torchaudio(monkeypatch, resample=unexpected_resample)

    processor = make_processor(vocal_reduction=1.0)
    processor._separator = FakeSeparator()

    samples = array("f", [0.1, 0.2, 0.3, 0.4])
    result = processor._run_inference(samples, sample_rate=1_000, channels=2)

    assert len(result) == len(samples)
    assert list(result) == pytest.approx(list(samples), abs=1e-6)


def test_run_inference_raises_when_vocals_stem_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSeparator:
        samplerate = 1_000

        def separate_tensor(self, waveform, *, sr):
            return waveform, {}

    install_fake_torchaudio(
        monkeypatch, resample=lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )

    processor = make_processor()
    processor._separator = FakeSeparator()

    with pytest.raises(RuntimeError, match="did not return a vocals stem"):
        processor._run_inference(array("f", [0.1, 0.2, 0.3, 0.4]), 1_000, 2)
