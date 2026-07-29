from array import array
import asyncio
import sys

import pytest
import torch

from audio_trombone.kany import HEADER_SIZE, KanyPacket
from audio_trombone.models import MediaPacket, ProcessedPacket
from audio_trombone.processors.convtasnet_lyrics import (
    ConvTasNetLyricsConfig,
    ConvTasNetLyricsProcessor,
)
from conftest import install_fake_torchaudio


def make_processor(**kwargs) -> ConvTasNetLyricsProcessor:
    inference_fn = kwargs.pop("inference_fn", None)
    return ConvTasNetLyricsProcessor(
        config=ConvTasNetLyricsConfig(**kwargs), inference_fn=inference_fn
    )


def make_media_packet(
    sequence: int, samples: list[float], sample_rate: int = 1_000
) -> MediaPacket:
    channels = 2
    frames = len(samples) // channels
    header = bytearray(HEADER_SIZE)
    header[0:4] = b"KANY"
    header[4] = 1
    header[6] = channels
    header[7] = 1
    header[8:12] = sample_rate.to_bytes(4, "big")
    header[12:16] = sequence.to_bytes(4, "big")
    header[16:24] = (sequence * 2_000).to_bytes(8, "big")
    header[24:26] = frames.to_bytes(2, "big")
    payload = bytes(header) + array("f", samples).tobytes()
    return MediaPacket.received(payload, "127.0.0.1", 40_000)


async def collect(processor: ConvTasNetLyricsProcessor, packet: MediaPacket):
    return [output async for output in processor.process(packet)]


def test_vocal_reduction_is_reported_in_diagnostics() -> None:
    processor = make_processor(vocal_reduction=0.5, inference_fn=lambda s, _r, _c: s)

    diagnostics = processor.diagnostics()

    assert diagnostics["vocal_reduction"] == pytest.approx(0.5)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_vocal_reduction_must_be_between_zero_and_one(value: float) -> None:
    with pytest.raises(ValueError, match="vocal_reduction"):
        make_processor(vocal_reduction=value)


def test_segment_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="segment_seconds"):
        make_processor(segment_seconds=0)


@pytest.mark.parametrize(
    "kwargs", [{"vocal_source_index": -1}, {"accompaniment_source_index": -1}]
)
def test_source_indexes_must_be_non_negative(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="source indexes"):
        make_processor(**kwargs)


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


@pytest.mark.asyncio
async def test_start_loads_model_when_no_inference_fn_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = make_processor()
    calls = []
    monkeypatch.setattr(processor, "_load_model", lambda: calls.append(True))

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

    changed = KanyPacket.decode(
        make_media_packet(1, [0.1, 0.2, 0.3, 0.4], sample_rate=2_000).payload
    )

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


async def _boom() -> array:
    raise ValueError("boom")


async def _wrong_length() -> array:
    return array("f", [0.0, 0.0, 0.0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_task,match",
    [
        (_boom, "ConvTasNet inference failed"),
        (_wrong_length, "ConvTasNet returned"),
    ],
)
async def test_harvest_inference_wraps_failed_or_malformed_result(
    failing_task, match
) -> None:
    processor = make_processor(inference_fn=lambda s, _r, _c: s)
    decoded = KanyPacket.decode(make_media_packet(0, [0.1, 0.2, 0.3, 0.4]).payload)
    processor._active_packets = [decoded]

    task = asyncio.create_task(failing_task())
    await asyncio.sleep(0)
    processor._inference_task = task

    with pytest.raises(RuntimeError, match=match):
        await processor._harvest_inference()

    assert processor._active_packets == []


def test_load_model_raises_when_dependencies_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "audio_trombone.vendor.clarity_tasnet", None)
    processor = make_processor()

    with pytest.raises(RuntimeError, match="ConvTasNet dependencies are not installed"):
        processor._load_model()


def _tiny_model():
    from audio_trombone.vendor.clarity_tasnet import ConvTasNetStereo

    return ConvTasNetStereo(
        N=4, L=4, B=4, H=4, P=3, X=1, R=1, C=2, audio_channels=2, samplerate=8_000
    )


def test_load_model_auto_selects_cpu_when_no_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audio_trombone.vendor.clarity_tasnet import ConvTasNetStereo

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        ConvTasNetStereo,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _tiny_model()),
    )
    processor = make_processor(device="auto")

    processor._load_model()

    assert processor._device == "cpu"
    assert processor._model_sample_rate == 8_000
    assert processor._model is not None


def test_load_model_honours_explicit_cpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audio_trombone.vendor.clarity_tasnet import ConvTasNetStereo

    monkeypatch.setattr(
        ConvTasNetStereo,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _tiny_model()),
    )
    processor = make_processor(device="cpu")

    processor._load_model()

    assert processor._device == "cpu"


def test_load_model_rejects_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from audio_trombone.vendor.clarity_tasnet import ConvTasNetStereo

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        ConvTasNetStereo,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _tiny_model()),
    )
    processor = make_processor(device="cuda")

    with pytest.raises(RuntimeError, match="cannot see a GPU"):
        processor._load_model()


def test_run_inference_without_model_raises() -> None:
    processor = make_processor()
    with pytest.raises(RuntimeError, match="ConvTasNet model is not loaded"):
        processor._run_inference(array("f", [0.0, 0.0]), 1_000, 2)


def _reject_resample(*args, **kwargs):
    raise AssertionError("resample should not be called when rates match")


def _stub_model(shape_fn):
    class FakeModel:
        def __call__(self, waveform):
            return shape_fn(waveform)

    return FakeModel()


def _processor_with_stub_model(shape_fn, **kwargs) -> ConvTasNetLyricsProcessor:
    processor = make_processor(**kwargs)
    processor._model = _stub_model(shape_fn)
    processor._model_sample_rate = 1_000
    processor._device = "cpu"
    return processor


@pytest.mark.parametrize(
    "shape_fn,error_match",
    [
        (lambda waveform: torch.zeros(2, 2), "Expected ConvTasNet output"),
        (
            lambda waveform: torch.zeros(*waveform.shape[:1], 1, *waveform.shape[1:]),
            "source index exceeds model output count",
        ),
    ],
)
def test_run_inference_raises_on_bad_model_output(
    monkeypatch: pytest.MonkeyPatch, shape_fn, error_match: str
) -> None:
    install_fake_torchaudio(monkeypatch, resample=_reject_resample)
    processor = _processor_with_stub_model(
        shape_fn, vocal_source_index=0, accompaniment_source_index=1
    )

    with pytest.raises(RuntimeError, match=error_match):
        processor._run_inference(array("f", [0.1, 0.2, 0.3, 0.4]), 1_000, 2)


def test_run_inference_skips_resample_when_rates_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_torchaudio(monkeypatch, resample=_reject_resample)
    processor = _processor_with_stub_model(
        lambda waveform: torch.zeros(*waveform.shape[:1], 2, *waveform.shape[1:]),
        vocal_reduction=1.0,
    )

    samples = array("f", [0.1, 0.2, 0.3, 0.4])
    result = processor._run_inference(samples, sample_rate=1_000, channels=2)

    assert len(result) == len(samples)


def test_run_inference_runs_real_torch_path_with_resample_and_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_torchaudio(
        monkeypatch, resample=lambda tensor, orig_freq, new_freq: tensor[..., :-1]
    )
    processor = _processor_with_stub_model(
        lambda waveform: torch.zeros(*waveform.shape[:1], 2, *waveform.shape[1:]),
        vocal_reduction=1.0,
    )
    processor._model_sample_rate = 2_000

    samples = array("f", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    result = processor._run_inference(samples, sample_rate=1_000, channels=2)

    assert len(result) == len(samples)
    assert all(-1.0 <= value <= 1.0 for value in result)
