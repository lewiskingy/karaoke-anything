import pytest
import torch
from audio_trombone.vendor.clarity_tasnet import (
    ChannelwiseLayerNorm,
    Chomp1d,
    ConvTasNetStereo,
    GlobalLayerNorm,
    TemporalConvNet,
    chose_norm,
    overlap_and_add,
)
from torch import nn


def _tiny_kwargs(**overrides):
    kwargs = {
        "N": 4,
        "L": 4,
        "B": 4,
        "H": 4,
        "P": 3,
        "X": 1,
        "R": 1,
        "C": 2,
        "audio_channels": 2,
        "samplerate": 8_000,
    }
    kwargs.update(overrides)
    return kwargs


def test_forward_pads_output_to_match_input_length() -> None:
    model = ConvTasNetStereo(**_tiny_kwargs())
    mixture = torch.randn(1, 2, 37)

    output = model(mixture)

    assert output.shape == (1, 2, 2, 37)


def test_valid_length_returns_input_unchanged() -> None:
    model = ConvTasNetStereo(**_tiny_kwargs())
    assert model.valid_length(123) == 123


def test_causal_model_uses_chomp1d_and_matches_input_length() -> None:
    model = ConvTasNetStereo(**_tiny_kwargs(causal=True))
    mixture = torch.randn(1, 2, 29)

    output = model(mixture)

    assert output.shape == (1, 2, 2, 29)
    assert any(isinstance(module, Chomp1d) for module in model.modules())


def test_temporal_conv_net_softmax_branch_sums_to_one() -> None:
    network = TemporalConvNet(
        N=4, B=4, H=4, P=3, X=1, R=1, C=3, mask_nonlinear="softmax"
    )
    mixture_w = torch.randn(1, 4, 10)

    mask = network(mixture_w)

    assert mask.shape == (1, 3, 4, 10)
    sums = mask.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_temporal_conv_net_rejects_unsupported_mask_nonlinear() -> None:
    network = TemporalConvNet(N=4, B=4, H=4, P=3, X=1, R=1, C=2, mask_nonlinear="bogus")
    mixture_w = torch.randn(1, 4, 10)

    with pytest.raises(ValueError, match="Unsupported mask non-linear function"):
        network(mixture_w)


def test_chomp1d_zero_chomp_returns_value_unchanged() -> None:
    chomp = Chomp1d(0)
    value = torch.randn(1, 2, 5)

    result = chomp(value)

    assert torch.equal(result, value)


def test_chomp1d_trims_trailing_padding() -> None:
    chomp = Chomp1d(2)
    value = torch.randn(1, 2, 5)

    result = chomp(value)

    assert result.shape == (1, 2, 3)
    assert torch.equal(result, value[:, :, :-2])


def test_chose_norm_returns_expected_types() -> None:
    assert isinstance(chose_norm("gLN", 4), GlobalLayerNorm)
    assert isinstance(chose_norm("cLN", 4), ChannelwiseLayerNorm)
    assert isinstance(chose_norm("id", 4), nn.Identity)
    assert isinstance(chose_norm("other", 4), nn.BatchNorm1d)


def test_channelwise_layer_norm_normalises_across_channels() -> None:
    norm = ChannelwiseLayerNorm(4)
    value = torch.randn(2, 4, 6)

    output = norm(value)

    assert output.shape == value.shape
    mean = output.mean(dim=1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)


def test_global_layer_norm_normalises_across_channels_and_time() -> None:
    norm = GlobalLayerNorm(4)
    value = torch.randn(2, 4, 6)

    output = norm(value)

    assert output.shape == value.shape


def test_overlap_and_add_matches_reference_implementation() -> None:
    signal = torch.randn(1, 1, 3, 5)
    frame_step = 2

    result = overlap_and_add(signal, frame_step)

    frames, frame_length = signal.shape[-2:]
    output_len = frame_step * (frames - 1) + frame_length
    expected = torch.zeros(*signal.shape[:-2], output_len)
    for index in range(frames):
        start = index * frame_step
        expected[..., start : start + frame_length] += signal[..., index, :]

    assert torch.allclose(result, expected, atol=1e-6)
