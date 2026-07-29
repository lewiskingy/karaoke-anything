# Derived from claritychallenge/clarity
# recipes/cad2/task1/ConvTasNet/local/tasnet.py
# Commit: 9df6486fb0bddc7619b3b99f1b3a5c72c109a3ec
# Original implementation includes the MIT licence notice below.
#
# The MIT License (MIT)
# Copyright (c) 2018 Kaituo XU
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

EPS = 1e-8


def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length
    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    frame = torch.arange(0, output_subframes, device=signal.device).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = frame.long().contiguous().view(-1)
    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    return result.view(*outer_dimensions, -1)


class ConvTasNetStereo(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        N=256,
        L=20,
        B=256,
        H=512,
        P=3,
        X=8,
        R=4,
        C=4,
        audio_channels=1,
        samplerate=44100,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
    ):
        super().__init__()
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = (
            N,
            L,
            B,
            H,
            P,
            X,
            R,
            C,
        )
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        self.encoder = Encoder(L, N, audio_channels)
        self.separator = TemporalConvNet(
            N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear
        )
        self.decoder = Decoder(N, L, audio_channels)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)

    def valid_length(self, length):
        return length

    def forward(self, mixture):
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)
        origin = mixture.size(-1)
        current = est_source.size(-1)
        return F.pad(est_source, (0, origin - current))


class Encoder(nn.Module):
    def __init__(self, L, N, audio_channels):
        super().__init__()
        self.conv1d_U = nn.Conv1d(
            audio_channels, N, kernel_size=L, stride=L // 2, bias=False
        )

    def forward(self, mixture):
        return F.relu(self.conv1d_U(mixture))


class Decoder(nn.Module):
    def __init__(self, N, L, audio_channels):
        super().__init__()
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)

    def forward(self, mixture_w, est_mask):
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask
        source_w = torch.transpose(source_w, 2, 3)
        est_source = self.basis_signals(source_w)
        m, c, k, _ = est_source.size()
        est_source = (
            est_source.view(m, c, k, self.audio_channels, -1)
            .transpose(2, 3)
            .contiguous()
        )
        return overlap_and_add(est_source, self.L // 2)


class TemporalConvNet(nn.Module):
    def __init__(
        self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear="relu"
    ):
        super().__init__()
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        repeats = []
        for _ in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks.append(
                    TemporalBlock(
                        B,
                        H,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                )
            repeats.append(nn.Sequential(*blocks))
        self.network = nn.Sequential(
            ChannelwiseLayerNorm(N),
            nn.Conv1d(N, B, 1, bias=False),
            nn.Sequential(*repeats),
            nn.Conv1d(B, C * N, 1, bias=False),
        )

    def forward(self, mixture_w):
        batch, channels, frames = mixture_w.size()
        score = self.network(mixture_w).view(batch, self.C, channels, frames)
        if self.mask_nonlinear == "softmax":
            return F.softmax(score, dim=1)
        if self.mask_nonlinear == "relu":
            return F.relu(score)
        raise ValueError("Unsupported mask non-linear function")


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.PReLU(),
            chose_norm(norm_type, out_channels),
            DepthwiseSeparableConv(
                out_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                norm_type,
                causal,
            ),
        )

    def forward(self, value):
        return self.net(value) + value


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            )
        ]
        if causal:
            layers.append(Chomp1d(padding))
        layers.extend(
            [
                nn.PReLU(),
                chose_norm(norm_type, in_channels),
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, value):
        return self.net(value)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, value):
        if self.chomp_size == 0:
            return value
        return value[:, :, : -self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    if norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    if norm_type == "id":
        return nn.Identity()
    return nn.BatchNorm1d(channel_size)


class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))

    def forward(self, value):
        mean = torch.mean(value, dim=1, keepdim=True)
        variance = torch.var(value, dim=1, keepdim=True, unbiased=False)
        return self.gamma * (value - mean) / torch.pow(variance + EPS, 0.5) + self.beta


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))

    def forward(self, value):
        mean = value.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        variance = (
            ((value - mean) ** 2).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        )
        return self.gamma * (value - mean) / torch.pow(variance + EPS, 0.5) + self.beta
