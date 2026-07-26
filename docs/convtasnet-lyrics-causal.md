# Causal Conv-TasNet lyrics processor

## Purpose

`convtasnet-lyrics-causal` is an experimental GPU processor for low-latency karaoke vocal reduction. It uses the pretrained Cadenza model `cadenzachallenge/ConvTasNet_Lyrics_Causal` and the matching `ConvTasNetStereo` architecture from the Clarity Challenge repository.

The model is causal and trained for stereo lyrics/accompaniment separation at 44.1 kHz. This first integration still uses bounded segment buffering so it can share the proven KANY packet lifecycle and paced output behaviour already used by the HTDemucs processor. Once performance and source ordering are validated, the segment size can be reduced and a genuinely stateful streaming execution path can be considered.

## Model and architecture provenance

- Weights: `cadenzachallenge/ConvTasNet_Lyrics_Causal` on Hugging Face.
- Architecture source: `claritychallenge/clarity`, file `recipes/cad2/task1/ConvTasNet/local/tasnet.py`, commit `9df6486fb0bddc7619b3b99f1b3a5c72c109a3ec`.
- The minimal inference architecture is vendored in `app/audio_trombone/vendor/clarity_tasnet.py` with attribution and the original MIT notice.

The Docker build downloads the complete Hugging Face snapshot and then performs an offline model-load smoke test. Runtime model loading uses `local_files_only=True` and therefore does not require network access.

## Selecting the processor

```bash
PROCESSOR=convtasnet-lyrics-causal \
CONVTASNET_SEGMENT_SECONDS=1.0 \
CONVTASNET_VOCAL_REDUCTION=1.0 \
docker compose -f compose.yaml -f compose.demucs.yaml up -d --build
```

The GPU image is still named `Dockerfile.demucs` for compatibility, but it now contains both Demucs and Conv-TasNet dependencies and model assets.

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `CONVTASNET_MODEL_REPO` | `cadenzachallenge/ConvTasNet_Lyrics_Causal` | Hugging Face model repository downloaded during image build. |
| `CONVTASNET_MODEL_REVISION` | `main` | Build-time Hugging Face revision. Pin this to a commit for reproducible production builds. |
| `CONVTASNET_MODEL_PATH` | `/models/convtasnet-lyrics-causal` | Local runtime model directory. |
| `CONVTASNET_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or a specific CUDA device. |
| `CONVTASNET_SEGMENT_SECONDS` | `1.0` | Initial packet buffer duration before inference. |
| `CONVTASNET_VOCAL_REDUCTION` | `1.0` | `0.0` keeps the original mix; `1.0` uses the estimated accompaniment only. |
| `CONVTASNET_VOCAL_SOURCE_INDEX` | `0` | Model output index assumed to contain lyrics/vocals. |
| `CONVTASNET_ACCOMPANIMENT_SOURCE_INDEX` | `1` | Model output index assumed to contain accompaniment. |

The source indexes are configurable because the published model must be validated against real audio before their semantic order is treated as final.

## Audio behaviour

The processor accepts KANY v1 stereo float32 PCM only.

For each segment it:

1. preserves the original packet boundaries;
2. resamples to the model sample rate when needed;
3. runs `ConvTasNetStereo` inference on the selected device;
4. reads the configured vocal and accompaniment sources;
5. blends between original and accompaniment according to `CONVTASNET_VOCAL_REDUCTION`;
6. resamples back to the stream rate;
7. clamps to float32 PCM range;
8. reconstructs and releases one output packet per subsequent input packet.

## Validation

Build without cache after changing the model revision or Python dependencies:

```bash
docker compose -f compose.yaml -f compose.demucs.yaml build --no-cache
```

Verify the processor and baked model:

```bash
docker compose -f compose.yaml -f compose.demucs.yaml run --rm karaoke-anything \
  python3 -c "from audio_trombone.vendor.clarity_tasnet import ConvTasNetStereo; m=ConvTasNetStereo.from_pretrained('/models/convtasnet-lyrics-causal', local_files_only=True); print(m.samplerate, m.audio_channels, m.C, m.causal, m.norm_type)"
```

Start with the new processor and inspect diagnostics:

```bash
PROCESSOR=convtasnet-lyrics-causal \
docker compose -f compose.yaml -f compose.demucs.yaml up -d

curl http://localhost:8080/status
```

Confirm:

- the model loads on CUDA;
- `last_real_time_factor` remains below `1.0` during sustained playback;
- vocals are reduced rather than accompaniment;
- stereo orientation is preserved;
- there are no discontinuities at segment boundaries;
- reset works after seeks and track changes.

If the model outputs are reversed, swap the two source-index variables. If the result is unstable at one second, increase `CONVTASNET_SEGMENT_SECONDS`; if performance has ample margin, reduce it gradually.

## Current limitations

- The architecture is causal, but this implementation does not yet preserve internal convolution state between calls. It runs finite causal segments.
- The default build revision is `main`; production deployment should pin a Hugging Face commit.
- Source ordering requires runtime validation.
- The settings HTTP API and web UI have not yet gained model-specific editable controls; environment variables and processor selection are supported.
- Subjective quality and true end-to-end latency must be compared with `htdemucs-vocals` on the target GPU.
