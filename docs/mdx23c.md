# MDX23C delivery and target-host validation

## Provenance and stage status

Stage 0 remains reproducible exactly as recorded in `mdx23c-stage-0.md`: the
8KFFT YAML and checkpoint are pinned to `Politrees/UVR_resources` revision
`0472e5cd4e0f77b95ca2df7a8162992e6799fa49`, and the minimal architecture is
from ZFTurbo's training repository commit
`83d495dfc81b2ede9bc62f4209619f8bdfd14995`. Loading remains offline and strict.

Stage 1 adds `MDX23CAdapter`, whose contract is stereo float PCM shaped
`[batch, 2, frames]` at the YAML's sample rate. It normalises output to
`[batch, stem, 2, frames]`, takes stem order from `training.instruments`, and
pads only short model output before trimming it to the exact input length.
Run a stereo 16-bit WAV proof with:

```bash
python3 -m audio_trombone.tools.infer_mdx23c fixture.wav --device cuda
```

The command writes a WAV named for every YAML-declared stem. The pinned pair's
vocal and instrumental semantics must be confirmed audibly on the target host;
the adapter refuses to guess a missing accompaniment name.

Stage 2 registers `mdx23c-vocals`. It loads the model in `start()`, retains it
on the selected device, runs inference outside the asyncio loop under
`torch.inference_mode()`, reconstructs exact KANY packet boundaries, and paces
ready output one packet per incoming packet. Input is bounded to three segments
and excess oldest packets are counted as drops. Reset cancels inference and
clears input, output, active packets, and crossfade history. MDX23C is **not
causal or zero-lookahead**. The current join treatment crossfades the beginning
of a completed segment against the preceding segment tail; it does not pretend
to make the network stateful.

## Settings and latency

| Environment | Default | Valid values | Application |
|---|---:|---|---|
| `MDX23C_SEGMENT_SECONDS` | `1.0` | `0.25, 0.5, 0.75, 1.0, 1.5, 2.0` | reload/reset |
| `MDX23C_OVERLAP` | `0.25` | `0 <= x < 0.5` | reload/reset |
| `MDX23C_BATCH_SIZE` | `1` | `1` | reload/reset |
| `MDX23C_VOCAL_REDUCTION` | `1.0` | `0..1` | live, next segment |
| `MDX23C_DEVICE` | `auto` | `auto`, `cpu`, `cuda`/device | reload/reset |
| `MDX23C_PRECISION` | `float32` | `float32`, `float16`, `bfloat16` | reload/reset; reduced modes CUDA only |

Checkpoint and YAML paths are deployment-only fields and are deliberately not
in the ordinary console. Effective algorithmic latency is approximately
`segment duration + inference time + paced output queue`, plus transport and
client jitter buffering. Diagnostics expose each measurable component, RTF,
drops, underruns, device, precision, and load state. Smaller segments reduce
buffering but context and GPU efficiency may suffer; overlap smooths joins but
costs compute. The 1.0 s default is conservative and **provisional**, not a
target-host measurement.

Suggested starting points—not validated recommendations—are 0.5 s / 0.15
overlap / float32 for low latency and 1.5 s / 0.25 / float32 for quality. A
target-host run must select the eventual default. Boundary coloration, vocal
leakage, loss of centred instruments, and crossfade modulation remain possible.

## Stage 4 benchmark (pending target GPU)

```bash
python3 -m audio_trombone.tools.benchmark_mdx23c --device cuda \
  --segments 0.25,0.5,0.75,1.0,1.5,2.0 --overlaps 0,0.15,0.25
```

It prints readable rows and a final JSON object containing cold load, warm-up,
inference time, RTF, peak CUDA memory, buffering and estimated latency, and
sample-count evidence. Continuity is deliberately reported as null until a
real music fixture is assessed; reset/discontinuity is covered by processor
tests. Stage 4, subjective stem semantics, a measured default, clean image
build, and real-time stability on the RTX 5070 Ti remain owner-run work. Do not
claim completion until sustained RTF is below one with a non-growing queue.
