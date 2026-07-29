# Contributor guidance

This repository contains a low-latency, containerised karaoke audio pipeline. Changes should preserve the separation between transport, audio framing, processor lifecycle, model-specific inference and user-facing runtime control.

## Read this first

Before changing code, read:

1. `README.md` for deployment, supported processors and development commands.
2. `docs/architecture.md` for system boundaries and design principles.
3. `docs/protocol.md` for the KANY UDP PCM contract.
4. The model-specific document under `docs/` for any processor being changed.
5. `docs/mdx23c-stage-0.md` when working on MDX23C.

Repository documentation is part of the implementation. Update it in the same change whenever behaviour, configuration, model provenance, build steps or architectural intent changes.

## Current architecture

```text
Windows source application
  -> virtual audio cable
  -> Rust client capture and packetisation
  -> KANY v1 UDP packets
  -> Python server ingress and bounded queue
  -> selected AudioProcessor
  -> paced KANY output packets
  -> Rust client jitter buffer and playback
```

The Python server owns processor lifecycle, model loading, server-side model buffering, runtime settings and diagnostics. The Rust client owns local audio devices, capture/playback clocks, packet creation and the playback jitter buffer.

Processors implement the common lifecycle:

- `start()` allocates resources or loads a model;
- `process()` accepts one media packet and may emit zero, one or many packets;
- `reset()` discards state after discontinuity;
- `flush()` emits queued output where meaningful;
- `stop()` releases resources;
- `diagnostics()` exposes processor-specific status without changing control-plane contracts.

## Boundaries that must be preserved

- Do not put Windows audio-device control in the server.
- Do not put UDP socket ownership, HTTP routing or Compose concerns inside model adapters.
- Do not put model-specific fields into the KANY protocol.
- Do not allow model buffers or queues to grow without a bound.
- Do not silently download model assets at runtime. GPU image builds should acquire pinned assets and runtime loading should be offline.
- Do not register an experimental processor until its model can be constructed and loaded reliably.
- Do not claim successful model compatibility unless an executable smoke test proves it.
- Keep `passthrough` available as the transport baseline and diagnostic mode.
- Keep runtime changes ephemeral: Compose/environment values remain startup defaults after container restart.

## Model integration sequence

New model families should normally be delivered in explicit stages:

0. **Asset and checkpoint proof**: pin provenance, download assets, construct the exact architecture, load the checkpoint strictly and fail the build on incompatibility. No processor registration.
1. **Inference adapter proof**: run deterministic offline inference against a short fixture and establish input/output shapes, sample rate and stem semantics.
2. **Buffered processor integration**: connect inference to KANY buffering, packet reconstruction, pacing, lifecycle and diagnostics.
3. **Runtime controls**: add settings dataclasses, validation, API schema, restore-default behaviour and console controls.
4. **Target-host validation**: measure inference duration, real-time factor, latency, VRAM, output quality and reset/track-change behaviour.

Do not collapse these stages when the checkpoint/architecture pairing is not already proven.

## Runtime settings rules

`Settings.from_environment()` defines startup defaults. `TromboneService.runtime_settings()` is the control-plane view. `PATCH /api/settings` applies changes for the current process only; `DELETE /api/settings` restores startup defaults.

A processor-specific setting must be represented consistently in:

- `app/audio_trombone/config.py`;
- the processor registry constructor;
- service allow-list, validation, runtime settings and restore-default handling;
- `app/main.py` request models and flattening;
- `app/settings.html` when it is intended to be user-editable;
- Compose/environment documentation;
- processor diagnostics.

Only settings that are demonstrably safe to mutate on the active processor should be applied live. Changes to architecture, model path, device, segmenting or source semantics normally require constructing a replacement processor and switching it under the processor lock.

## Docker and model assets

The GPU image is currently `Dockerfile.demucs`, despite supporting more than Demucs. Retain that filename unless a separately scoped change renames it everywhere.

- CUDA/PyTorch compatibility with the RTX 5070 Ti (`sm_120`) is intentional.
- Pin external repositories and model revisions by immutable commit where practical.
- Keep downloads in dedicated `/models/<model-family>` paths.
- Add a build-time offline smoke test for every baked model.
- A smoke test must use the same model construction and state-dict normalisation intended for later runtime use.
- Avoid copying entire training, GUI or competition repositories into the service. Vendor the smallest necessary inference surface with attribution and licence notices.

## Testing and completion

For ordinary Python changes run `pytest`. For client changes run `cargo fmt --check`, `cargo llvm-cov --ignore-filename-regex "main\.rs$" --fail-under-lines 100` and `cargo build --release`.

`app/` is held at 100% statement coverage; `pytest` fails the run if coverage drops below that (`--cov-fail-under=100` in `pyproject.toml`). `client/src/protocol.rs` and `client/src/network.rs` are held at 100% line coverage the same way via `cargo-llvm-cov`; `client/src/main.rs` (cpal device/stream glue, `run()`, `main()`) is excluded from that gate since it has no mock backend for real audio hardware. New code needs tests in the same change, not a follow-up. A pre-commit hook (`pre-commit install`, config in `.pre-commit-config.yaml`) runs both coverage checks before each commit.

A model stage is not complete merely because code imports. Record the exact evidence required by its model-specific document. When target GPU execution is unavailable in the development environment, state that limitation explicitly and provide a reproducible command for the owner to run.

## MDX23C scope now

The current MDX23C task is **Stage 0 only**. See `docs/mdx23c-stage-0.md`.

Do not add `mdx23c-vocals` to the processor registry, runtime settings API, console or Compose runtime environment during Stage 0. The deliverable is a pinned, reproducible build/test path proving that the selected YAML can construct a matching MDX23C model and that the selected checkpoint loads strictly and offline.