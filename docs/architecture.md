# Architecture

## Purpose

Karaoke Anything is a low-latency audio-processing system that captures audio from an arbitrary desktop source, transports it to a server, applies a selectable audio-processing pipeline, and returns the resulting audio for local playback.

The system is intentionally model-agnostic. Transport, packet framing, processor lifecycle, model inference and runtime control are separate concerns so that separator implementations remain replaceable.

## System context

```text
Application audio
  -> virtual audio cable
  -> Rust client capture
  -> KANY v1 UDP PCM datagrams
  -> Karaoke Anything server
  -> selected AudioProcessor
  -> paced KANY v1 UDP PCM datagrams
  -> Rust client jitter buffer
  -> physical playback device
```

The intended deployment is a Windows laptop on a trusted home LAN and a Linux Docker host with an NVIDIA GPU for model inference.

## Components

### Virtual audio cable

A third-party virtual audio device, initially VB-CABLE on Windows, receives audio from Spotify, a browser, OpenKJ or another application. The project does not implement a kernel audio driver.

### Rust client

The client uses CPAL for cross-platform audio-device access. It:

1. lists and selects capture and playback devices;
2. captures interleaved PCM samples;
3. normalises samples to 32-bit floating point;
4. packetises samples using KANY v1 from `docs/protocol.md`;
5. sends datagrams to the server;
6. receives returned datagrams;
7. places received samples into a bounded playback buffer; and
8. renders them to the selected physical output device.

The client owns local device selection and playback timing. It does not load separator models or manage server-side processor settings.

### Python server

The server owns:

- UDP ingress and egress;
- protocol-bearing media packets;
- a bounded input queue;
- processor lifecycle and replacement;
- model loading and accelerator use;
- model-required server-side buffering;
- paced reconstruction of output packets;
- runtime settings and startup-default restoration;
- health, status, diagnostics and metrics;
- the web settings console.

The server does not attempt to control Windows audio routing.

### Audio processor lifecycle

Every processor implements a common lifecycle:

- `start`: load a model or allocate resources;
- `process`: accept one media packet and emit zero, one or many outputs;
- `reset`: discard state after seek, reconnect or track change;
- `flush`: emit buffered output before replacement or shutdown where meaningful;
- `stop`: release resources;
- `diagnostics`: expose processor-specific status through the common status endpoint.

Processor replacement constructs and starts the replacement before switching under the processor lock. Buffered output from the old processor is flushed before it is stopped.

### Processor registry

The registry is the authoritative mapping from stable processor names to constructors. Current processors are:

- `passthrough`: returns each payload unchanged;
- `delay-passthrough`: introduces a small artificial delay for timing tests;
- `null`: emits nothing for timeout and failover tests;
- `stereo-centre-reduction`: zero-lookahead mid/side vocal reduction;
- `htdemucs-vocals`: buffered HTDemucs vocal reduction;
- `convtasnet-lyrics-causal`: buffered finite-segment execution of the causal Cadenza lyrics/accompaniment model.

A model under investigation is not registered until its architecture/checkpoint compatibility and its processor contract have been proven at the appropriate delivery stage.

## Architectural boundaries

### Transport and model processing

The wire contract is KANY v1: versioned stereo float32 PCM datagrams. Model processors may decode KANY packets because they require PCM, but model-specific code must not own sockets, HTTP routes or Compose configuration.

The effective pipeline is:

```text
UDP datagram
  -> KANY protocol validation and PCM decoding
  -> processor-owned bounded segment buffering where required
  -> model-specific inference
  -> output PCM validation/clamping
  -> reconstruction using original packet boundaries
  -> paced UDP output
```

Responsibility remains split as follows:

- transport code owns UDP ingress/egress, sender/destination information, queues and metrics;
- KANY code owns packet headers, format validation and PCM encode/decode;
- processor lifecycle code owns safe replacement, reset, flush and runtime settings;
- model adapters own model construction, inference semantics and only the buffering required by that model.

Do not introduce model fields into the wire protocol and do not move network concerns into model adapters.

### Client and server responsibilities

The client owns:

- desktop audio-device selection;
- capture and playback clock interaction;
- virtual-device routing;
- packet creation and validation;
- the receive jitter buffer; and
- client-facing device and playback diagnostics.

The server owns:

- processor selection and lifecycle;
- model loading and accelerator use;
- server-side buffering required by a model;
- audio transformation;
- runtime model configuration; and
- processing metrics and diagnostics.

### Runtime control

Startup configuration is read from Compose/environment into immutable `Settings`. The running service may replace its current settings for the lifetime of the process.

- `GET /api/settings` returns current settings and startup defaults.
- `PATCH` or `PUT /api/settings` applies validated runtime changes.
- `DELETE /api/settings` restores startup defaults.
- Restarting the container restores Compose/environment values.

Settings that only alter a safe mutable property of the active processor may apply live. Model path, architecture, device, segmentation or source-semantic changes normally require a new processor instance.

A processor-specific setting must remain consistent across configuration, registry construction, service validation, API schemas, console controls, Compose documentation and diagnostics.

### Model assets and dependencies

External models and architecture code are supply-chain inputs and must be treated explicitly:

- pin model revisions and architecture repositories to immutable commits where practical;
- download model assets during image build rather than at runtime;
- load baked assets offline at runtime;
- fail the image build when a model cannot be constructed or loaded;
- vendor only the minimal inference surface required, with attribution and licence notices;
- avoid embedding complete training, GUI or competition repositories in the service;
- do not use `strict=False` to disguise checkpoint incompatibility.

The GPU image is currently named `Dockerfile.demucs` for historical compatibility, although it supports multiple model families.

## Buffering and latency

The target is interactive karaoke rather than offline stem production. End-to-end latency is the sum of:

- capture-device buffering;
- client packet accumulation;
- outbound LAN transit;
- server queueing;
- processor segment accumulation or model look-ahead;
- inference duration;
- paced output delay;
- inbound LAN transit;
- client jitter buffering; and
- playback-device buffering.

`passthrough` is the permanent baseline. Model evaluation must report additional end-to-end latency relative to that baseline, as well as inference duration and real-time factor.

Finite-window processors currently collect bounded segments and run inference asynchronously. Output is reconstructed according to the original packet boundaries and released at the client's natural packet cadence rather than in a burst.

A causal architecture does not automatically make the current integration truly streaming. The ConvTasNet processor, for example, runs finite causal segments and does not yet preserve internal convolution state between calls.

## Failure behaviour

The implementation favours bounded memory and visible degradation:

- client playback buffers are bounded;
- server ingress queues are bounded;
- processor segment buffers must be bounded;
- excess packets are dropped rather than growing memory indefinitely;
- malformed KANY packets fail visibly;
- missing output results in silence rather than stale replay;
- processor reset is explicit after discontinuities;
- model-loading and inference failures are exposed through logs and diagnostics;
- a replacement processor is not installed until it has started successfully.

## Security boundary

The current implementation is intended only for a trusted home LAN. UDP media and HTTP control endpoints are unauthenticated and unencrypted.

Do not expose the service directly to the internet. Any remote deployment should add an authenticated session-control plane and encrypted media transport or a trusted tunnel.

## Delivery status

### Completed foundations

- UDP/KANY end-to-end PCM transport
- bounded server queue and client playback buffer
- stable processor lifecycle and registry
- health, status, metrics and runtime settings API
- web settings console
- passthrough, delay, null and centre-reduction processors
- buffered HTDemucs processor
- buffered finite-segment causal ConvTasNet processor

### Current model-delivery sequence

New model families are delivered in explicit stages when compatibility is not already known:

#### Stage 0: asset and checkpoint proof

- pin model and architecture provenance;
- download only required assets;
- construct the exact model from configuration;
- load the checkpoint strictly on CPU;
- prove the validation works offline;
- fail the build on incompatibility;
- do not register a processor.

#### Stage 1: offline inference proof

- run a short deterministic fixture;
- establish sample rate, tensor shapes, target names/order and output range;
- prove inference without KANY or runtime controls.

#### Stage 2: processor integration

- add bounded buffering, resampling where required, inference, packet reconstruction, pacing, lifecycle and diagnostics.

#### Stage 3: runtime controls

- add environment/startup settings, validation, API contracts, restore-default behaviour and console controls.

#### Stage 4: target-host validation

- measure GPU compatibility, VRAM, inference time, real-time factor, added latency, output quality, discontinuities and reset behaviour.

The current MDX23C work is Stage 0 only. See `docs/mdx23c-stage-0.md`.

## Design principles

1. Preserve passthrough as a permanent transport baseline.
2. Reuse operating-system virtual audio devices rather than writing drivers.
3. Keep networking, KANY framing, lifecycle and model inference as separate concerns.
4. Bound every queue and model buffer.
5. Pin and validate external model assets reproducibly.
6. Make unsupported compatibility fail early and visibly.
7. Measure end-to-end latency from capture to audible playback.
8. Optimise for replaceable processor adapters rather than one chosen model.
9. Add user-facing runtime controls only after model behaviour is proven.
10. Keep documentation and validation evidence aligned with implementation.