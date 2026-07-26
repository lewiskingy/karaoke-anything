# Architecture

## Purpose

Karaoke Anything is a low-latency audio-processing system that captures audio from an arbitrary desktop source, transports it to a server, applies a selectable audio-processing pipeline, and returns the resulting audio for local playback.

The first end-to-end milestone deliberately performs no audible processing. It proves device routing, capture, packetisation, network transport, buffering and playback before a source-separation model is introduced.

## System context

```text
Application audio
  -> virtual audio cable
  -> Rust client capture
  -> UDP PCM datagrams
  -> Karaoke Anything server
  -> selected AudioProcessor
  -> UDP PCM datagrams
  -> Rust client jitter buffer
  -> physical playback device
```

The intended first deployment is a Windows laptop on a trusted home LAN and a Linux Docker host with a GPU available for later model inference.

## Components

### Virtual audio cable

A third-party virtual audio device, initially VB-CABLE on Windows, receives audio from Spotify, a browser, OpenKJ or another application. The project does not implement a kernel audio driver.

### Rust client

The client uses CPAL for cross-platform audio-device access. It:

1. lists and selects capture and playback devices;
2. captures interleaved PCM samples;
3. normalises samples to 32-bit floating point;
4. packetises the samples using the versioned protocol in `docs/protocol.md`;
5. sends datagrams to the server;
6. receives returned datagrams;
7. places received samples into a bounded playback buffer; and
8. renders them to the selected physical output device.

The first client is a command-line application. A tray application or graphical configuration surface is a later concern.

### Python server

The server owns UDP ingress and egress, a bounded input queue, processor lifecycle and observability endpoints.

The current server treats each datagram as an opaque payload and forwards it through an `AudioProcessor`. This preserves byte-for-byte passthrough while establishing lifecycle semantics required by future models:

- `start`: load a model or allocate accelerator resources;
- `process`: accept work and emit zero, one or many outputs;
- `reset`: discard state after seek, reconnect or track change;
- `flush`: emit buffered output before replacement or shutdown; and
- `stop`: release resources.

### Audio processor registry

The registry provides named processor selection. Initially it contains:

- `passthrough`: returns each datagram unchanged;
- `delay-passthrough`: introduces a small artificial delay for queue testing; and
- `null`: emits nothing for timeout and failover testing.

## Architectural boundaries

### Transport and audio processing

The current `AudioProcessor` still receives opaque datagrams. This is an intentional transitional design, not the desired model-inference boundary.

Before integrating a source separator, the server should introduce these layers:

```text
UDP datagram
  -> protocol validation
  -> sequencing and loss handling
  -> timestamped PCM frames
  -> AudioFrameProcessor pipeline
  -> packetisation
  -> UDP datagram
```

At that point:

- transport code owns headers, sequencing, timing and packet loss;
- audio-frame code owns channel layout, sample rate and PCM buffers; and
- model adapters own only inference state and audio transformation.

This prevents networking concerns from leaking into separator implementations.

### Client and server responsibilities

The client owns:

- desktop audio-device selection;
- capture and playback clock interaction;
- virtual-device routing;
- packet creation and validation;
- the receive jitter buffer; and
- user-facing diagnostics.

The server owns:

- processor selection and lifecycle;
- model loading and accelerator use;
- server-side buffering required by a model;
- audio transformation; and
- processing metrics.

The server does not attempt to control Windows audio routing.

## Latency budget

The target is interactive karaoke rather than offline stem production. End-to-end latency should be measured as the sum of:

- capture-device buffering;
- client packet accumulation;
- outbound LAN transit;
- server queueing;
- model look-ahead and inference;
- inbound LAN transit;
- client jitter buffering; and
- playback-device buffering.

The passthrough milestone establishes a baseline. Model evaluation must report additional latency relative to that baseline, not only inference duration.

## Failure behaviour

The initial implementation favours bounded memory and visible degradation:

- client playback buffers are bounded;
- server ingress queues are bounded;
- excess packets are dropped rather than growing memory indefinitely;
- malformed protocol packets are ignored;
- missing output results in silence rather than replaying stale audio; and
- processor reset is explicit after discontinuities.

Future work should add sequence-loss metrics and client reconnection state.

## Security boundary

The first implementation is intended only for a trusted home LAN. UDP media is unauthenticated and unencrypted. HTTP control endpoints are also unauthenticated.

Do not expose the service directly to the internet. Any remote deployment should add an authenticated session-control plane and an encrypted media transport or trusted tunnel.

## Delivery phases

### Phase 1: transport foundation

- UDP server passthrough
- bounded queue
- processor registry
- health and metrics

### Phase 2: end-to-end audio passthrough

- virtual audio device
- Rust capture and playback client
- versioned PCM datagram protocol
- receive buffer and device diagnostics

### Phase 3: explicit PCM frame boundary

- packet parsing and sequencing on the server
- timestamped audio frames
- format negotiation or fixed-session configuration
- processor-chain abstraction

### Phase 4: model reference implementation

- integrate one separator adapter
- measure added latency, GPU use and quality
- retain passthrough as the transport baseline

### Phase 5: low-latency model evaluation

- compare causal and windowed separators
- evaluate chunk sizes and overlap
- add processor chaining, limiter and optional vocal attenuation controls

## Design principles

1. Prove transport independently of AI inference.
2. Reuse operating-system virtual audio devices rather than writing drivers.
3. Keep networking, PCM framing and model inference as separate concerns.
4. Bound every queue and buffer.
5. Preserve passthrough as a permanent diagnostic mode.
6. Measure end-to-end latency from capture to audible playback.
7. Optimise for replaceable processor adapters rather than one chosen model.
