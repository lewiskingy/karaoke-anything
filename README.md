# Karaoke Anything

A containerised low-latency streaming-audio system designed to turn arbitrary desktop audio sources into karaoke.

The repository now contains a proven end-to-end KANY PCM transport, runtime-selectable audio processors, a web control surface, and two experimental GPU separator integrations:

```text
Application audio
  -> virtual audio cable
  -> Rust capture client
  -> KANY v1 UDP PCM datagrams
  -> Python server
  -> selected AudioProcessor
  -> paced KANY v1 UDP PCM datagrams
  -> Rust playback client
  -> physical output device
```

`passthrough` remains the permanent transport baseline. Model processors are intentionally replaceable and must preserve the same processor lifecycle, bounded-buffer behaviour and control-plane contracts.

## Contributor entry points

Before making changes, read:

- `AGENTS.md`: contributor rules, architectural boundaries and model-delivery stages
- `docs/architecture.md`: current system architecture and ownership boundaries
- `docs/protocol.md`: KANY v1 UDP PCM contract
- `docs/convtasnet-lyrics-causal.md`: ConvTasNet implementation, provenance and limitations
- `docs/mdx23c-stage-0.md`: tightly scoped MDX23C checkpoint-load proof

Documentation is part of the implementation. Behaviour, configuration, model provenance and validation evidence must be updated in the same change as code.

## Repository contents

- `app/`: Python UDP server, processor lifecycle, runtime settings API and console
- `client/`: Rust command-line capture, transport and playback client
- `docs/architecture.md`: system architecture, boundaries and delivery model
- `docs/protocol.md`: KANY v1 UDP PCM datagram format
- `docs/convtasnet-lyrics-causal.md`: causal ConvTasNet integration and validation
- `docs/mdx23c-stage-0.md`: MDX23C Stage 0 scope and acceptance criteria
- `tests/`: Python processor and runtime-settings tests

## Current processors

| Processor | Purpose | GPU/model requirement |
|---|---|---|
| `passthrough` | Byte-preserving transport baseline | None |
| `delay-passthrough` | Artificial delay for queue/timing tests | None |
| `null` | Emits no output for failure/timeout tests | None |
| `stereo-centre-reduction` | Zero-lookahead mid/side vocal reduction | None |
| `htdemucs-vocals` | Buffered HTDemucs vocal reduction | GPU image recommended |
| `convtasnet-lyrics-causal` | Buffered finite-segment causal lyrics/accompaniment separation | GPU image recommended |
| `mdx23c-vocals` | Non-causal finite-window MDX23C vocal reduction | GPU image required; target validation pending |

MDX23C Stage 0 provenance is retained in `docs/mdx23c-stage-0.md`; offline inference, streaming, settings and pending target-host validation are documented in `docs/mdx23c.md`.

## Server deployment

On the Linux server:

```bash
git clone git@github.com:lewiskingy/karaoke-anything.git
cd karaoke-anything
docker compose up --build -d
```

Check the service:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/status
curl http://localhost:8080/processors
curl http://localhost:8080/metrics
```

Default ports:

- `5004/udp`: audio sent from the client to the server
- `5006/udp`: audio returned to the client
- `8080/tcp`: control and observability API and web console

By default, returned packets are sent to the source IP of each received packet. Set `RETURN_HOST` to force a fixed client address.

No GPU configuration is required for passthrough, delay, null or stereo-centre reduction.

## GPU model deployment

HTDemucs and the causal Cadenza ConvTasNet processor use the GPU-specific Compose override and `Dockerfile.demucs`:

```bash
docker compose \
  -f compose.yaml \
  -f compose.demucs.yaml \
  up -d --build
```

The filename `Dockerfile.demucs` is historical: the image now contains the shared GPU runtime and more than one model family. Do not rename it as part of an unrelated processor change.

The GPU image deliberately uses CUDA 12.8 with PyTorch and torchaudio 2.7.1 from the `cu128` wheel index. This is required for NVIDIA Blackwell GPUs such as the GeForce RTX 5070 Ti (`sm_120`). Do not downgrade this image to PyTorch 2.4/CUDA 12.4: those binaries do not contain kernels for `sm_120` and fail at runtime with `CUDA error: no kernel image is available for execution on the device`.

`requirements-demucs.txt` also pins NumPy explicitly because Demucs imports it during application startup. Do not add a conflicting torchaudio pin there: torch and torchaudio are installed together from the CUDA 12.8 wheel index by `Dockerfile.demucs`.

The image downloads `cadenzachallenge/ConvTasNet_Lyrics_Causal` during build and verifies that the matching vendored Clarity `ConvTasNetStereo` architecture can load it offline. For reproducible deployment, pin `CONVTASNET_MODEL_REVISION` to a Hugging Face commit instead of `main`.

The image also downloads only the pinned MDX23C 8KFFT YAML and checkpoint into
`/models/mdx23c`, copies the pinned upstream MDX23C architecture module, and
strictly loads the checkpoint on CPU during the build, as an offline
compatibility smoke test independent of the `mdx23c-vocals` processor's own
runtime model loading. Repeat the proof offline after building with:

```bash
docker compose -f compose.yaml -f compose.demucs.yaml run --rm --no-deps \
  karaoke-anything \
  python3 -m audio_trombone.tools.validate_mdx23c
```

Asset paths, upstream revisions, failure testing and remaining Stage 0 limits
are recorded in `docs/mdx23c-stage-0.md`.

After changing the GPU image, model revision or Python dependency versions, force a clean rebuild:

```bash
docker compose \
  -f compose.yaml \
  -f compose.demucs.yaml \
  down

docker compose \
  -f compose.yaml \
  -f compose.demucs.yaml \
  build --no-cache

docker compose \
  -f compose.yaml \
  -f compose.demucs.yaml \
  up -d
```

Verify the installed build and GPU support inside the running container:

```bash
docker compose \
  -f compose.yaml \
  -f compose.demucs.yaml \
  exec karaoke-anything python3 -c \
  "import numpy, torch, torchaudio, demucs; print('numpy', numpy.__version__); print('torch', torch.__version__); print('torchaudio', torchaudio.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0)); print('architectures', torch.cuda.get_arch_list())"
```

For an RTX 5070 Ti, the architecture list must include `sm_120`.

## Windows client setup

1. Install VB-CABLE and reboot Windows.
2. Route Spotify, OpenKJ, a browser or another source application to `CABLE Input`.
3. Install Rust using `rustup`.
4. List available devices:

```powershell
cd client
cargo run --release -- devices
```

5. Start the client, selecting `CABLE Output` for capture and a physical output device for playback:

```powershell
cargo run --release -- --server SERVER_LAN_IP:5004 --receive-port 5006 --capture "CABLE Output" --playback "Speakers"
```

Device names are matched using case-insensitive substrings and must identify exactly one device.

See `client/README.md` for current format constraints and diagnostics.

## Processor selection and runtime control

The startup processor is configured by Compose/environment:

```yaml
PROCESSOR: "passthrough"
```

Processors can also be selected at runtime:

```bash
curl -X PUT http://localhost:8080/processor/passthrough
curl -X PUT http://localhost:8080/processor/delay-passthrough
curl -X PUT http://localhost:8080/processor/null
curl -X PUT http://localhost:8080/processor/stereo-centre-reduction
curl -X PUT http://localhost:8080/processor/htdemucs-vocals
curl -X PUT http://localhost:8080/processor/convtasnet-lyrics-causal
curl -X PUT http://localhost:8080/processor/mdx23c-vocals
```

The web console is available at:

```text
http://SERVER_LAN_IP:8080/
```

`GET /api/settings` returns current and startup-default settings. `PATCH /api/settings` changes the running process only. `DELETE /api/settings` restores the startup defaults read from Compose/environment. Restarting the container always restores those startup defaults.

To start directly with ConvTasNet:

```bash
PROCESSOR=convtasnet-lyrics-causal \
CONVTASNET_SEGMENT_SECONDS=1.0 \
CONVTASNET_VOCAL_REDUCTION=1.0 \
docker compose -f compose.yaml -f compose.demucs.yaml up -d --build
```

Reset processor state after a seek, track change or reconnect:

```bash
curl -X POST http://localhost:8080/processor/reset
```

## Model-integration policy

New model families are introduced in stages. When checkpoint compatibility is uncertain, do not begin with an `AudioProcessor` implementation.

0. Prove pinned asset acquisition, architecture construction and strict offline checkpoint loading.
1. Prove offline inference shapes, sample rate and stem semantics against a short fixture.
2. Integrate bounded buffering, packet reconstruction, pacing and lifecycle.
3. Add runtime settings API and console controls.
4. Validate latency, RTF, VRAM, quality and discontinuity behaviour on the target host.

See `AGENTS.md` for the rules and `docs/mdx23c-stage-0.md` for the current MDX23C scope.

## Development

### Server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
pre-commit install
```

`pytest` enforces 100% statement coverage on `app/` (`--cov-fail-under=100` in `pyproject.toml`); a failing or coverage-reducing change fails the test run. `pre-commit install` wires the same check into a git pre-commit hook (config in `.pre-commit-config.yaml`) so it also runs before each commit.

### Client

```bash
cd client
cargo fmt --check
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov
cargo llvm-cov --ignore-filename-regex "main\.rs$" --fail-under-lines 100
cargo build --release
```

`protocol.rs` (KANY packet encode/decode) and `network.rs` (`sender_loop`/`receiver_loop`) are held at 100% line coverage; `cargo llvm-cov` fails if it drops below that. `main.rs` is excluded from the gate: `select_device`, `choose_*_config`, `build_*_stream`, `run()` and `main()` all operate on concrete `cpal::Device`/`cpal::Host` types with no mock backend, so they aren't practically unit-testable without real audio hardware. `sender_loop`/`receiver_loop` take `socket: &dyn AudioSocket` rather than a generic `<S: AudioSocket>` specifically so tests can inject a fake without creating a second, separately-covered monomorphization for the real `UdpSocket` call site in `run()`. The repository's pre-commit hook (see the Server section above) runs this same check.

## Security boundary

This version is intended for a trusted home LAN. UDP media and HTTP control endpoints are unauthenticated and unencrypted. Do not expose the service directly to the internet.
