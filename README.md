# Karaoke Anything

A containerised streaming-audio system designed to turn arbitrary audio sources into karaoke.

The current implementation proves end-to-end audio transport before any source-separation model is introduced:

```text
Application audio
  -> virtual audio cable
  -> Rust capture client
  -> UDP PCM datagrams
  -> Python server
  -> selected AudioProcessor
  -> UDP PCM datagrams
  -> Rust playback client
  -> physical output device
```

With the server's default `passthrough` processor, each UDP datagram is returned byte-for-byte. There is no server-side decoding, resampling, codec conversion or GPU use.

## Repository contents

- `app/`: Python UDP server and pluggable processor lifecycle
- `client/`: Rust command-line capture, transport and playback client
- `docs/architecture.md`: system architecture, boundaries and delivery phases
- `docs/protocol.md`: version-one UDP PCM datagram format
- `tests/`: Python processor tests

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
- `8080/tcp`: control and observability API

By default, returned packets are sent to the IP address from which each packet arrived. Set `RETURN_HOST` in `compose.yaml` to force a fixed client address.

No GPU configuration is required for passthrough.

## Windows client setup

1. Install VB-CABLE and reboot Windows.
2. Route Spotify, OpenKJ, a browser or another source application to `CABLE Input`.
3. Install Rust using `rustup`.
4. List available devices:

```powershell
cd client
cargo run --release -- devices
```

5. Start the client, selecting `CABLE Output` for capture and your physical output device for playback:

```powershell
cargo run --release -- --server SERVER_LAN_IP:5004 --receive-port 5006 --capture "CABLE Output" --playback "Speakers"
```

Device names are matched using case-insensitive substrings and must identify exactly one device.

See `client/README.md` for current format constraints and diagnostics.

## Processor selection

The default is configured in Compose:

```yaml
PROCESSOR: "passthrough"
```

Processors can also be selected at runtime:

```bash
curl -X PUT http://localhost:8080/processor/passthrough
curl -X PUT http://localhost:8080/processor/delay-passthrough
curl -X PUT http://localhost:8080/processor/null
```

Reset processor state after a seek, track change or reconnect:

```bash
curl -X POST http://localhost:8080/processor/reset
```

## Development

### Server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

### Client

```bash
cd client
cargo fmt --check
cargo test
cargo build --release
```

## Model insertion point

`AudioProcessor` currently receives opaque packet payloads so passthrough remains byte-identical. Before integrating a separator, the server should introduce an explicit protocol and PCM-frame boundary:

```text
UDP datagram
  -> protocol validation and sequencing
  -> timestamped PCM frames
  -> AudioFrameProcessor
  -> packetisation
  -> UDP datagram
```

This keeps networking, audio framing and model inference separate. See `docs/architecture.md` for the detailed design.

## Security boundary

This version is intended for a trusted home LAN. UDP media and HTTP control endpoints are unauthenticated and unencrypted. Do not expose the service directly to the internet.
