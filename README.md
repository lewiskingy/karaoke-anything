# Karaoke Anything

A containerised streaming-audio service designed to turn arbitrary audio sources into karaoke.

The initial implementation establishes a reliable network passthrough and a pluggable processor lifecycle before any source-separation model is introduced.

```text
UDP ingress
  -> bounded queue
  -> selected MediaProcessor
  -> UDP egress
```

With the default `passthrough` processor, every UDP payload is returned byte-for-byte. There is no decoding, resampling, codec conversion or GPU use.

## Current scope

- Single-client UDP transport on a trusted home LAN
- Configurable return host and output port
- Bounded ingress queue
- Pluggable asynchronous processor interface
- Passthrough, delayed-passthrough and null development processors
- Runtime processor selection and reset
- Health, status and Prometheus-style metrics endpoints
- Container and Docker Compose deployment
- Processor unit tests and GitHub Actions CI

## Run

```bash
docker compose up --build -d
```

Check the service:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/status
curl http://localhost:8080/processors
curl http://localhost:8080/metrics
```

The default ports are:

- `5004/udp`: media sent from the client to the server
- `5006/udp`: media returned by the server to the client
- `8080/tcp`: HTTP control and observability API

By default, returned packets are sent to the IP address from which each packet arrived. Set `RETURN_HOST` to force a fixed client address.

## Processor selection

The production-safe default is configured in Compose:

```yaml
PROCESSOR: "passthrough"
```

Processors can also be selected at runtime:

```bash
curl -X PUT http://localhost:8080/processor/passthrough
curl -X PUT http://localhost:8080/processor/delay-passthrough
curl -X PUT http://localhost:8080/processor/null
```

Reset state after a seek, track change or reconnect:

```bash
curl -X POST http://localhost:8080/processor/reset
```

## Transport test

On the client machine, start the receiver:

```bash
python test_udp.py receive --port 5006
```

Then send packets through the server:

```bash
python test_udp.py send --server SERVER_LAN_IP --port 5004
```

With `passthrough`, received payloads must match the transmitted payloads exactly.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

## Model insertion point

A processor can emit zero, one or many output packets per input packet:

```python
async def process(
    self,
    packet: MediaPacket,
) -> AsyncIterator[ProcessedPacket]:
    ...
```

This provides lifecycle and buffering semantics for causal and windowed processors. The next significant iteration should introduce the explicit audio boundary needed by a real source separator:

```text
RTP/UDP packet
  -> RTP parsing and ordering
  -> timestamped PCM frames
  -> AudioFrameProcessor
  -> PCM packetisation
  -> RTP/UDP output
```

The current `MediaProcessor` therefore provides the service extension structure while intentionally treating network payloads as opaque bytes.

## Security boundary

This version is intended for a trusted home LAN. It is single-session and unauthenticated. Do not expose its ports through an internet router.
