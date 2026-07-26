# HTDemucs streaming prototype

This processor is the first model-backed Karaoke Anything implementation. It separates the incoming stereo mix into vocals and accompaniment with HTDemucs and returns the accompaniment.

It is deliberately experimental. The objective is continuous, audibly useful vocal reduction and measured real-time performance, not low latency.

## Processing shape

```text
KANY f32 PCM packets
  -> validate and assemble six seconds of stereo audio
  -> run HTDemucs on a background worker using CUDA
  -> subtract the vocals stem from the resampled mixture
  -> resample back to the client rate
  -> rebuild the original KANY packets
  -> release one processed packet per incoming packet
```

Releasing one output packet per new input packet is important. It uses the live input cadence to pace playback and avoids overflowing the client's bounded receive buffer with a multi-second burst.

## Expected latency

The first output arrives after one complete segment plus inference time. With the default configuration this is approximately:

```text
6 seconds of input buffering + one HTDemucs inference
```

Once running, output remains continuous only when the real-time factor is below 1.0. For example, a six-second segment must finish in less than six seconds.

This prototype processes adjacent outer segments independently. Demucs still uses its configured internal overlap, but there is not yet an additional crossfade between successive six-second streaming segments. Audible boundary artefacts are therefore possible.

## GPU prerequisites

The host must have:

- a supported NVIDIA GPU and driver;
- the NVIDIA Container Toolkit;
- Docker configured so `docker run --gpus all ...` works.

Confirm GPU access before building the service:

```bash
nvidia-smi
docker run --rm --gpus all pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime nvidia-smi
```

## Start the model-backed server

Use the standard Compose file together with the GPU override:

```bash
docker compose -f compose.yaml -f compose.demucs.yaml up -d --build
```

The first start downloads the selected model into the named `demucs-model-cache` volume. Startup can therefore take several minutes. Later starts reuse the cached model.

The override selects:

```text
PROCESSOR=htdemucs-vocals
DEMUCS_MODEL=htdemucs
DEMUCS_DEVICE=auto
DEMUCS_SEGMENT_SECONDS=6.0
DEMUCS_OVERLAP=0.25
DEMUCS_SHIFTS=0
```

`DEMUCS_DEVICE=auto` selects CUDA when PyTorch can see it and otherwise falls back to CPU. CPU mode is useful for diagnosis but is unlikely to sustain real-time HTDemucs processing.

If the normal HTTP host port is already in use, retain the local port override already used for the passthrough deployment, for example `8180:8080`.

## Observe startup

```bash
docker compose -f compose.yaml -f compose.demucs.yaml logs -f karaoke-anything
```

Expected startup messages include the selected device, model download/loading and the model sample rate.

Check the service after model loading completes:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/status
curl http://localhost:8080/processors
```

The active processor should be `htdemucs-vocals`.

## Run the existing Windows client

No client change is required. Continue using the working 48 kHz stereo command:

```powershell
.\target\release\karaoke-anything-client.exe --server SERVER_IP:5004 --receive-port 5006 --capture "CABLE Output" --playback "Speakers"
```

Leave Spotify or the karaoke application routed to `CABLE Input`.

There will be silence during the initial segment and inference delay. After that, accompaniment should begin and continue at a fixed delay behind the source.

## Switching processors

The GPU image still contains the lightweight processors. Runtime switching is available through the management API:

```bash
curl -X PUT http://localhost:8080/processor/passthrough
curl -X PUT http://localhost:8080/processor/htdemucs-vocals
curl -X POST http://localhost:8080/processor/reset
```

Selecting HTDemucs loads the model before the request completes. Reset after a seek, track change or audio format change.

## Initial tuning

Start with the defaults. Once the first song works, capture:

- time until first sound;
- inference time per segment from logs;
- real-time factor;
- GPU utilisation and memory;
- whether playback remains continuous;
- vocal leakage;
- artefacts at six-second boundaries.

If processing cannot keep up, try reducing overlap before reducing segment duration:

```yaml
DEMUCS_OVERLAP: "0.10"
```

Shorter segments reduce buffering latency but can reduce quality and make segment boundaries more obvious. HTDemucs supports a maximum segment length of about 7.8 seconds.

## Known limitations

- Stereo KANY v1 `f32` input only.
- One active client and one stream format at a time.
- Multi-second fixed delay.
- No explicit crossfade between outer streaming segments yet.
- No adaptive jitter or model back-pressure policy.
- No vocal-level control: output is currently mixture minus estimated vocals.
- A failed or slower-than-real-time inference can cause silence or eventual underruns.
- The model and CUDA image are much larger than the passthrough image.
