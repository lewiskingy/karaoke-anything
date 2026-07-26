# HTDemucs streaming prototype

This processor is the first model-backed Karaoke Anything implementation. It separates the incoming stereo mix into vocals and accompaniment and can retain a configurable proportion of the estimated vocal stem.

It is deliberately experimental. The objective is continuous, audibly useful vocal reduction and measured real-time performance, not low latency.

## Processing shape

```text
KANY f32 PCM packets
  -> validate and assemble six seconds of stereo audio
  -> run HTDemucs on a background worker using CUDA
  -> subtract a configurable proportion of the vocals stem
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

## Start the model-backed server

Use the standard Compose file together with the GPU override:

```bash
docker compose -f compose.yaml -f compose.demucs.yaml up -d --build
```

The first start downloads the selected model into the named `demucs-model-cache` volume. Later starts reuse the cached model.

The override reads these values from `.env`, with defaults when they are absent:

```text
PROCESSOR=htdemucs-vocals
DEMUCS_MODEL=htdemucs
DEMUCS_DEVICE=auto
DEMUCS_SEGMENT_SECONDS=6.0
DEMUCS_OVERLAP=0.25
DEMUCS_SHIFTS=0
DEMUCS_VOCAL_REDUCTION=1.0
```

`DEMUCS_VOCAL_REDUCTION` ranges from `0.0` (original mix) to `1.0` (full estimated-vocal subtraction). `0.5` retains approximately half of the estimated vocal stem.

## Runtime control page

Open the HTTP service root in a browser, for example:

```text
http://SERVER_IP:8080/
```

The page can view and change the processor, model, segment length, overlap, shifts and vocal reduction. It also shows basic processor diagnostics.

Runtime changes are intentionally ephemeral:

- Compose and `.env` values are loaded as startup defaults each time the container starts.
- UI/API changes override those defaults only for the current process lifetime.
- **Restore startup defaults** returns to the values loaded when the process started.
- Changing vocal reduction alone is applied from the next completed segment without reloading the model.
- Structural changes, such as model, segment length or processor, reinitialise the processor and can cause a temporary audio gap.

The settings API is unauthenticated:

```bash
curl http://localhost:8080/api/settings

curl -X PATCH http://localhost:8080/api/settings \
  -H 'Content-Type: application/json' \
  -d '{"demucs":{"vocal_reduction":0.5}}'

curl -X DELETE http://localhost:8080/api/settings
```

Do not expose this HTTP port directly to the public internet.

## Observe startup and status

```bash
docker compose -f compose.yaml -f compose.demucs.yaml logs -f karaoke-anything
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

There will be silence during the initial segment and inference delay. After that, processed audio should begin and continue at a fixed delay behind the source.

## Known limitations

- Stereo KANY v1 `f32` input only.
- One active client and one stream format at a time.
- Multi-second fixed delay.
- No explicit crossfade between outer streaming segments yet.
- No adaptive jitter or model back-pressure policy.
- Runtime settings are not persisted across process/container restart.
- The settings UI and API have no authentication.
- A failed or slower-than-real-time inference can cause silence or eventual underruns.
