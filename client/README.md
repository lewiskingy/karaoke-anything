# Rust client

The client captures PCM audio from an input device, sends it to the Karaoke Anything server, receives the returned stream, and plays it through an output device.

## Windows prerequisites

1. Install Rust using `rustup`.
2. Install VB-CABLE and reboot Windows.
3. Route the source application to `CABLE Input` in Windows App volume and device preferences.
4. Use the client to capture from `CABLE Output` and play to your physical speakers or mixer.

## Build

```powershell
cd client
cargo build --release
```

The executable is created at:

```text
target\release\karaoke-anything-client.exe
```

## List devices

```powershell
cargo run --release -- devices
```

Device selectors are case-insensitive substrings. They must match exactly one device.

## Run

```powershell
cargo run --release -- \
  --server 192.168.1.20:5004 \
  --receive-port 5006 \
  --capture "CABLE Output" \
  --playback "Speakers"
```

On PowerShell, use backticks rather than backslashes for multiline commands, or put the command on one line:

```powershell
cargo run --release -- --server 192.168.1.20:5004 --receive-port 5006 --capture "CABLE Output" --playback "Speakers"
```

The client defaults to 48 kHz stereo `f32`, 2.5 ms packets, a 30 ms initial prebuffer and a 250 ms bounded receive buffer.

## Current constraints

- The capture and playback devices must expose matching sample rates and channel counts.
- The prototype currently requires an `f32` CPAL stream format.
- There is no resampling or channel remapping.
- Sequence discontinuities are reported but packets are not reordered.
- UDP media is unauthenticated and unencrypted.
- The server must remain on the `passthrough` processor for a byte-identical end-to-end test.

These constraints are intentional for the first transport milestone and are described in `../docs/architecture.md`.
