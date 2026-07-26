# UDP PCM protocol

## Status

This is a deliberately small version-one transport for the passthrough milestone. It is not RTP and does not claim interoperability with other media systems.

The server currently forwards the complete datagram unchanged. Protocol parsing is performed by the Rust client. Server-side parsing and sequencing are planned before model integration.

## Datagram layout

All integer fields are network byte order (big endian).

```text
Offset  Size  Field
0       4     Magic: ASCII `KANY`
4       1     Protocol version: 1
5       1     Flags: 0
6       1     Channels
7       1     Sample format: 1 = interleaved IEEE-754 f32 little endian
8       4     Sample rate in Hz
12      4     Sequence number, wrapping u32
16      8     Capture timestamp in monotonic microseconds
24      2     Frames in this packet
26      2     Reserved: 0
28      ...   Interleaved PCM payload
```

A frame contains one sample for every channel. For stereo, each frame is left then right.

Payload length must equal:

```text
frames × channels × 4 bytes
```

## Initial fixed format

The client attempts to use:

- 48,000 Hz;
- two channels; and
- interleaved `f32` samples.

CPAL device negotiation may select a different sample rate or channel count when the requested configuration is unavailable. Each packet therefore carries its actual format. The receiver accepts only packets matching the active playback stream configuration and reports incompatible packets.

A later session-control protocol should negotiate one format explicitly and add resampling where needed.

## Packet size

The default client packet duration is 10 ms. At 48 kHz stereo this is:

```text
480 frames × 2 channels × 4 bytes = 3,840 payload bytes
```

Including the 28-byte header, the datagram is larger than a typical Ethernet MTU and may be fragmented by IP. For the initial trusted-LAN experiment this is accepted so that implementation remains simple.

The first refinement should use a shorter packet duration, compact PCM encoding, or an MTU-aware payload size. A 2.5 ms stereo `f32` packet at 48 kHz is 960 payload bytes.

## Sequence handling

The sender increments a wrapping 32-bit sequence number per datagram. The initial receiver records discontinuities but does not reorder packets. The playback buffer follows arrival order.

Future server and client transport layers should:

- reject duplicate packets;
- reorder within a small window;
- account for loss;
- reset processing state after a material discontinuity; and
- derive playout timing from timestamps rather than arrival order alone.

## Timestamp

The timestamp is microseconds from the sender's monotonic clock. It is useful only for relative timing within one process lifetime and is not wall-clock time.

## Validation

A receiver must reject a packet when:

- the magic is incorrect;
- the version is unsupported;
- the header is truncated;
- channels or frame count are zero;
- the sample format is unsupported; or
- the payload length does not match the declared format.

## Evolution

Protocol evolution should increment the version field. New optional fields should not be introduced by silently changing the version-one layout.

Likely later additions include:

- stream or session identifier;
- explicit discontinuity and end-of-stream flags;
- payload encoding identifiers;
- negotiated processor settings;
- authentication; and
- encrypted transport.
