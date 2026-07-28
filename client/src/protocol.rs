use anyhow::{bail, Result};

pub const MAGIC: &[u8; 4] = b"KANY";
pub const VERSION: u8 = 1;
pub const HEADER_SIZE: usize = 28;
pub const SAMPLE_FORMAT_F32_LE: u8 = 1;

#[derive(Debug)]
pub struct AudioPacket {
    pub channels: u16,
    pub sample_rate: u32,
    pub sequence: u32,
    pub timestamp_us: u64,
    pub frames: u16,
    pub samples: Vec<f32>,
}

pub fn encode_packet(
    channels: u16,
    sample_rate: u32,
    sequence: u32,
    timestamp_us: u64,
    frames: u16,
    samples: &[f32],
) -> Result<Vec<u8>> {
    if channels == 0 || channels > u8::MAX as u16 {
        bail!("channel count must be between 1 and 255");
    }
    if frames == 0 {
        bail!("frame count must be greater than zero");
    }

    let expected_samples = frames as usize * channels as usize;
    if samples.len() != expected_samples {
        bail!(
            "sample count {} does not match {} frames × {} channels",
            samples.len(),
            frames,
            channels
        );
    }

    let mut output = Vec::with_capacity(HEADER_SIZE + samples.len() * 4);
    output.extend_from_slice(MAGIC);
    output.push(VERSION);
    output.push(0);
    output.push(channels as u8);
    output.push(SAMPLE_FORMAT_F32_LE);
    output.extend_from_slice(&sample_rate.to_be_bytes());
    output.extend_from_slice(&sequence.to_be_bytes());
    output.extend_from_slice(&timestamp_us.to_be_bytes());
    output.extend_from_slice(&frames.to_be_bytes());
    output.extend_from_slice(&0u16.to_be_bytes());
    for sample in samples {
        output.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(output)
}

pub fn decode_packet(data: &[u8]) -> Result<AudioPacket> {
    if data.len() < HEADER_SIZE {
        bail!("packet is shorter than the {}-byte header", HEADER_SIZE);
    }
    if &data[0..4] != MAGIC {
        bail!("incorrect packet magic");
    }
    if data[4] != VERSION {
        bail!("unsupported protocol version {}", data[4]);
    }
    if data[7] != SAMPLE_FORMAT_F32_LE {
        bail!("unsupported sample format {}", data[7]);
    }

    let channels = data[6] as u16;
    let sample_rate = u32::from_be_bytes(data[8..12].try_into().unwrap());
    let sequence = u32::from_be_bytes(data[12..16].try_into().unwrap());
    let timestamp_us = u64::from_be_bytes(data[16..24].try_into().unwrap());
    let frames = u16::from_be_bytes(data[24..26].try_into().unwrap());

    if channels == 0 || frames == 0 {
        bail!("channels and frames must be non-zero");
    }

    let expected_payload = frames as usize * channels as usize * 4;
    if data.len() != HEADER_SIZE + expected_payload {
        bail!(
            "payload length mismatch: expected {}, received {}",
            expected_payload,
            data.len() - HEADER_SIZE
        );
    }

    let samples = data[HEADER_SIZE..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok(AudioPacket {
        channels,
        sample_rate,
        sequence,
        timestamp_us,
        frames,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_round_trip() {
        let samples = vec![0.0, 0.5, -0.5, 1.0];
        let encoded = encode_packet(2, 48_000, 42, 1234, 2, &samples).unwrap();
        let decoded = decode_packet(&encoded).unwrap();

        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.sequence, 42);
        assert_eq!(decoded.timestamp_us, 1234);
        assert_eq!(decoded.frames, 2);
        assert_eq!(decoded.samples, samples);
    }

    #[test]
    fn malformed_payload_is_rejected() {
        let mut encoded = encode_packet(2, 48_000, 1, 0, 1, &[0.0, 0.0]).unwrap();
        encoded.pop();
        assert!(decode_packet(&encoded).is_err());
    }

    #[test]
    fn encode_rejects_sample_count_mismatch() {
        let error = encode_packet(2, 48_000, 0, 0, 2, &[0.0, 0.0]).unwrap_err();
        assert!(error.to_string().contains("sample count"));
    }

    #[test]
    fn encode_rejects_zero_channels() {
        let error = encode_packet(0, 48_000, 0, 0, 1, &[]).unwrap_err();
        assert!(error.to_string().contains("channel count"));
    }

    #[test]
    fn encode_rejects_channels_above_255() {
        let samples = vec![0.0; 256];
        let error = encode_packet(256, 48_000, 0, 0, 1, &samples).unwrap_err();
        assert!(error.to_string().contains("channel count"));
    }

    #[test]
    fn encode_rejects_zero_frames() {
        let error = encode_packet(2, 48_000, 0, 0, 0, &[]).unwrap_err();
        assert!(error.to_string().contains("frame count"));
    }

    #[test]
    fn decode_rejects_payload_shorter_than_header() {
        let error = decode_packet(&[0u8; HEADER_SIZE - 1]).unwrap_err();
        assert!(error.to_string().contains("shorter than the"));
    }

    #[test]
    fn decode_rejects_incorrect_magic() {
        let mut encoded = encode_packet(2, 48_000, 1, 0, 1, &[0.0, 0.0]).unwrap();
        encoded[0] = b'X';
        let error = decode_packet(&encoded).unwrap_err();
        assert!(error.to_string().contains("incorrect packet magic"));
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let mut encoded = encode_packet(2, 48_000, 1, 0, 1, &[0.0, 0.0]).unwrap();
        encoded[4] = 2;
        let error = decode_packet(&encoded).unwrap_err();
        assert!(error.to_string().contains("unsupported protocol version"));
    }

    #[test]
    fn decode_rejects_unsupported_sample_format() {
        let mut encoded = encode_packet(2, 48_000, 1, 0, 1, &[0.0, 0.0]).unwrap();
        encoded[7] = 9;
        let error = decode_packet(&encoded).unwrap_err();
        assert!(error.to_string().contains("unsupported sample format"));
    }

    #[test]
    fn decode_rejects_zero_channels_or_frames() {
        let mut encoded = encode_packet(2, 48_000, 1, 0, 1, &[0.0, 0.0]).unwrap();
        encoded[6] = 0;
        let error = decode_packet(&encoded).unwrap_err();
        assert!(error.to_string().contains("must be non-zero"));
    }

    #[test]
    fn decode_rejects_payload_length_mismatch() {
        let mut encoded = encode_packet(2, 48_000, 1, 0, 1, &[0.0, 0.0]).unwrap();
        encoded.extend_from_slice(&[0u8; 4]);
        let error = decode_packet(&encoded).unwrap_err();
        assert!(error.to_string().contains("payload length mismatch"));
    }
}
