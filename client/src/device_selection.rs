//! Pure device/config decision logic, kept apart from device.rs's cpal I/O
//! so it can be unit tested without real audio hardware.
//!
//! `cpal::Device`'s fields are private and platform-specific, so it can't be
//! constructed in tests -- `select_device` is instead written against the
//! `NamedDevice` trait (mirroring `AudioSocket` in network.rs). The real
//! `impl NamedDevice for cpal::Device` lives in device.rs alongside the rest
//! of the un-testable-without-hardware cpal glue; only a fake implementation
//! is needed here.
//! `SupportedStreamConfigRange`, by contrast, has a public constructor, so
//! `config_matches`/`find_matching_config` are tested directly against the
//! real cpal type.

use anyhow::{anyhow, bail, Result};
use cpal::{SampleFormat, StreamConfig, SupportedStreamConfigRange};

/// The subset of device behaviour `select_device` depends on.
pub trait NamedDevice {
    fn display_name(&self) -> String;
}

pub fn select_device<D, I>(
    devices: I,
    needle: Option<&str>,
    default: Option<D>,
    kind: &str,
) -> Result<D>
where
    D: NamedDevice,
    I: Iterator<Item = D>,
{
    if let Some(needle) = needle {
        let needle = needle.to_lowercase();
        let mut matches = Vec::new();
        for device in devices {
            let name = device.display_name();
            if name.to_lowercase().contains(&needle) {
                matches.push((name, device));
            }
        }

        return match matches.len() {
            0 => bail!("no {kind} device contains '{needle}'"),
            1 => Ok(matches.remove(0).1),
            _ => {
                let names = matches
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>();
                bail!("{kind} device selector is ambiguous: {}", names.join(", "))
            }
        };
    }

    default.ok_or_else(|| anyhow!("no default {kind} device is available"))
}

pub fn config_matches(
    range: &SupportedStreamConfigRange,
    preferred_rate: u32,
    preferred_channels: u16,
) -> bool {
    let rate_in_range =
        range.min_sample_rate().0 <= preferred_rate && range.max_sample_rate().0 >= preferred_rate;
    range.sample_format() == SampleFormat::F32
        && range.channels() == preferred_channels
        && rate_in_range
}

pub fn find_matching_config(
    mut ranges: impl Iterator<Item = SupportedStreamConfigRange>,
    preferred_rate: u32,
    preferred_channels: u16,
) -> Option<StreamConfig> {
    ranges
        .find(|range| config_matches(range, preferred_rate, preferred_channels))
        .map(|range| {
            range
                .with_sample_rate(cpal::SampleRate(preferred_rate))
                .config()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::SupportedBufferSize;

    #[derive(Debug)]
    struct FakeDevice(&'static str);

    impl NamedDevice for FakeDevice {
        fn display_name(&self) -> String {
            self.0.to_string()
        }
    }

    fn fake_devices(names: &[&'static str]) -> impl Iterator<Item = FakeDevice> {
        names
            .iter()
            .map(|name| FakeDevice(name))
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn select_device_returns_default_when_no_needle_given() {
        let chosen = select_device(
            fake_devices(&["Mic", "Speakers"]),
            None,
            Some(FakeDevice("Default")),
            "input",
        )
        .unwrap();
        assert_eq!(chosen.0, "Default");
    }

    #[test]
    fn select_device_errors_when_no_needle_and_no_default() {
        let error =
            select_device(fake_devices(&["Mic"]), None, None::<FakeDevice>, "input").unwrap_err();
        assert!(error.to_string().contains("no default input device"));
    }

    #[test]
    fn select_device_matches_case_insensitive_substring() {
        let chosen = select_device(
            fake_devices(&["USB Microphone", "Speakers"]),
            Some("micro"),
            None,
            "input",
        )
        .unwrap();
        assert_eq!(chosen.0, "USB Microphone");
    }

    #[test]
    fn select_device_errors_when_no_device_matches_needle() {
        let error =
            select_device(fake_devices(&["Speakers"]), Some("micro"), None, "input").unwrap_err();
        assert!(error
            .to_string()
            .contains("no input device contains 'micro'"));
    }

    #[test]
    fn select_device_errors_when_needle_is_ambiguous() {
        let error = select_device(
            fake_devices(&["USB Microphone 1", "USB Microphone 2"]),
            Some("usb"),
            None,
            "input",
        )
        .unwrap_err();
        assert!(error.to_string().contains("ambiguous"));
    }

    fn range(
        channels: u16,
        min_rate: u32,
        max_rate: u32,
        format: SampleFormat,
    ) -> SupportedStreamConfigRange {
        SupportedStreamConfigRange::new(
            channels,
            cpal::SampleRate(min_rate),
            cpal::SampleRate(max_rate),
            SupportedBufferSize::Unknown,
            format,
        )
    }

    #[test]
    fn config_matches_true_for_f32_in_range_with_matching_channels() {
        let candidate = range(2, 44_100, 48_000, SampleFormat::F32);
        assert!(config_matches(&candidate, 48_000, 2));
    }

    #[test]
    fn config_matches_false_for_wrong_channel_count() {
        let candidate = range(2, 44_100, 48_000, SampleFormat::F32);
        assert!(!config_matches(&candidate, 48_000, 1));
    }

    #[test]
    fn config_matches_false_for_non_f32_format() {
        let candidate = range(2, 44_100, 48_000, SampleFormat::I16);
        assert!(!config_matches(&candidate, 48_000, 2));
    }

    #[test]
    fn config_matches_false_for_rate_outside_range() {
        let candidate = range(2, 44_100, 48_000, SampleFormat::F32);
        assert!(!config_matches(&candidate, 96_000, 2));
    }

    #[test]
    fn find_matching_config_returns_none_when_nothing_matches() {
        let ranges = vec![range(1, 44_100, 48_000, SampleFormat::F32)];
        assert!(find_matching_config(ranges.into_iter(), 48_000, 2).is_none());
    }

    #[test]
    fn find_matching_config_resolves_preferred_rate() {
        let ranges = vec![range(2, 44_100, 48_000, SampleFormat::F32)];
        let config = find_matching_config(ranges.into_iter(), 48_000, 2).unwrap();
        assert_eq!(config.sample_rate.0, 48_000);
        assert_eq!(config.channels, 2);
    }
}
