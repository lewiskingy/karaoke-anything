//! Audio device enumeration and stream-config negotiation.
//!
//! This is thin cpal I/O glue -- talking to real `Host`/`Device` handles --
//! and, like main.rs, is excluded from the coverage requirement for that
//! reason (see scripts/check-rust-coverage.sh). The decision logic it calls
//! into (`select_device`, `config_matches`, `find_matching_config`) is
//! separated out in device_selection.rs, where it's fully unit tested.

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, SampleFormat, StreamConfig, SupportedStreamConfigRange};

use crate::device_selection::{find_matching_config, select_device, NamedDevice};
use crate::Cli;

impl NamedDevice for Device {
    fn display_name(&self) -> String {
        self.name().unwrap_or_else(|_| "<unknown>".into())
    }
}

/// Devices and negotiated stream configs, matched in sample rate and channel count.
pub struct DeviceSetup {
    pub input_device: Device,
    pub output_device: Device,
    pub input_config: StreamConfig,
    pub output_config: StreamConfig,
}

pub fn negotiate_devices(host: &cpal::Host, cli: &Cli) -> Result<DeviceSetup> {
    let input_device = select_device(
        host.input_devices()
            .context("failed to enumerate input devices")?,
        cli.capture.as_deref(),
        host.default_input_device(),
        "input",
    )?;
    let output_device = select_device(
        host.output_devices()
            .context("failed to enumerate output devices")?,
        cli.playback.as_deref(),
        host.default_output_device(),
        "output",
    )?;

    let input_config = choose_config(
        &input_device,
        Direction::Input,
        cli.sample_rate,
        cli.channels,
    )?;
    let output_config = choose_config(
        &output_device,
        Direction::Output,
        input_config.sample_rate.0,
        input_config.channels,
    )?;

    if input_config.sample_rate != output_config.sample_rate
        || input_config.channels != output_config.channels
    {
        bail!(
            "capture and playback formats must match in this prototype: input={}Hz/{}ch, output={}Hz/{}ch",
            input_config.sample_rate.0,
            input_config.channels,
            output_config.sample_rate.0,
            output_config.channels
        );
    }

    Ok(DeviceSetup {
        input_device,
        output_device,
        input_config,
        output_config,
    })
}

pub fn list_devices(host: &cpal::Host) -> Result<()> {
    println!("Input devices:");
    for device in host
        .input_devices()
        .context("failed to enumerate input devices")?
    {
        println!("  {}", device.name().unwrap_or_else(|_| "<unknown>".into()));
    }

    println!("\nOutput devices:");
    for device in host
        .output_devices()
        .context("failed to enumerate output devices")?
    {
        println!("  {}", device.name().unwrap_or_else(|_| "<unknown>".into()));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Direction {
    Input,
    Output,
}

impl Direction {
    fn label(self) -> &'static str {
        match self {
            Direction::Input => "input",
            Direction::Output => "output",
        }
    }

    /// cpal exposes separate, differently-typed iterators for input vs.
    /// output configs, so this still boxes to unify them into one return type.
    fn supported_configs(
        self,
        device: &Device,
    ) -> Result<
        Box<dyn Iterator<Item = SupportedStreamConfigRange>>,
        cpal::SupportedStreamConfigsError,
    > {
        match self {
            Direction::Input => Ok(Box::new(device.supported_input_configs()?)),
            Direction::Output => Ok(Box::new(device.supported_output_configs()?)),
        }
    }

    fn default_config(
        self,
        device: &Device,
    ) -> Result<cpal::SupportedStreamConfig, cpal::DefaultStreamConfigError> {
        match self {
            Direction::Input => device.default_input_config(),
            Direction::Output => device.default_output_config(),
        }
    }
}

/// Picks a device's stream config, preferring an exact match for
/// `preferred_rate`/`preferred_channels` and falling back to the device
/// default (if it's f32) otherwise.
fn choose_config(
    device: &Device,
    direction: Direction,
    preferred_rate: u32,
    preferred_channels: u16,
) -> Result<StreamConfig> {
    let kind = direction.label();
    let ranges = direction
        .supported_configs(device)
        .with_context(|| format!("failed to read {kind} configurations"))?;
    if let Some(config) = find_matching_config(ranges, preferred_rate, preferred_channels) {
        return Ok(config);
    }

    let default = direction
        .default_config(device)
        .with_context(|| format!("no default {kind} configuration"))?;
    if default.sample_format() != SampleFormat::F32 {
        bail!("prototype currently requires an f32 {kind} device configuration");
    }
    Ok(default.config())
}
