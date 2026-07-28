mod network;
mod protocol;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use crossbeam_channel::{bounded, Sender};
use std::collections::VecDeque;
use std::net::{SocketAddr, UdpSocket};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use network::{receiver_loop, sender_loop, AudioFormat, ReceiverConfig, SenderConfig};

#[derive(Parser, Debug)]
#[command(name = "karaoke-anything-client")]
#[command(
    about = "Capture desktop audio, send it through Karaoke Anything, and play the returned stream"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Server address, for example 192.168.1.20:5004
    #[arg(long, default_value = "127.0.0.1:5004")]
    server: SocketAddr,

    /// Local UDP port used to receive returned audio
    #[arg(long, default_value_t = 5006)]
    receive_port: u16,

    /// Case-insensitive substring identifying the capture device
    #[arg(long)]
    capture: Option<String>,

    /// Case-insensitive substring identifying the playback device
    #[arg(long)]
    playback: Option<String>,

    /// Preferred sample rate
    #[arg(long, default_value_t = 48_000)]
    sample_rate: u32,

    /// Preferred channel count
    #[arg(long, default_value_t = 2)]
    channels: u16,

    /// Audio packet duration in milliseconds
    #[arg(long, default_value_t = 2.5)]
    packet_ms: f64,

    /// Maximum buffered playback duration in milliseconds
    #[arg(long, default_value_t = 250)]
    playback_buffer_ms: u32,

    /// Initial buffering before playback begins, in milliseconds
    #[arg(long, default_value_t = 30)]
    prebuffer_ms: u32,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List available input and output audio devices
    Devices,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let host = cpal::default_host();

    if matches!(cli.command, Some(Command::Devices)) {
        list_devices(&host)?;
        return Ok(());
    }

    run(host, cli)
}

fn run(host: cpal::Host, cli: Cli) -> Result<()> {
    if cli.packet_ms <= 0.0 {
        bail!("--packet-ms must be greater than zero");
    }

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

    let input_name = input_device.name().unwrap_or_else(|_| "<unknown>".into());
    let output_name = output_device.name().unwrap_or_else(|_| "<unknown>".into());

    let input_config = choose_input_config(&input_device, cli.sample_rate, cli.channels)?;
    let output_config = choose_output_config(
        &output_device,
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

    let sample_rate = input_config.sample_rate.0;
    let channels = input_config.channels;
    let frames_per_packet = ((sample_rate as f64 * cli.packet_ms / 1000.0).round() as usize)
        .clamp(1, u16::MAX as usize);
    let samples_per_packet = frames_per_packet * channels as usize;

    let max_playback_samples =
        (sample_rate as usize * channels as usize * cli.playback_buffer_ms as usize) / 1000;
    let prebuffer_samples =
        (sample_rate as usize * channels as usize * cli.prebuffer_ms as usize) / 1000;

    println!("Capture:  {input_name}");
    println!("Playback: {output_name}");
    println!("Format:   {sample_rate} Hz, {channels} channels, f32");
    println!(
        "Packet:   {:.2} ms ({} frames)",
        cli.packet_ms, frames_per_packet
    );
    println!("Server:   {}", cli.server);
    println!("Receive:  0.0.0.0:{}", cli.receive_port);

    let running = Arc::new(AtomicBool::new(true));
    let signal_running = Arc::clone(&running);
    ctrlc::set_handler(move || {
        signal_running.store(false, Ordering::SeqCst);
    })
    .context("failed to install Ctrl-C handler")?;

    let socket = UdpSocket::bind(("0.0.0.0", cli.receive_port))
        .with_context(|| format!("failed to bind UDP receive port {}", cli.receive_port))?;
    socket
        .set_read_timeout(Some(Duration::from_millis(200)))
        .context("failed to configure UDP receive timeout")?;
    let send_socket = socket.try_clone().context("failed to clone UDP socket")?;

    let (packet_tx, packet_rx) = bounded::<Vec<f32>>(128);
    let playback_queue = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(
        max_playback_samples.max(1),
    )));

    let format = AudioFormat {
        channels,
        sample_rate,
    };

    let sender_running = Arc::clone(&running);
    let sender_config = SenderConfig {
        server: cli.server,
        format,
        frames_per_packet: frames_per_packet as u16,
    };
    let sender =
        thread::spawn(move || sender_loop(&send_socket, packet_rx, sender_running, sender_config));

    let receiver_running = Arc::clone(&running);
    let receiver_queue = Arc::clone(&playback_queue);
    let receiver_config = ReceiverConfig {
        format,
        max_samples: max_playback_samples,
    };
    let receiver = thread::spawn(move || {
        receiver_loop(&socket, receiver_queue, receiver_running, receiver_config)
    });

    let capture_stream =
        build_input_stream(&input_device, &input_config, packet_tx, samples_per_packet)?;
    let playback_stream = build_output_stream(
        &output_device,
        &output_config,
        Arc::clone(&playback_queue),
        prebuffer_samples,
    )?;

    capture_stream
        .play()
        .context("failed to start capture stream")?;
    playback_stream
        .play()
        .context("failed to start playback stream")?;

    println!("Streaming. Press Ctrl-C to stop.");
    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(200));
    }

    drop(capture_stream);
    drop(playback_stream);

    sender
        .join()
        .map_err(|_| anyhow!("sender thread panicked"))??;
    receiver
        .join()
        .map_err(|_| anyhow!("receiver thread panicked"))??;

    Ok(())
}

fn list_devices(host: &cpal::Host) -> Result<()> {
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

fn select_device<I>(
    devices: I,
    needle: Option<&str>,
    default: Option<Device>,
    kind: &str,
) -> Result<Device>
where
    I: Iterator<Item = Device>,
{
    if let Some(needle) = needle {
        let needle = needle.to_lowercase();
        let mut matches = Vec::new();
        for device in devices {
            let name = device.name().unwrap_or_else(|_| "<unknown>".into());
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

fn choose_input_config(
    device: &Device,
    preferred_rate: u32,
    preferred_channels: u16,
) -> Result<StreamConfig> {
    for range in device
        .supported_input_configs()
        .context("failed to read input configurations")?
    {
        if range.sample_format() == SampleFormat::F32
            && range.channels() == preferred_channels
            && range.min_sample_rate().0 <= preferred_rate
            && range.max_sample_rate().0 >= preferred_rate
        {
            return Ok(range
                .with_sample_rate(cpal::SampleRate(preferred_rate))
                .config());
        }
    }

    let default = device
        .default_input_config()
        .context("no default input configuration")?;
    if default.sample_format() != SampleFormat::F32 {
        bail!("prototype currently requires an f32 input device configuration");
    }
    Ok(default.config())
}

fn choose_output_config(
    device: &Device,
    preferred_rate: u32,
    preferred_channels: u16,
) -> Result<StreamConfig> {
    for range in device
        .supported_output_configs()
        .context("failed to read output configurations")?
    {
        if range.sample_format() == SampleFormat::F32
            && range.channels() == preferred_channels
            && range.min_sample_rate().0 <= preferred_rate
            && range.max_sample_rate().0 >= preferred_rate
        {
            return Ok(range
                .with_sample_rate(cpal::SampleRate(preferred_rate))
                .config());
        }
    }

    let default = device
        .default_output_config()
        .context("no default output configuration")?;
    if default.sample_format() != SampleFormat::F32 {
        bail!("prototype currently requires an f32 output device configuration");
    }
    Ok(default.config())
}

fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    packet_tx: Sender<Vec<f32>>,
    samples_per_packet: usize,
) -> Result<cpal::Stream> {
    let mut pending = Vec::<f32>::with_capacity(samples_per_packet * 2);
    let err_fn = |error| eprintln!("capture stream error: {error}");

    device
        .build_input_stream(
            config,
            move |data: &[f32], _| {
                pending.extend_from_slice(data);
                while pending.len() >= samples_per_packet {
                    let tail = pending.split_off(samples_per_packet);
                    let packet = std::mem::replace(&mut pending, tail);
                    if packet_tx.try_send(packet).is_err() {
                        eprintln!("capture packet dropped because sender queue is full");
                    }
                }
            },
            err_fn,
            None,
        )
        .context("failed to create input stream")
}

fn build_output_stream(
    device: &Device,
    config: &StreamConfig,
    queue: Arc<Mutex<VecDeque<f32>>>,
    prebuffer_samples: usize,
) -> Result<cpal::Stream> {
    let err_fn = |error| eprintln!("playback stream error: {error}");
    let mut started = prebuffer_samples == 0;

    device
        .build_output_stream(
            config,
            move |output: &mut [f32], _| {
                let mut queue = match queue.lock() {
                    Ok(queue) => queue,
                    Err(_) => {
                        output.fill(0.0);
                        return;
                    }
                };

                if !started && queue.len() >= prebuffer_samples {
                    started = true;
                }

                if !started {
                    output.fill(0.0);
                    return;
                }

                for sample in output.iter_mut() {
                    *sample = queue.pop_front().unwrap_or(0.0);
                }
            },
            err_fn,
            None,
        )
        .context("failed to create output stream")
}
