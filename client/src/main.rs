mod network;
mod protocol;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig, SupportedStreamConfigRange};
use crossbeam_channel::{bounded, Receiver, Sender};
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

/// Devices and negotiated stream configs, matched in sample rate and channel count.
struct DeviceSetup {
    input_device: Device,
    output_device: Device,
    input_config: StreamConfig,
    output_config: StreamConfig,
}

fn negotiate_devices(host: &cpal::Host, cli: &Cli) -> Result<DeviceSetup> {
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

    let input_config =
        choose_config::<InputDirection>(&input_device, cli.sample_rate, cli.channels)?;
    let output_config = choose_config::<OutputDirection>(
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

    Ok(DeviceSetup {
        input_device,
        output_device,
        input_config,
        output_config,
    })
}

/// Frame/sample counts derived from the negotiated format and CLI durations.
struct PacketSizing {
    sample_rate: u32,
    channels: u16,
    frames_per_packet: usize,
    samples_per_packet: usize,
    max_playback_samples: usize,
    prebuffer_samples: usize,
}

fn compute_packet_sizing(config: &StreamConfig, cli: &Cli) -> PacketSizing {
    let sample_rate = config.sample_rate.0;
    let channels = config.channels;
    let frames_per_packet = ((sample_rate as f64 * cli.packet_ms / 1000.0).round() as usize)
        .clamp(1, u16::MAX as usize);

    PacketSizing {
        sample_rate,
        channels,
        frames_per_packet,
        samples_per_packet: frames_per_packet * channels as usize,
        max_playback_samples: (sample_rate as usize
            * channels as usize
            * cli.playback_buffer_ms as usize)
            / 1000,
        prebuffer_samples: (sample_rate as usize * channels as usize * cli.prebuffer_ms as usize)
            / 1000,
    }
}

fn print_session_info(input_name: &str, output_name: &str, sizing: &PacketSizing, cli: &Cli) {
    println!("Capture:  {input_name}");
    println!("Playback: {output_name}");
    println!(
        "Format:   {} Hz, {} channels, f32",
        sizing.sample_rate, sizing.channels
    );
    println!(
        "Packet:   {:.2} ms ({} frames)",
        cli.packet_ms, sizing.frames_per_packet
    );
    println!("Server:   {}", cli.server);
    println!("Receive:  0.0.0.0:{}", cli.receive_port);
}

fn install_running_flag() -> Result<Arc<AtomicBool>> {
    let running = Arc::new(AtomicBool::new(true));
    let signal_running = Arc::clone(&running);
    ctrlc::set_handler(move || {
        signal_running.store(false, Ordering::SeqCst);
    })
    .context("failed to install Ctrl-C handler")?;
    Ok(running)
}

fn bind_udp_sockets(receive_port: u16) -> Result<(UdpSocket, UdpSocket)> {
    let socket = UdpSocket::bind(("0.0.0.0", receive_port))
        .with_context(|| format!("failed to bind UDP receive port {receive_port}"))?;
    socket
        .set_read_timeout(Some(Duration::from_millis(200)))
        .context("failed to configure UDP receive timeout")?;
    let send_socket = socket.try_clone().context("failed to clone UDP socket")?;
    Ok((socket, send_socket))
}

/// Background threads moving audio to and from the network; joined once streaming stops.
struct NetworkThreads {
    sender: thread::JoinHandle<Result<()>>,
    receiver: thread::JoinHandle<Result<()>>,
}

/// The UDP sockets and channels a network thread pair needs, grouped so
/// `spawn_network_threads` can take one argument instead of four.
struct NetworkIo {
    socket: UdpSocket,
    send_socket: UdpSocket,
    packet_rx: Receiver<Vec<f32>>,
    playback_queue: Arc<Mutex<VecDeque<f32>>>,
}

/// Session parameters a network thread pair needs, grouped for the same reason.
struct NetworkParams {
    server: SocketAddr,
    format: AudioFormat,
    frames_per_packet: u16,
    max_playback_samples: usize,
}

fn spawn_network_threads(
    io: NetworkIo,
    running: &Arc<AtomicBool>,
    params: NetworkParams,
) -> NetworkThreads {
    let sender_running = Arc::clone(running);
    let sender_config = SenderConfig {
        server: params.server,
        format: params.format,
        frames_per_packet: params.frames_per_packet,
    };
    let send_socket = io.send_socket;
    let packet_rx = io.packet_rx;
    let sender =
        thread::spawn(move || sender_loop(&send_socket, packet_rx, sender_running, sender_config));

    let receiver_running = Arc::clone(running);
    let receiver_config = ReceiverConfig {
        format: params.format,
        max_samples: params.max_playback_samples,
    };
    let socket = io.socket;
    let playback_queue = io.playback_queue;
    let receiver = thread::spawn(move || {
        receiver_loop(&socket, playback_queue, receiver_running, receiver_config)
    });

    NetworkThreads { sender, receiver }
}

fn start_streams(
    devices: &DeviceSetup,
    packet_tx: Sender<Vec<f32>>,
    playback_queue: Arc<Mutex<VecDeque<f32>>>,
    sizing: &PacketSizing,
) -> Result<(cpal::Stream, cpal::Stream)> {
    let capture_stream = build_input_stream(
        &devices.input_device,
        &devices.input_config,
        packet_tx,
        sizing.samples_per_packet,
    )?;
    let playback_stream = build_output_stream(
        &devices.output_device,
        &devices.output_config,
        playback_queue,
        sizing.prebuffer_samples,
    )?;

    capture_stream
        .play()
        .context("failed to start capture stream")?;
    playback_stream
        .play()
        .context("failed to start playback stream")?;

    Ok((capture_stream, playback_stream))
}

fn stream_until_stopped(
    running: &Arc<AtomicBool>,
    capture_stream: cpal::Stream,
    playback_stream: cpal::Stream,
    threads: NetworkThreads,
) -> Result<()> {
    println!("Streaming. Press Ctrl-C to stop.");
    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(200));
    }

    drop(capture_stream);
    drop(playback_stream);

    threads
        .sender
        .join()
        .map_err(|_| anyhow!("sender thread panicked"))??;
    threads
        .receiver
        .join()
        .map_err(|_| anyhow!("receiver thread panicked"))??;

    Ok(())
}

fn run(host: cpal::Host, cli: Cli) -> Result<()> {
    if cli.packet_ms <= 0.0 {
        bail!("--packet-ms must be greater than zero");
    }

    let devices = negotiate_devices(&host, &cli)?;
    let input_name = devices
        .input_device
        .name()
        .unwrap_or_else(|_| "<unknown>".into());
    let output_name = devices
        .output_device
        .name()
        .unwrap_or_else(|_| "<unknown>".into());

    let sizing = compute_packet_sizing(&devices.input_config, &cli);
    print_session_info(&input_name, &output_name, &sizing, &cli);

    let running = install_running_flag()?;
    let (socket, send_socket) = bind_udp_sockets(cli.receive_port)?;

    let (packet_tx, packet_rx) = bounded::<Vec<f32>>(128);
    let playback_queue = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(
        sizing.max_playback_samples.max(1),
    )));

    let format = AudioFormat {
        channels: sizing.channels,
        sample_rate: sizing.sample_rate,
    };

    let io = NetworkIo {
        socket,
        send_socket,
        packet_rx,
        playback_queue: Arc::clone(&playback_queue),
    };
    let params = NetworkParams {
        server: cli.server,
        format,
        frames_per_packet: sizing.frames_per_packet as u16,
        max_playback_samples: sizing.max_playback_samples,
    };
    let threads = spawn_network_threads(io, &running, params);

    let (capture_stream, playback_stream) =
        start_streams(&devices, packet_tx, playback_queue, &sizing)?;

    stream_until_stopped(&running, capture_stream, playback_stream, threads)
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

fn config_matches(
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

fn find_matching_config(
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

/// Bridges cpal's asymmetric `Device` API (separate input/output methods,
/// no shared trait) so `choose_config` can pick between them via the type
/// system instead of a runtime direction flag. `supported_configs` still
/// boxes its iterator — cpal's input/output config iterators are different
/// concrete types — but which device methods get called is resolved at
/// compile time via monomorphization, not a branch inside `choose_config`.
trait DeviceDirection {
    const LABEL: &'static str;

    fn supported_configs(
        device: &Device,
    ) -> Result<
        Box<dyn Iterator<Item = SupportedStreamConfigRange>>,
        cpal::SupportedStreamConfigsError,
    >;

    fn default_config(
        device: &Device,
    ) -> Result<cpal::SupportedStreamConfig, cpal::DefaultStreamConfigError>;
}

struct InputDirection;

impl DeviceDirection for InputDirection {
    const LABEL: &'static str = "input";

    fn supported_configs(
        device: &Device,
    ) -> Result<
        Box<dyn Iterator<Item = SupportedStreamConfigRange>>,
        cpal::SupportedStreamConfigsError,
    > {
        Ok(Box::new(device.supported_input_configs()?))
    }

    fn default_config(
        device: &Device,
    ) -> Result<cpal::SupportedStreamConfig, cpal::DefaultStreamConfigError> {
        device.default_input_config()
    }
}

struct OutputDirection;

impl DeviceDirection for OutputDirection {
    const LABEL: &'static str = "output";

    fn supported_configs(
        device: &Device,
    ) -> Result<
        Box<dyn Iterator<Item = SupportedStreamConfigRange>>,
        cpal::SupportedStreamConfigsError,
    > {
        Ok(Box::new(device.supported_output_configs()?))
    }

    fn default_config(
        device: &Device,
    ) -> Result<cpal::SupportedStreamConfig, cpal::DefaultStreamConfigError> {
        device.default_output_config()
    }
}

/// Picks a device's stream config, preferring an exact match for
/// `preferred_rate`/`preferred_channels` and falling back to the device
/// default (if it's f32) otherwise.
fn choose_config<D: DeviceDirection>(
    device: &Device,
    preferred_rate: u32,
    preferred_channels: u16,
) -> Result<StreamConfig> {
    let kind = D::LABEL;
    let ranges = D::supported_configs(device)
        .with_context(|| format!("failed to read {kind} configurations"))?;
    if let Some(config) = find_matching_config(ranges, preferred_rate, preferred_channels) {
        return Ok(config);
    }

    let default =
        D::default_config(device).with_context(|| format!("no default {kind} configuration"))?;
    if default.sample_format() != SampleFormat::F32 {
        bail!("prototype currently requires an f32 {kind} device configuration");
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
