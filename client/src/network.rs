use anyhow::{anyhow, Context, Result};
use crossbeam_channel::Receiver;
use std::collections::VecDeque;
use std::net::{SocketAddr, UdpSocket};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use crate::protocol::{decode_packet, encode_packet, AudioPacket};

/// The subset of UDP socket behaviour `sender_loop`/`receiver_loop` depend on,
/// extracted so tests can exercise every branch without a real network stack.
pub trait AudioSocket {
    fn send_datagram(&self, data: &[u8], addr: SocketAddr) -> std::io::Result<usize>;
    fn recv_datagram(&self, buffer: &mut [u8]) -> std::io::Result<(usize, SocketAddr)>;
}

impl AudioSocket for UdpSocket {
    fn send_datagram(&self, data: &[u8], addr: SocketAddr) -> std::io::Result<usize> {
        self.send_to(data, addr)
    }

    fn recv_datagram(&self, buffer: &mut [u8]) -> std::io::Result<(usize, SocketAddr)> {
        self.recv_from(buffer)
    }
}

/// A stream's channel count and sample rate, shared by both directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioFormat {
    pub channels: u16,
    pub sample_rate: u32,
}

#[derive(Clone, Copy)]
pub struct SenderConfig {
    pub server: SocketAddr,
    pub format: AudioFormat,
    pub frames_per_packet: u16,
}

#[derive(Clone, Copy)]
pub struct ReceiverConfig {
    pub format: AudioFormat,
    pub max_samples: usize,
}

pub fn sender_loop(
    socket: &dyn AudioSocket,
    packets: Receiver<Vec<f32>>,
    running: Arc<AtomicBool>,
    config: SenderConfig,
) -> Result<()> {
    let started = Instant::now();
    let mut sequence = 0u32;

    while running.load(Ordering::SeqCst) {
        match packets.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                let packet = AudioPacket {
                    channels: config.format.channels,
                    sample_rate: config.format.sample_rate,
                    sequence,
                    timestamp_us: started.elapsed().as_micros() as u64,
                    frames: config.frames_per_packet,
                    samples,
                };
                let datagram = encode_packet(&packet)?;
                socket
                    .send_datagram(&datagram, config.server)
                    .with_context(|| format!("failed to send audio to {}", config.server))?;
                sequence = sequence.wrapping_add(1);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

pub fn receiver_loop(
    socket: &dyn AudioSocket,
    queue: Arc<Mutex<VecDeque<f32>>>,
    running: Arc<AtomicBool>,
    config: ReceiverConfig,
) -> Result<()> {
    let mut buffer = vec![0u8; 65_535];
    let mut expected_sequence: Option<u32> = None;

    while running.load(Ordering::SeqCst) {
        match socket.recv_datagram(&mut buffer) {
            Ok((length, _)) => {
                handle_datagram(&buffer[..length], config, &mut expected_sequence, &queue)?
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock
                    || error.kind() == std::io::ErrorKind::TimedOut => {}
            Err(error) => return Err(error).context("failed receiving audio"),
        }
    }
    Ok(())
}

/// Decodes one datagram and, if it matches `config.format`, buffers its
/// samples. Malformed or incompatible datagrams are logged and otherwise
/// ignored - only a poisoned queue lock is treated as fatal for the receive
/// loop.
fn handle_datagram(
    data: &[u8],
    config: ReceiverConfig,
    expected_sequence: &mut Option<u32>,
    queue: &Mutex<VecDeque<f32>>,
) -> Result<()> {
    let packet = match decode_packet(data) {
        Ok(packet) => packet,
        Err(error) => {
            eprintln!("ignored invalid packet: {error:#}");
            return Ok(());
        }
    };

    if packet.channels != config.format.channels || packet.sample_rate != config.format.sample_rate
    {
        eprintln!(
            "ignored incompatible packet: {}Hz/{}ch",
            packet.sample_rate, packet.channels
        );
        return Ok(());
    }

    if let Some(expected) = *expected_sequence {
        if packet.sequence != expected {
            eprintln!(
                "sequence discontinuity: expected {}, received {}",
                expected, packet.sequence
            );
        }
    }
    *expected_sequence = Some(packet.sequence.wrapping_add(1));

    buffer_samples(queue, packet.samples, config.max_samples)
}

/// Appends `samples` to `queue`, evicting the oldest buffered samples (and,
/// if `samples` alone exceeds `max_samples`, its own oldest samples too) so
/// the queue never grows past `max_samples`.
fn buffer_samples(
    queue: &Mutex<VecDeque<f32>>,
    samples: Vec<f32>,
    max_samples: usize,
) -> Result<()> {
    let mut queue = queue
        .lock()
        .map_err(|_| anyhow!("playback queue poisoned"))?;
    let keep = samples.len().min(max_samples);
    let dropped = samples.len() - keep;
    while queue.len() + keep > max_samples && !queue.is_empty() {
        queue.pop_front();
    }
    queue.extend(samples.into_iter().skip(dropped));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::encode_packet;
    use std::io::{Error, ErrorKind};

    // `AudioSocket` is object-safe and `sender_loop`/`receiver_loop` take
    // `&dyn AudioSocket`, so every test below shares one non-generic compiled
    // body regardless of which concrete socket type it passes in. That keeps
    // coverage measurement meaningful: a generic `<S: AudioSocket>` parameter
    // would instead monomorphize a separate copy per call-site type (this
    // `FakeSocket`, and the real `UdpSocket` used below), each tracked as its
    // own instantiation and each needing independent full branch coverage.
    struct FakeSocket {
        recv_script: Mutex<VecDeque<std::io::Result<Vec<u8>>>>,
        // `send_only` sockets never call `recv_datagram`, so this is never
        // observed there; it exists so `recv_datagram` always has a flag to
        // stop the loop with once `recv_script` runs dry.
        running: Arc<AtomicBool>,
        sent: Mutex<Vec<(Vec<u8>, SocketAddr)>>,
        send_error: Option<ErrorKind>,
    }

    impl FakeSocket {
        fn recv_script(script: Vec<std::io::Result<Vec<u8>>>, running: Arc<AtomicBool>) -> Self {
            Self {
                recv_script: Mutex::new(script.into_iter().collect()),
                running,
                sent: Mutex::new(Vec::new()),
                send_error: None,
            }
        }

        fn send_only(send_error: Option<ErrorKind>) -> Self {
            Self {
                recv_script: Mutex::new(VecDeque::new()),
                running: Arc::new(AtomicBool::new(true)),
                sent: Mutex::new(Vec::new()),
                send_error,
            }
        }
    }

    impl AudioSocket for FakeSocket {
        fn send_datagram(&self, data: &[u8], addr: SocketAddr) -> std::io::Result<usize> {
            if let Some(kind) = self.send_error {
                return Err(Error::new(kind, "simulated send failure"));
            }
            self.sent.lock().unwrap().push((data.to_vec(), addr));
            Ok(data.len())
        }

        fn recv_datagram(&self, buffer: &mut [u8]) -> std::io::Result<(usize, SocketAddr)> {
            let next = self.recv_script.lock().unwrap().pop_front();
            match next {
                Some(Ok(bytes)) => {
                    buffer[..bytes.len()].copy_from_slice(&bytes);
                    Ok((bytes.len(), "127.0.0.1:0".parse().unwrap()))
                }
                Some(Err(error)) => Err(error),
                None => {
                    self.running.store(false, Ordering::SeqCst);
                    Err(Error::new(
                        ErrorKind::WouldBlock,
                        "scripted responses exhausted",
                    ))
                }
            }
        }
    }

    fn addr() -> SocketAddr {
        "127.0.0.1:9".parse().expect("valid socket address literal")
    }

    fn format() -> AudioFormat {
        AudioFormat {
            channels: 2,
            sample_rate: 48_000,
        }
    }

    fn sender_config() -> SenderConfig {
        SenderConfig {
            server: addr(),
            format: format(),
            frames_per_packet: 1,
        }
    }

    fn receiver_config(max_samples: usize) -> ReceiverConfig {
        ReceiverConfig {
            format: format(),
            max_samples,
        }
    }

    fn valid_packet(sequence: u32) -> Vec<u8> {
        encode_packet(&AudioPacket {
            channels: format().channels,
            sample_rate: format().sample_rate,
            sequence,
            timestamp_us: 0,
            frames: 1,
            samples: vec![0.1, 0.2],
        })
        .unwrap()
    }

    // --- sender_loop -----------------------------------------------------

    #[test]
    fn sender_loop_returns_immediately_when_not_running() {
        let (_tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        let socket = FakeSocket::send_only(None);
        let running = Arc::new(AtomicBool::new(false));

        let result = sender_loop(&socket, rx, running, sender_config());

        assert!(result.is_ok());
    }

    #[test]
    fn sender_loop_sends_encoded_packet_then_stops_on_disconnect() {
        let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        tx.send(vec![0.1, 0.2]).unwrap();
        drop(tx);
        let socket = FakeSocket::send_only(None);
        let running = Arc::new(AtomicBool::new(true));

        sender_loop(&socket, rx, running, sender_config()).unwrap();

        let sent = socket.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        let (payload, destination) = &sent[0];
        assert_eq!(*destination, addr());
        let decoded = decode_packet(payload).unwrap();
        assert_eq!(decoded.samples, vec![0.1, 0.2]);
    }

    #[test]
    fn sender_loop_propagates_encode_failure() {
        let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        tx.send(vec![0.1]).unwrap(); // wrong sample count for 2 channels x 1 frame
        drop(tx);
        let socket = FakeSocket::send_only(None);
        let running = Arc::new(AtomicBool::new(true));

        let error = sender_loop(&socket, rx, running, sender_config()).unwrap_err();

        assert!(error.to_string().contains("sample count"));
    }

    #[test]
    fn sender_loop_propagates_send_failure() {
        let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        tx.send(vec![0.1, 0.2]).unwrap();
        drop(tx);
        let socket = FakeSocket::send_only(Some(ErrorKind::Other));
        let running = Arc::new(AtomicBool::new(true));

        let error = sender_loop(&socket, rx, running, sender_config()).unwrap_err();

        assert!(error.to_string().contains("failed to send audio to"));
    }

    #[test]
    fn sender_loop_continues_through_recv_timeout() {
        // This sleep only needs to outlast thread *scheduling* latency
        // (routinely sub-millisecond) to guarantee the spawned thread's
        // first `running.load()` check reads `true` before this thread
        // flips it - otherwise the two racing loads can flip it before the
        // spawned thread ever checks, skipping `recv_timeout` (and its
        // `Timeout` arm) entirely. It does not need to outlast the 100ms
        // internal `recv_timeout` itself: once the spawned thread has
        // observed `running == true` and entered the loop body, `running`
        // is only re-checked after that call returns, so `join()` below
        // naturally waits out the rest with no further sleep-vs-runner-load
        // race.
        let (_tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        let socket = FakeSocket::send_only(None);
        let running = Arc::new(AtomicBool::new(true));
        let stopper = Arc::clone(&running);

        std::thread::scope(|scope| {
            let handle = scope.spawn(|| sender_loop(&socket, rx, running, sender_config()));
            std::thread::sleep(Duration::from_millis(20));
            stopper.store(false, Ordering::SeqCst);
            assert!(handle.join().unwrap().is_ok());
        });
    }

    // --- handle_datagram / buffer_samples -----------------------------------
    //
    // These exercise the per-packet business logic directly, without the
    // FakeSocket/receiver_loop scaffolding the loop-mechanics tests below
    // still need.

    #[test]
    fn handle_datagram_buffers_matching_packets_in_order() {
        let queue = Mutex::new(VecDeque::new());
        let mut expected_sequence = None;

        handle_datagram(
            &valid_packet(0),
            receiver_config(1024),
            &mut expected_sequence,
            &queue,
        )
        .unwrap();
        handle_datagram(
            &valid_packet(1),
            receiver_config(1024),
            &mut expected_sequence,
            &queue,
        )
        .unwrap();

        let buffered: Vec<f32> = queue.lock().unwrap().iter().copied().collect();
        assert_eq!(buffered, vec![0.1, 0.2, 0.1, 0.2]);
        assert_eq!(expected_sequence, Some(2));
    }

    #[test]
    fn handle_datagram_reports_sequence_discontinuity_but_keeps_buffering() {
        let queue = Mutex::new(VecDeque::new());
        let mut expected_sequence = None;

        handle_datagram(
            &valid_packet(0),
            receiver_config(1024),
            &mut expected_sequence,
            &queue,
        )
        .unwrap();
        handle_datagram(
            &valid_packet(5),
            receiver_config(1024),
            &mut expected_sequence,
            &queue,
        )
        .unwrap();

        assert_eq!(queue.lock().unwrap().len(), 4);
    }

    #[test]
    fn handle_datagram_ignores_packets_with_mismatched_format() {
        let queue = Mutex::new(VecDeque::new());
        let mismatched = encode_packet(&AudioPacket {
            channels: 1,
            sample_rate: format().sample_rate,
            sequence: 0,
            timestamp_us: 0,
            frames: 1,
            samples: vec![0.1],
        })
        .unwrap();
        let mut expected_sequence = None;

        handle_datagram(
            &mismatched,
            receiver_config(1024),
            &mut expected_sequence,
            &queue,
        )
        .unwrap();

        assert!(queue.lock().unwrap().is_empty());
    }

    #[test]
    fn handle_datagram_ignores_malformed_packets() {
        let queue = Mutex::new(VecDeque::new());
        let mut expected_sequence = None;

        handle_datagram(
            &[0u8; 4],
            receiver_config(1024),
            &mut expected_sequence,
            &queue,
        )
        .unwrap();

        assert!(queue.lock().unwrap().is_empty());
    }

    #[test]
    fn buffer_samples_evicts_oldest_samples_when_queue_is_full() {
        let queue = Mutex::new(VecDeque::new());

        buffer_samples(&queue, vec![0.1, 0.2], 2).unwrap();
        buffer_samples(&queue, vec![0.3, 0.4], 2).unwrap();

        let buffered: Vec<f32> = queue.lock().unwrap().iter().copied().collect();
        assert_eq!(buffered, vec![0.3, 0.4]);
    }

    #[test]
    fn buffer_samples_truncates_a_single_batch_larger_than_max_samples() {
        let queue = Mutex::new(VecDeque::new());

        buffer_samples(&queue, vec![0.1, 0.2], 1).unwrap();

        let buffered: Vec<f32> = queue.lock().unwrap().iter().copied().collect();
        assert_eq!(buffered, vec![0.2]);
    }

    #[test]
    fn buffer_samples_reports_poisoned_queue() {
        let queue = Mutex::new(VecDeque::new());
        std::thread::scope(|scope| {
            let _ = scope
                .spawn(|| {
                    let _guard = queue.lock().unwrap();
                    panic!("poison the mutex for test purposes");
                })
                .join();
        });

        let error = buffer_samples(&queue, vec![0.1], 8).unwrap_err();

        assert!(error.to_string().contains("playback queue poisoned"));
    }

    // --- receiver_loop -----------------------------------------------------
    //
    // Only the loop's own mechanics (not-running short-circuit, transient vs.
    // hard recv errors) live here; per-packet handling is covered above by
    // the handle_datagram/buffer_samples tests directly.

    #[test]
    fn receiver_loop_returns_immediately_when_not_running() {
        let running = Arc::new(AtomicBool::new(false));
        let socket = FakeSocket::recv_script(vec![], Arc::new(AtomicBool::new(true)));
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        let result = receiver_loop(&socket, queue, running, receiver_config(1024));

        assert!(result.is_ok());
    }

    #[test]
    fn receiver_loop_ignores_recv_timeout_and_would_block() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(
            vec![
                Err(Error::new(ErrorKind::WouldBlock, "no data yet")),
                Err(Error::new(ErrorKind::TimedOut, "no data yet")),
                Ok(valid_packet(0)),
            ],
            Arc::clone(&running),
        );
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        receiver_loop(&socket, Arc::clone(&queue), running, receiver_config(1024)).unwrap();

        assert_eq!(queue.lock().unwrap().len(), 2);
    }

    #[test]
    fn receiver_loop_propagates_hard_socket_errors() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(
            vec![Err(Error::new(ErrorKind::ConnectionReset, "reset"))],
            Arc::clone(&running),
        );
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        let error = receiver_loop(&socket, queue, running, receiver_config(1024)).unwrap_err();

        assert!(error.to_string().contains("failed receiving audio"));
    }

    /// Proves `handle_datagram`'s error path is actually wired up through
    /// `receiver_loop`'s `?`, not just correct in isolation (as the
    /// `buffer_samples_reports_poisoned_queue` unit test above already
    /// shows).
    #[test]
    fn receiver_loop_propagates_handle_datagram_errors() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(vec![Ok(valid_packet(0))], Arc::clone(&running));
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        let poisoned = Arc::clone(&queue);
        let _ = std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("poison the mutex for test purposes");
        })
        .join();

        let error = receiver_loop(&socket, queue, running, receiver_config(1024)).unwrap_err();

        assert!(error.to_string().contains("playback queue poisoned"));
    }

    // --- real loopback UDP sanity check -------------------------------------
    //
    // These exist to prove `impl AudioSocket for UdpSocket` faithfully
    // reflects real socket behaviour. Because sender_loop/receiver_loop take
    // `&dyn AudioSocket`, exercising them here contributes to the exact same
    // coverage counters as the `FakeSocket`-based tests above, rather than a
    // separate, independently-gated code path.

    #[test]
    fn real_udp_socket_round_trip_through_sender_loop() {
        let receiver_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        receiver_socket
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let receiver_addr = receiver_socket.local_addr().unwrap();

        let sender_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        tx.send(vec![0.1, 0.2]).unwrap();
        drop(tx);
        let running = Arc::new(AtomicBool::new(true));
        let config = SenderConfig {
            server: receiver_addr,
            ..sender_config()
        };

        sender_loop(&sender_socket, rx, running, config).unwrap();

        let mut buffer = [0u8; 65_535];
        let (length, _) = receiver_socket.recv_from(&mut buffer).unwrap();
        let decoded = decode_packet(&buffer[..length]).unwrap();
        assert_eq!(decoded.samples, vec![0.1, 0.2]);
    }

    #[test]
    fn real_udp_socket_round_trip_through_receiver_loop() {
        let receiver_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        receiver_socket
            .set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();
        let receiver_addr = receiver_socket.local_addr().unwrap();

        let sender_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let datagram = valid_packet(7);
        sender_socket.send_to(&datagram, receiver_addr).unwrap();

        let running = Arc::new(AtomicBool::new(true));
        let stopper = Arc::clone(&running);
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                receiver_loop(
                    &receiver_socket,
                    Arc::clone(&queue),
                    running,
                    receiver_config(1024),
                )
            });

            // Poll for the actual observable effect (the packet landing in
            // the queue) instead of sleeping a fixed duration: a loaded
            // runner could take longer than any fixed sleep to schedule the
            // receiver thread, and there's no reason to wait longer than
            // necessary once it has.
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while queue.lock().unwrap().is_empty() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "receiver_loop did not buffer the packet before the deadline"
                );
                std::thread::sleep(Duration::from_millis(5));
            }

            stopper.store(false, Ordering::SeqCst);
            handle.join().unwrap().unwrap();
        });

        let buffered: Vec<f32> = queue.lock().unwrap().iter().copied().collect();
        assert_eq!(buffered, vec![0.1, 0.2]);
    }
}
