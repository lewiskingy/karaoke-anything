use anyhow::{anyhow, Context, Result};
use crossbeam_channel::Receiver;
use std::collections::VecDeque;
use std::net::{SocketAddr, UdpSocket};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use crate::protocol::{decode_packet, encode_packet};

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

pub fn sender_loop(
    socket: &dyn AudioSocket,
    server: SocketAddr,
    packets: Receiver<Vec<f32>>,
    running: Arc<AtomicBool>,
    channels: u16,
    sample_rate: u32,
    frames_per_packet: u16,
) -> Result<()> {
    let started = Instant::now();
    let mut sequence = 0u32;

    while running.load(Ordering::SeqCst) {
        match packets.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                let timestamp_us = started.elapsed().as_micros() as u64;
                let datagram = encode_packet(
                    channels,
                    sample_rate,
                    sequence,
                    timestamp_us,
                    frames_per_packet,
                    &samples,
                )?;
                socket
                    .send_datagram(&datagram, server)
                    .with_context(|| format!("failed to send audio to {server}"))?;
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
    expected_channels: u16,
    expected_sample_rate: u32,
    max_samples: usize,
) -> Result<()> {
    let mut buffer = vec![0u8; 65_535];
    let mut expected_sequence: Option<u32> = None;

    while running.load(Ordering::SeqCst) {
        match socket.recv_datagram(&mut buffer) {
            Ok((length, _)) => match decode_packet(&buffer[..length]) {
                Ok(packet) => {
                    if packet.channels != expected_channels
                        || packet.sample_rate != expected_sample_rate
                    {
                        eprintln!(
                            "ignored incompatible packet: {}Hz/{}ch",
                            packet.sample_rate, packet.channels
                        );
                        continue;
                    }

                    if let Some(expected) = expected_sequence {
                        if packet.sequence != expected {
                            eprintln!(
                                "sequence discontinuity: expected {}, received {}",
                                expected, packet.sequence
                            );
                        }
                    }
                    expected_sequence = Some(packet.sequence.wrapping_add(1));

                    let mut queue = queue
                        .lock()
                        .map_err(|_| anyhow!("playback queue poisoned"))?;
                    let keep = packet.samples.len().min(max_samples);
                    let dropped = packet.samples.len() - keep;
                    while queue.len() + keep > max_samples && !queue.is_empty() {
                        queue.pop_front();
                    }
                    queue.extend(packet.samples.into_iter().skip(dropped));
                }
                Err(error) => eprintln!("ignored invalid packet: {error:#}"),
            },
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock
                    || error.kind() == std::io::ErrorKind::TimedOut => {}
            Err(error) => return Err(error).context("failed receiving audio"),
        }
    }
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

    // --- sender_loop -----------------------------------------------------

    #[test]
    fn sender_loop_returns_immediately_when_not_running() {
        let (_tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        let socket = FakeSocket::send_only(None);
        let running = Arc::new(AtomicBool::new(false));

        let result = sender_loop(&socket, addr(), rx, running, 2, 48_000, 1);

        assert!(result.is_ok());
    }

    #[test]
    fn sender_loop_sends_encoded_packet_then_stops_on_disconnect() {
        let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        tx.send(vec![0.1, 0.2]).unwrap();
        drop(tx);
        let socket = FakeSocket::send_only(None);
        let running = Arc::new(AtomicBool::new(true));

        sender_loop(&socket, addr(), rx, running, 2, 48_000, 1).unwrap();

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

        let error = sender_loop(&socket, addr(), rx, running, 2, 48_000, 1).unwrap_err();

        assert!(error.to_string().contains("sample count"));
    }

    #[test]
    fn sender_loop_propagates_send_failure() {
        let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        tx.send(vec![0.1, 0.2]).unwrap();
        drop(tx);
        let socket = FakeSocket::send_only(Some(ErrorKind::Other));
        let running = Arc::new(AtomicBool::new(true));

        let error = sender_loop(&socket, addr(), rx, running, 2, 48_000, 1).unwrap_err();

        assert!(error.to_string().contains("failed to send audio to"));
    }

    #[test]
    fn sender_loop_continues_through_recv_timeout() {
        let (_tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(1);
        let socket = FakeSocket::send_only(None);
        let running = Arc::new(AtomicBool::new(true));
        let stopper = Arc::clone(&running);

        std::thread::scope(|scope| {
            let handle = scope.spawn(|| sender_loop(&socket, addr(), rx, running, 2, 48_000, 1));
            std::thread::sleep(Duration::from_millis(150));
            stopper.store(false, Ordering::SeqCst);
            assert!(handle.join().unwrap().is_ok());
        });
    }

    // --- receiver_loop -----------------------------------------------------

    fn valid_packet(sequence: u32) -> Vec<u8> {
        encode_packet(2, 48_000, sequence, 0, 1, &[0.1, 0.2]).unwrap()
    }

    #[test]
    fn receiver_loop_returns_immediately_when_not_running() {
        let running = Arc::new(AtomicBool::new(false));
        let socket = FakeSocket::recv_script(vec![], Arc::new(AtomicBool::new(true)));
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        let result = receiver_loop(&socket, queue, running, 2, 48_000, 1024);

        assert!(result.is_ok());
    }

    #[test]
    fn receiver_loop_buffers_matching_packets_in_order() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(
            vec![Ok(valid_packet(0)), Ok(valid_packet(1))],
            Arc::clone(&running),
        );
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        receiver_loop(&socket, Arc::clone(&queue), running, 2, 48_000, 1024).unwrap();

        let buffered: Vec<f32> = queue.lock().unwrap().iter().copied().collect();
        assert_eq!(buffered, vec![0.1, 0.2, 0.1, 0.2]);
    }

    #[test]
    fn receiver_loop_reports_sequence_discontinuity_but_keeps_buffering() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(
            vec![Ok(valid_packet(0)), Ok(valid_packet(5))],
            Arc::clone(&running),
        );
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        receiver_loop(&socket, Arc::clone(&queue), running, 2, 48_000, 1024).unwrap();

        assert_eq!(queue.lock().unwrap().len(), 4);
    }

    #[test]
    fn receiver_loop_ignores_packets_with_mismatched_format() {
        let running = Arc::new(AtomicBool::new(true));
        let mismatched = encode_packet(1, 48_000, 0, 0, 1, &[0.1]).unwrap();
        let socket = FakeSocket::recv_script(vec![Ok(mismatched)], Arc::clone(&running));
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        receiver_loop(&socket, Arc::clone(&queue), running, 2, 48_000, 1024).unwrap();

        assert!(queue.lock().unwrap().is_empty());
    }

    #[test]
    fn receiver_loop_ignores_malformed_packets() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(vec![Ok(vec![0u8; 4])], Arc::clone(&running));
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        receiver_loop(&socket, Arc::clone(&queue), running, 2, 48_000, 1024).unwrap();

        assert!(queue.lock().unwrap().is_empty());
    }

    #[test]
    fn receiver_loop_evicts_oldest_samples_when_queue_is_full() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(
            vec![Ok(valid_packet(0)), Ok(valid_packet(1))],
            Arc::clone(&running),
        );
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        receiver_loop(&socket, Arc::clone(&queue), running, 2, 48_000, 2).unwrap();

        let buffered: Vec<f32> = queue.lock().unwrap().iter().copied().collect();
        assert_eq!(buffered, vec![0.1, 0.2]);
    }

    #[test]
    fn receiver_loop_truncates_a_single_packet_larger_than_max_samples() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(vec![Ok(valid_packet(0))], Arc::clone(&running));
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        receiver_loop(&socket, Arc::clone(&queue), running, 2, 48_000, 1).unwrap();

        let buffered: Vec<f32> = queue.lock().unwrap().iter().copied().collect();
        assert_eq!(buffered.len(), 1);
        assert_eq!(buffered, vec![0.2]);
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

        receiver_loop(&socket, Arc::clone(&queue), running, 2, 48_000, 1024).unwrap();

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

        let error = receiver_loop(&socket, queue, running, 2, 48_000, 1024).unwrap_err();

        assert!(error.to_string().contains("failed receiving audio"));
    }

    #[test]
    fn receiver_loop_reports_poisoned_queue() {
        let running = Arc::new(AtomicBool::new(true));
        let socket = FakeSocket::recv_script(vec![Ok(valid_packet(0))], Arc::clone(&running));
        let queue = Arc::new(Mutex::new(VecDeque::new()));

        let poisoned = Arc::clone(&queue);
        let _ = std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("poison the mutex for test purposes");
        })
        .join();

        let error = receiver_loop(&socket, queue, running, 2, 48_000, 1024).unwrap_err();

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

        sender_loop(&sender_socket, receiver_addr, rx, running, 2, 48_000, 1).unwrap();

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
                    2,
                    48_000,
                    1024,
                )
            });
            std::thread::sleep(Duration::from_millis(150));
            stopper.store(false, Ordering::SeqCst);
            handle.join().unwrap().unwrap();
        });

        let buffered: Vec<f32> = queue.lock().unwrap().iter().copied().collect();
        assert_eq!(buffered, vec![0.1, 0.2]);
    }
}
