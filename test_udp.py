import argparse
import socket
import time


def receive(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(5.0)
    print(f"Listening on UDP {port}")

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except TimeoutError:
            print("No packet received for 5 seconds")
            continue
        print(f"From {addr}: {data!r}")


def send(server: str, port: int, count: int, interval: float) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for sequence in range(count):
        payload = f"audio-test-packet:{sequence}".encode()
        sock.sendto(payload, (server, port))
        print(f"Sent {payload!r} to {server}:{port}")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    receive_parser = subparsers.add_parser("receive")
    receive_parser.add_argument("--port", type=int, default=5006)

    send_parser = subparsers.add_parser("send")
    send_parser.add_argument("--server", default="127.0.0.1")
    send_parser.add_argument("--port", type=int, default=5004)
    send_parser.add_argument("--count", type=int, default=10)
    send_parser.add_argument("--interval", type=float, default=0.1)

    args = parser.parse_args()
    if args.command == "receive":
        receive(args.port)
    else:
        send(args.server, args.port, args.count, args.interval)


if __name__ == "__main__":
    main()
