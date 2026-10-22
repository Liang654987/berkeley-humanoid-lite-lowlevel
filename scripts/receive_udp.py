#!/usr/bin/env python3
"""Receive NumPy observation arrays sent by run_idle.py over UDP."""

import argparse
import socket
from datetime import datetime

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive and print robot observations sent over UDP."
    )
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="Local address to listen on (default: 0.0.0.0, all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11000,
        help="Local UDP port to listen on (default: 11000)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="NumPy type used by the sender (default: float32)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=35,
        help="Expected number of values per packet; use 0 to disable the check (default: 35)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))

    np.set_printoptions(precision=5, suppress=True, linewidth=160)
    print(f"Listening for UDP packets on {args.bind}:{args.port} ({dtype})")
    print("Press Ctrl+C to stop.")

    packet_count = 0
    try:
        while True:
            payload, sender = sock.recvfrom(65535)
            packet_count += 1

            if len(payload) % dtype.itemsize != 0:
                print(
                    f"Packet {packet_count} from {sender[0]}:{sender[1]} ignored: "
                    f"{len(payload)} bytes is not divisible by {dtype.itemsize}."
                )
                continue

            values = np.frombuffer(payload, dtype=dtype)
            timestamp = datetime.now().isoformat(timespec="milliseconds")

            count_warning = ""
            if args.expected_count and values.size != args.expected_count:
                count_warning = (
                    f" [expected {args.expected_count} values; check --dtype if this is wrong]"
                )

            print(
                f"[{timestamp}] packet={packet_count} "
                f"from={sender[0]}:{sender[1]} values={values.size}{count_warning}\n"
                f"{values}"
            )
    except KeyboardInterrupt:
        print("\nReceiver stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
