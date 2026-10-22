"""
calibrate_joints.py

Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

Run this script after each power cycle to calibrate the encoder offset of each joint.
"""

import argparse
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from berkeley_humanoid_lite_lowlevel.robot import Humanoid


CALIBRATION_PATH = ROOT / "calibration.yaml"

IDEAL_VALUES = np.deg2rad(np.array([
    -10.0, +33.75, +56.25,
    0.0,
    -45.0, -15.0,
    +10.0, -33.75, +56.25,
    0.0,
    -45.0, +15.0,
]))

# True means that the positive extreme is the mechanical calibration limit.
TAKE_MAXIMUM = np.array([
    False, True, True,
    False,
    False, False,
    True, False, True,
    False,
    False, True,
])


class JointReadError(RuntimeError):
    """Raised when an actuator does not answer a position request in time."""


class StopKeyMonitor:
    """Read q or Enter without making the user press Enter after q."""

    def __init__(self):
        self._fd = None
        self._terminal_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._terminal_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._terminal_settings is not None:
            termios.tcsetattr(
                self._fd,
                termios.TCSADRAIN,
                self._terminal_settings,
            )

    def requested(self) -> bool:
        while select.select([sys.stdin], [], [], 0.0)[0]:
            if self._fd is not None:
                key = os.read(self._fd, 1).decode(errors="ignore")
            else:
                key = sys.stdin.read(1)
            if key == "":
                return False
            if key in ("q", "Q", "\n", "\r"):
                return True
        return False


def read_joint_positions(robot: Humanoid, timeout: float) -> np.ndarray:
    readings = []
    missing_joints = []
    for bus, device_id, joint_name in robot.joints:
        reading = bus.read_position_measured(device_id, timeout=timeout)
        if reading is None:
            missing_joints.append(
                f"{joint_name} ({bus.channel}, device ID {device_id})"
            )
        else:
            readings.append(reading)

    if missing_joints:
        raise JointReadError(
            f"no response within {timeout:.3f} s from: "
            + ", ".join(missing_joints)
        )

    return np.asarray(readings) * robot.joint_axis_directions


def update_limits(limit_readings: np.ndarray,
                  joint_readings: np.ndarray) -> np.ndarray:
    return np.where(
        TAKE_MAXIMUM,
        np.maximum(limit_readings, joint_readings),
        np.minimum(limit_readings, joint_readings),
    )


def close_can_transports(robot: Humanoid) -> None:
    """Close each shared CAN transport once; calibration starts no motors."""
    transports = {bus for bus, _, _ in robot.joints}
    for bus in transports:
        bus.stop()


def calibrate(robot: Humanoid, timeout: float) -> np.ndarray:
    if len(robot.joints) != len(IDEAL_VALUES):
        raise ValueError(
            f"expected {len(IDEAL_VALUES)} joints, found {len(robot.joints)}"
        )

    print("Reading initial joint positions...", flush=True)
    limit_readings = read_joint_positions(robot, timeout)
    print([f"{reading:.4f}" for reading in limit_readings])
    print(
        "Move every joint to its mechanical calibration limit. "
        "Press q or Enter to finish and save; Ctrl+C cancels without saving."
    )

    with StopKeyMonitor() as keyboard:
        while not keyboard.requested():
            joint_readings = read_joint_positions(robot, timeout)
            limit_readings = update_limits(limit_readings, joint_readings)
            print(
                "\rLimits: "
                + " ".join(f"{reading:+.4f}" for reading in limit_readings),
                end="",
                flush=True,
            )
            time.sleep(0.05)

    print()
    return limit_readings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure joint mechanical limits and write calibration.yaml."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.1,
        help="maximum seconds to wait for each actuator response (default: 0.1)",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    robot = None

    try:
        # Calibration only needs CAN. Avoid starting the IMU and gamepad threads,
        # which can otherwise keep Python alive after a keyboard interruption.
        robot = Humanoid(start_peripherals=False)
        limit_readings = calibrate(robot, args.timeout)
        offsets = limit_readings - IDEAL_VALUES

        print("Final readings at the limits:")
        print([f"{limit:.4f}" for limit in limit_readings])
        print("Offsets:")
        print([f"{offset:.4f}" for offset in offsets])

        calibration_data = {
            "position_offsets": [float(offset) for offset in offsets],
        }
        with CALIBRATION_PATH.open("w") as calibration_file:
            yaml.safe_dump(calibration_data, calibration_file)
        print(f"Calibration saved to {CALIBRATION_PATH}")
        return 0
    except JointReadError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        print(
            "Check that both CAN interfaces are UP, motor power is on, and the "
            "configured CAN channel/device ID matches the wiring. No calibration "
            "file was changed.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nCalibration cancelled; no calibration file was changed.")
        return 130
    finally:
        if robot is not None:
            close_can_transports(robot)


if __name__ == "__main__":
    raise SystemExit(main())
