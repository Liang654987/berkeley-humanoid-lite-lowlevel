# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.robot.humanoid import Humanoid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read or configure actuator gear ratios. Without --gear-ratio, only read parameters.",
        epilog="Examples: --all | -c can3 -i 14 --gear-ratio -15 --store",
    )
    parser.add_argument("-c", "--channel")
    parser.add_argument("-i", "--id", type=int, dest="device_id")
    parser.add_argument("--all", action="store_true", help="select all 12 leg joints")
    parser.add_argument(
        "--gear-ratio", type=float,
        help="signed firmware gear ratio to write (standard robot: -15)",
    )
    parser.add_argument("--store", action="store_true", help="also send the Flash save command")
    parser.add_argument("--timeout", type=float, default=0.1)
    args = parser.parse_args()
    if args.all and (args.channel is not None or args.device_id is not None):
        parser.error("--all cannot be combined with --channel or --id")
    if not args.all and (args.channel is None or args.device_id is None):
        parser.error("use --all, or provide both --channel and --id")
    if args.store and args.gear_ratio is None:
        parser.error("--store requires --gear-ratio")
    if args.gear_ratio is not None and (
        not math.isfinite(args.gear_ratio) or args.gear_ratio == 0
    ):
        parser.error("--gear-ratio must be finite and nonzero")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and greater than zero")
    return args


def configure_gear_ratio(bus: recoil.Bus, device_id: int, ratio: float, timeout: float) -> None:
    # Change the feedback coordinate system only while position control is off.
    bus.set_mode(device_id, recoil.Mode.IDLE)
    time.sleep(0.05)
    bus.write_gear_ratio(device_id, ratio)
    time.sleep(0.05)
    actual = bus.read_gear_ratio(device_id, timeout=timeout)
    if actual is None or not math.isclose(actual, ratio, rel_tol=1e-6):
        raise RuntimeError(
            f"{bus.channel} ID {device_id}: gear ratio readback {actual}, expected {ratio}; "
            "Flash save command was not sent for this actuator"
        )
    print(f"{bus.channel} ID {device_id}: gear_ratio={actual} verified")


def main() -> int:
    args = parse_args()
    buses = set()
    try:
        if args.all:
            robot = Humanoid(start_peripherals=False)
            joints = robot.joints
            buses = {bus for bus, _, _ in joints}
        else:
            bus = recoil.Bus(args.channel)
            buses.add(bus)
            joints = [(bus, args.device_id, "actuator")]

        # Read every selected device before allowing the first parameter write.
        all_match = True
        for bus, device_id, name in joints:
            ratio = bus.read_gear_ratio(device_id, timeout=args.timeout)
            if ratio is None or not math.isfinite(ratio):
                raise RuntimeError(f"{name} ({bus.channel}, ID {device_id}): no valid gear ratio feedback")
            matches = math.isclose(ratio, Humanoid.JOINT_GEAR_RATIO, rel_tol=1e-6)
            all_match = all_match and matches
            print(
                f"{name} ({bus.channel}, ID {device_id}): gear_ratio={ratio}, "
                f"expected {Humanoid.JOINT_GEAR_RATIO} [{'OK' if matches else 'MISMATCH'}]"
            )

        if args.gear_ratio is None:
            return 0 if all_match else 1

        for bus, device_id, _ in joints:
            configure_gear_ratio(bus, device_id, args.gear_ratio, args.timeout)
            if args.store:
                bus.store_settings_to_flash(device_id)
                time.sleep(0.5)
                print(f"{bus.channel} ID {device_id}: Flash save command sent")

        if args.store:
            print("Power-cycle the actuators, then read the gear ratios again to verify persistence.")
        else:
            print("Parameters changed in RAM only; use --store to persist them across power cycles.")
        print("Re-run scripts/calibrate_joints.py before initialization; previous offsets must not be reused.")
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        for bus in buses:
            bus.stop()


if __name__ == "__main__":
    raise SystemExit(main())
