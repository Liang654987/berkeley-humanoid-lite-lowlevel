# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import argparse
import math
import time

import numpy as np

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.robot.humanoid import (
    Humanoid,
    linear_interpolate,
)


INIT_STEPS = 100
DEFAULT_FREQUENCY = 25.0  # 100 initialization steps take about 4 seconds.
POSITION_TIMEOUT = 0.1
CAN_RETRY_DELAY = 0.02
STABLE_SAMPLES = 3
MAX_SAMPLE_SPREAD = 0.03
MAX_POSITION_STEP = 0.20
MAX_TRACKING_ERROR = 0.15
MAX_VELOCITY = 1.0
FINAL_TOLERANCE = 0.05
CAN_TIMEOUT_LIMIT = 10

SYMMETRIC_GROUPS = [
    (("can2", 1), ("can3", 2)),
    (("can2", 3), ("can3", 4)),
    (("can2", 5), ("can3", 6)),
    (("can2", 7), ("can3", 8)),
    (("can2", 11), ("can3", 12)),
    (("can2", 13), ("can3", 14)),
]


class TestError(RuntimeError):
    pass


class CommunicationError(TestError):
    def __init__(self, missing_joint_ids: list[int], attempts: int, timeout_count: int):
        super().__init__(f"PDO feedback missing for {attempts} consecutive attempts")
        self.missing_joint_ids = missing_joint_ids
        self.attempts = attempts
        self.timeout_count = timeout_count


class MotionError(TestError):
    def __init__(
        self,
        reason: str,
        group: list[int],
        fault_joint_ids: list[int],
        step: int,
        commands: dict[int, float],
        feedback: dict[int, tuple[float, float]],
        timeout_count: int,
        communication_failure: bool = False,
    ):
        super().__init__(reason)
        self.reason = reason
        self.group = group
        self.fault_joint_ids = fault_joint_ids
        self.step = step
        self.commands = commands
        self.feedback = feedback
        self.timeout_count = timeout_count
        self.communication_failure = communication_failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read joints or test the run_locomotion initialization one joint/group at a time."
    )
    parser.add_argument("--mode", choices=("read", "initialize"), required=True)
    parser.add_argument("-c", "--channel", choices=("can2", "can3"))
    parser.add_argument("-i", "--id", type=int, dest="device_id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--enable",
        action="store_true",
        help="required before the script is allowed to command motor motion",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_FREQUENCY,
        help="initialization update frequency in Hz (default: 25; about 4 seconds)",
    )
    parser.add_argument(
        "--max-travel",
        type=float,
        default=0.5,
        help="maximum allowed start-to-target travel in radians (default: 0.5)",
    )
    args = parser.parse_args()

    has_selector = args.channel is not None or args.device_id is not None
    if args.mode == "read" and args.all:
        parser.error("read mode requires -c/--channel and -i/--id; --all is for initialization only")
    if args.all and has_selector:
        parser.error("--all cannot be combined with -c/--channel or -i/--id")
    if not args.all and (args.channel is None or args.device_id is None):
        parser.error("use --all, or provide both -c/--channel and -i/--id")
    if args.mode == "initialize" and not args.enable:
        parser.error("initialize mode requires --enable")
    if args.frequency <= 0:
        parser.error("--frequency must be greater than zero")
    if args.max_travel <= 0:
        parser.error("--max-travel must be greater than zero")
    return args


def joint_lookup(robot: Humanoid) -> dict[tuple[str, int], int]:
    return {
        (bus.channel, device_id): joint_id
        for joint_id, (bus, device_id, _) in enumerate(robot.joints)
    }


def select_groups(robot: Humanoid, args: argparse.Namespace) -> list[list[int]]:
    lookup = joint_lookup(robot)
    requested_groups = SYMMETRIC_GROUPS if args.all else [[(args.channel, args.device_id)]]
    groups = []

    for requested_group in requested_groups:
        missing = [entry for entry in requested_group if entry not in lookup]
        if missing:
            missing_text = ", ".join(f"{channel} ID {device_id}" for channel, device_id in missing)
            raise TestError(f"joint mapping not found for {missing_text}")
        groups.append([lookup[entry] for entry in requested_group])

    return groups


def close_can_transports(robot: Humanoid) -> None:
    for bus in {bus for bus, _, _ in robot.joints}:
        bus.stop()


def model_position(robot: Humanoid, joint_id: int, motor_position: float) -> float:
    return (
        motor_position * robot.joint_axis_directions[joint_id]
        - robot.position_offsets[joint_id]
    )


def motor_position(robot: Humanoid, joint_id: int, model_target: float) -> float:
    return (
        (model_target + robot.position_offsets[joint_id])
        * robot.joint_axis_directions[joint_id]
    )


def read_position(robot: Humanoid, joint_id: int) -> tuple[float, float]:
    bus, device_id, joint_name = robot.joints[joint_id]
    try:
        raw_position = bus.read_position_measured(device_id, timeout=POSITION_TIMEOUT)
    except Exception as error:
        raise TestError(
            f"读取 {joint_name} ({bus.channel}, ID {device_id}) 位置失败: {error}"
        ) from error
    if raw_position is None or not math.isfinite(raw_position):
        raise TestError(
            f"no valid position feedback from {joint_name} "
            f"({bus.channel}, ID {device_id})"
        )
    return raw_position, model_position(robot, joint_id, raw_position)


def read_stable_group(robot: Humanoid, group: list[int]) -> dict[int, tuple[float, float]]:
    samples = {joint_id: [] for joint_id in group}

    for _ in range(STABLE_SAMPLES):
        for joint_id in group:
            samples[joint_id].append(read_position(robot, joint_id))
        time.sleep(0.02)

    result = {}
    for joint_id in group:
        model_samples = [sample[1] for sample in samples[joint_id]]
        spread = max(model_samples) - min(model_samples)
        if spread > MAX_SAMPLE_SPREAD:
            _, _, joint_name = robot.joints[joint_id]
            raise TestError(
                f"unstable feedback from {joint_name}: spread {spread:.4f} rad"
            )
        result[joint_id] = samples[joint_id][-1]

    return result


def joint_label(robot: Humanoid, joint_id: int) -> str:
    bus, device_id, joint_name = robot.joints[joint_id]
    return f"{joint_name} ({bus.channel}, ID {device_id})"


def print_group_plan(
    robot: Humanoid,
    group: list[int],
    starts: dict[int, tuple[float, float]],
    max_travel: float,
) -> None:
    print("\nJoint initialization plan:")
    print("joint                              current(rad/deg)     target(rad/deg)      travel(rad/deg)")
    for joint_id in group:
        _, current = starts[joint_id]
        target = float(robot.rl_init_positions[joint_id])
        travel = target - current
        print(
            f"{joint_label(robot, joint_id):34s} "
            f"{current:+7.3f}/{math.degrees(current):+7.2f}  "
            f"{target:+7.3f}/{math.degrees(target):+7.2f}  "
            f"{travel:+7.3f}/{math.degrees(travel):+7.2f}"
        )
        if abs(travel) > max_travel:
            raise TestError(
                f"{joint_label(robot, joint_id)} requires {abs(travel):.3f} rad, "
                f"exceeding --max-travel {max_travel:.3f} rad"
            )


def confirm_motion() -> bool:
    while True:
        answer = input("Type MOVE to run this group, or Q to stop: ").strip().upper()
        if answer == "MOVE":
            return True
        if answer == "Q":
            return False


def exchange_group(
    robot: Humanoid,
    group: list[int],
    commands: dict[int, float],
) -> tuple[dict[int, tuple[float, float]], int]:
    timeout_count = 0
    consecutive_timeouts = {joint_id: 0 for joint_id in group}

    while True:
        for joint_id in group:
            bus, device_id, _ = robot.joints[joint_id]
            bus.transmit_pdo_2(
                device_id,
                position_target=motor_position(robot, joint_id, commands[joint_id]),
                velocity_target=0.0,
            )

        feedback = {}
        missing = []
        for joint_id in group:
            bus, device_id, _ = robot.joints[joint_id]
            raw_position, raw_velocity = bus.receive_pdo_2(device_id)
            if (
                raw_position is None
                or raw_velocity is None
                or not math.isfinite(raw_position)
                or not math.isfinite(raw_velocity)
            ):
                missing.append(joint_id)
                continue
            feedback[joint_id] = (
                model_position(robot, joint_id, raw_position),
                raw_velocity * robot.joint_axis_directions[joint_id],
            )

        if not missing:
            return feedback, timeout_count
        for joint_id in group:
            if joint_id in missing:
                consecutive_timeouts[joint_id] += 1
                timeout_count += 1
            else:
                consecutive_timeouts[joint_id] = 0

        blocked = [
            joint_id
            for joint_id, count in consecutive_timeouts.items()
            if count >= CAN_TIMEOUT_LIMIT
        ]
        if blocked:
            raise CommunicationError(blocked, CAN_TIMEOUT_LIMIT, timeout_count)
        time.sleep(CAN_RETRY_DELAY)


def set_group_mode(robot: Humanoid, group: list[int], mode: int) -> None:
    for joint_id in group:
        bus, device_id, _ = robot.joints[joint_id]
        bus.feed(device_id)
        bus.set_mode(device_id, mode)


def hold_group_at_current_position(
    robot: Humanoid,
    group: list[int],
    starts: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    commands = {joint_id: starts[joint_id][1] for joint_id in group}
    try:
        feedback, timeout_count = exchange_group(robot, group, commands)
    except CommunicationError as error:
        raise MotionError(
            reason=f"连续{error.attempts}次未收到PDO反馈",
            group=group,
            fault_joint_ids=error.missing_joint_ids,
            step=0,
            commands=commands,
            feedback=starts,
            timeout_count=error.timeout_count,
            communication_failure=True,
        ) from error

    set_group_mode(robot, group, recoil.Mode.POSITION)
    if timeout_count:
        print(f"预加载当前位置时发生 {timeout_count} 次超时重试")
    return feedback


def emergency_damping(robot: Humanoid, joint_ids: list[int]) -> list[str]:
    errors = []
    for joint_id in dict.fromkeys(joint_ids):
        bus, device_id, _ = robot.joints[joint_id]
        try:
            bus.feed(device_id)
            bus.set_mode(device_id, recoil.Mode.DAMPING)
        except Exception as error:
            errors.append(f"{bus.channel} ID {device_id}: {error}")
    return errors


def enter_idle(robot: Humanoid) -> list[str]:
    errors = []
    for bus, device_id, _ in robot.joints:
        try:
            bus.feed(device_id)
            bus.set_mode(device_id, recoil.Mode.IDLE)
        except Exception as error:
            errors.append(f"{bus.channel} ID {device_id}: {error}")
    return errors


def wait_for_exit_interrupt() -> int:
    print("所有电机已切换到 IDLE（不保持位置），脚本即将退出。")
    return 130


def wait_for_damping_exit(robot: Humanoid) -> int:
    print("初始化失败，电机保持 DAMPING。按 Ctrl+C 切换到 IDLE 并退出脚本。")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        idle_errors = enter_idle(robot)
        if idle_errors:
            print("切换到 IDLE 时发生异常：" + "; ".join(idle_errors))
            return 1
        print("\n所有电机已切换到 IDLE（不保持位置），脚本退出。")
        return 130


def run_group(
    robot: Humanoid,
    group: list[int],
    starts: dict[int, tuple[float, float]],
    frequency: float,
) -> list[dict]:
    start_vector = np.array([starts[joint_id][1] for joint_id in group])
    target_vector = np.array([robot.rl_init_positions[joint_id] for joint_id in group])
    start_commands = {
        joint_id: float(start_vector[index])
        for index, joint_id in enumerate(group)
    }
    period = 1.0 / frequency
    last_feedback = {
        joint_id: (starts[joint_id][1], 0.0)
        for joint_id in group
    }
    timeout_count = 0

    # Match run_locomotion: preload measured positions, then enter position mode.
    try:
        feedback, timeout_count = exchange_group(
            robot,
            group,
            start_commands,
        )
    except CommunicationError as error:
        raise MotionError(
            reason=f"连续{error.attempts}次未收到PDO反馈",
            group=group,
            fault_joint_ids=error.missing_joint_ids,
            step=0,
            commands=start_commands,
            feedback=last_feedback,
            timeout_count=error.timeout_count,
            communication_failure=True,
        ) from error

    set_group_mode(robot, group, recoil.Mode.POSITION)

    previous_positions = {
        joint_id: feedback[joint_id][0]
        for joint_id in group
    }
    last_feedback = feedback

    for step in range(1, INIT_STEPS + 1):
        loop_start = time.perf_counter()
        command_vector = linear_interpolate(
            start_vector,
            target_vector,
            step / INIT_STEPS,
        )
        commands = {
            joint_id: float(command_vector[index])
            for index, joint_id in enumerate(group)
        }
        # The firmware watchdog requires HEARTBEAT even while PDOs are flowing.
        for joint_id in group:
            bus, device_id, _ = robot.joints[joint_id]
            bus.feed(device_id)
        try:
            last_feedback, new_timeouts = exchange_group(
                robot,
                group,
                commands,
            )
        except CommunicationError as error:
            raise MotionError(
                reason=f"连续{error.attempts}次未收到PDO反馈",
                group=group,
                fault_joint_ids=error.missing_joint_ids,
                step=step,
                commands=commands,
                feedback=last_feedback,
                timeout_count=timeout_count + error.timeout_count,
                communication_failure=True,
            ) from error
        timeout_count += new_timeouts

        for joint_id in group:
            position, velocity = last_feedback[joint_id]
            position_step = abs(position - previous_positions[joint_id])
            tracking_error = abs(commands[joint_id] - position)
            if position_step > MAX_POSITION_STEP:
                raise MotionError(
                    reason=f"位置单步跳变 {position_step:.3f} rad",
                    group=group,
                    fault_joint_ids=[joint_id],
                    step=step,
                    commands=commands,
                    feedback=last_feedback,
                    timeout_count=timeout_count,
                )
            if abs(velocity) > MAX_VELOCITY:
                raise MotionError(
                    reason=f"速度过高 {velocity:.3f} rad/s",
                    group=group,
                    fault_joint_ids=[joint_id],
                    step=step,
                    commands=commands,
                    feedback=last_feedback,
                    timeout_count=timeout_count,
                )
            if tracking_error > MAX_TRACKING_ERROR:
                raise MotionError(
                    reason=f"跟踪误差过大 {tracking_error:.3f} rad",
                    group=group,
                    fault_joint_ids=[joint_id],
                    step=step,
                    commands=commands,
                    feedback=last_feedback,
                    timeout_count=timeout_count,
                )
            previous_positions[joint_id] = position

        if step == 1 or step % 5 == 0 or step == INIT_STEPS:
            status = " | ".join(
                f"{robot.joints[joint_id][0].channel}:{robot.joints[joint_id][1]} "
                f"cmd={commands[joint_id]:+.3f} "
                f"pos={last_feedback[joint_id][0]:+.3f} "
                f"err={commands[joint_id] - last_feedback[joint_id][0]:+.3f} "
                f"vel={last_feedback[joint_id][1]:+.3f}"
                for joint_id in group
            )
            print(f"\r{step:3d}% | {status}", end="", flush=True)

        elapsed = time.perf_counter() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)

    print()
    results = []
    for index, joint_id in enumerate(group):
        final_position, final_velocity = last_feedback[joint_id]
        target = float(target_vector[index])
        error = target - final_position
        results.append({
            "joint_id": joint_id,
            "start": float(start_vector[index]),
            "target": target,
            "final": final_position,
            "error": error,
            "velocity": final_velocity,
            "timeouts": timeout_count,
            "passed": abs(error) <= FINAL_TOLERANCE,
        })
    return results


def print_results(robot: Humanoid, results: list[dict]) -> None:
    if not results:
        return
    print("\nFinal results:")
    print("joint                              start     target    final(rad/deg)       error     CAN   result")
    for result in results:
        print(
            f"{joint_label(robot, result['joint_id']):34s} "
            f"{result['start']:+8.3f} "
            f"{result['target']:+8.3f} "
            f"{result['final']:+8.3f}/{math.degrees(result['final']):+7.2f} "
            f"{result['error']:+8.3f} "
            f"{result['timeouts']:7d}   "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )


def diagnose_can_channels(robot: Humanoid, fault_joint_ids: list[int]) -> list[str]:
    affected_channels = {
        robot.joints[joint_id][0].channel
        for joint_id in fault_joint_ids
    }
    reports = []

    for channel in sorted(affected_channels):
        channel_joints = [
            (bus, device_id)
            for bus, device_id, _ in robot.joints
            if bus.channel == channel
        ]
        responsive = []
        for bus, device_id in channel_joints:
            try:
                if bus.ping(device_id, timeout=0.02):
                    responsive.append(device_id)
            except Exception:
                pass

        if not responsive:
            reports.append(f"{channel}全部设备无响应，通道可能完全阻断")
        else:
            device_list = ", ".join(str(device_id) for device_id in responsive)
            reports.append(
                f"{channel}仍有设备响应（ID {device_list}），可能是节点或PDO通信故障"
            )

    return reports


def print_emergency_report(
    robot: Humanoid,
    error: MotionError,
    damping_errors: list[str],
) -> None:
    print("\n========== EMERGENCY STOP ==========")
    print(f"原因：{error.reason}")
    print(f"初始化步骤：{error.step}/{INIT_STEPS}")
    print(f"累计丢帧计数：{error.timeout_count}")

    for joint_id in error.fault_joint_ids:
        command = error.commands.get(joint_id)
        position, velocity = error.feedback.get(joint_id, (None, None))
        print(f"故障设备：{joint_label(robot, joint_id)}")
        if command is not None:
            print(f"最后指令：{command:+.3f} rad ({math.degrees(command):+.2f} deg)")
        if position is not None:
            print(f"最后有效位置：{position:+.3f} rad ({math.degrees(position):+.2f} deg)")
            if command is not None:
                print(f"最后有效误差：{command - position:+.3f} rad")
        if velocity is not None:
            print(f"最后有效速度：{velocity:+.3f} rad/s")

    if error.communication_failure:
        for report in diagnose_can_channels(robot, error.fault_joint_ids):
            print(f"通道检测：{report}")

    stopped_devices = ", ".join(
        f"{robot.joints[joint_id][0].channel} ID {robot.joints[joint_id][1]}"
        for joint_id in error.group
    )
    print(f"处理：已尝试向活动组发送DAMPING命令（{stopped_devices}）")
    if damping_errors:
        print("阻尼发送异常：" + "; ".join(damping_errors))
    else:
        print("注意：DAMPING命令没有应答确认，仍需观察实机并准备物理急停")
    print("====================================")


def run_read_mode(robot: Humanoid, joint_id: int) -> None:
    print("Read-only mode: no modes, gains, or targets will be written. Press Ctrl+C to stop.")
    while True:
        print("\njoint                              raw(rad)   model(rad/deg)       CAN")
        try:
            raw_position, model_angle = read_position(robot, joint_id)
            value = f"{raw_position:+8.3f}   {model_angle:+8.3f}/{math.degrees(model_angle):+7.2f}   OK"
        except TestError:
            value = "     ---         ---/    ---   TIMEOUT"
        print(f"{joint_label(robot, joint_id):34s} {value}")
        time.sleep(0.5)


def run_initialize_mode(
    robot: Humanoid,
    groups: list[list[int]],
    args: argparse.Namespace,
) -> list[dict]:
    # This is the same gain/mode setup used by run_locomotion.py.
    try:
        robot.enter_damping()
    except Exception as error:
        raise TestError(f"电机进入 DAMPING 初始化配置失败: {error}") from error
    all_results = []

    for group_number, group in enumerate(groups, start=1):
        # Damping mode lets a joint move while its current position is sampled.
        # Preload that position and hold it before checking feedback stability.
        starts = {
            joint_id: read_position(robot, joint_id)
            for joint_id in group
        }
        hold_group_at_current_position(robot, group, starts)
        starts = read_stable_group(robot, group)
        print(f"\nGroup {group_number}/{len(groups)}")
        print_group_plan(robot, group, starts, args.max_travel)
        if not confirm_motion():
            break

        interrupted = False
        try:
            results = run_group(
                robot,
                group,
                starts,
                args.frequency,
            )
            all_results.extend(results)
            print_results(robot, results)
            if not all(result["passed"] for result in results):
                print("Group failed; remaining groups will not be started.")
                break
        except KeyboardInterrupt:
            interrupted = True
            set_group_mode(robot, group, recoil.Mode.IDLE)
            raise
        finally:
            if not interrupted:
                emergency_damping(robot, group)

    return all_results


def main() -> int:
    args = parse_args()
    robot = Humanoid(start_peripherals=False)
    groups = []
    initialize_mode = args.mode == "initialize"
    idle_requested = False

    try:
        groups = select_groups(robot, args)
        if initialize_mode:
            results = run_initialize_mode(robot, groups, args)
            print_results(robot, results)
            return 0 if all(result["passed"] for result in results) else 1
        run_read_mode(robot, groups[0][0])
    except KeyboardInterrupt:
        idle_requested = True
        idle_errors = enter_idle(robot)
        if idle_errors:
            print("切换到 IDLE 时发生异常：" + "; ".join(idle_errors))
        return wait_for_exit_interrupt()
    except MotionError as error:
        damping_errors = emergency_damping(robot, error.group)
        print_emergency_report(robot, error, damping_errors)
        idle_requested = True
        return wait_for_damping_exit(robot)
    except TestError as error:
        print(f"\nABORTED: {error}")
        if initialize_mode:
            damping_errors = emergency_damping(
                robot,
                [joint_id for group in groups for joint_id in group],
            )
            if damping_errors:
                print("阻尼发送异常：" + "; ".join(damping_errors))
            idle_requested = True
            return wait_for_damping_exit(robot)
        return 1
    except Exception as error:
        print(f"\n初始化过程中发生未处理异常: {error}")
        if initialize_mode:
            damping_errors = emergency_damping(
                robot,
                [joint_id for group in groups for joint_id in group],
            )
            if damping_errors:
                print("阻尼发送异常：" + "; ".join(damping_errors))
            idle_requested = True
            return wait_for_damping_exit(robot)
        return 1
    finally:
        if initialize_mode and not idle_requested:
            emergency_damping(
                robot,
                [joint_id for group in groups for joint_id in group],
            )
        close_can_transports(robot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
