# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import math
import time
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.robot.imu import SerialImu, Baudrate
from berkeley_humanoid_lite_lowlevel.policy.gamepad import Se2Gamepad


class State:
    INVALID = 0
    IDLE = 1
    RL_INIT = 2
    RL_RUNNING = 3


def linear_interpolate(start: np.ndarray, end: np.ndarray, percentage: float) -> np.ndarray:
    percentage = min(max(percentage, 0.0), 1.0)
    target = start * (1. - percentage) + end * percentage
    return target


class JointConfigurationError(RuntimeError):
    pass


def receive_initialization_feedback(bus, device_id):
    frame = bus.receive(
        filter_device_id=device_id,
        filter_function=recoil.Function.TRANSMIT_PDO_2,
        timeout=0.001,
    )
    if frame is None:
        return None, None
    feedback = bus.unpack("<ff", frame.data)
    if len(feedback) != 2 or not all(math.isfinite(value) for value in feedback):
        raise RuntimeError(f"{bus.channel} ID {device_id}: invalid position/velocity feedback")
    return feedback


class InitializationRamp:
    MAX_SPEED = 0.10
    MAX_ACCELERATION = 0.20
    MAX_TIMESTEP = 0.05
    PAUSE_ERROR = 0.08
    RESUME_ERROR = 0.06
    PROGRESS_WINDOW = 0.5
    STALL_TIMEOUT = 2.0
    TIMEOUT = 30.0

    def __init__(self, start, target, labels, progress_error, now):
        self.start = start
        self.target = target
        self.labels = labels
        self.progress_error = progress_error
        self.distance = float(np.max(np.abs(target - start)))
        self.travelled = 0.0
        self.speed = 0.0
        self.paused = False
        self.started_at = self.updated_at = self.window_started_at = now
        self.window_error = np.abs(target - start)
        self.stall_time = np.zeros(len(start))

    @property
    def progress(self):
        return self.travelled / self.distance if self.distance else 1.0

    @property
    def command(self):
        return linear_interpolate(self.start, self.target, self.progress)

    def advance(self, measured, received, now):
        dt = min(now - self.updated_at, self.MAX_TIMESTEP)
        self.updated_at = now
        if self.progress >= 1.0:
            return
        if now - self.started_at >= self.TIMEOUT:
            raise RuntimeError("Initialization trajectory timed out")
        if not received:
            # Hold the last command until a complete fresh set of feedback arrives.
            self.speed = 0.0
            self.window_started_at = now
            self.window_error = None
            self.stall_time.fill(0.0)
            return

        tracking_error = np.abs(self.command - measured)
        goal_error = np.abs(self.target - measured)
        if self.window_error is None:
            self.window_error = goal_error
            self.window_started_at = now
        window = now - self.window_started_at
        if window >= self.PROGRESS_WINDOW:
            improvement = self.window_error - goal_error
            outside_goal = goal_error > self.progress_error
            wrong_way = np.flatnonzero(outside_goal & (improvement < -0.02))
            if wrong_way.size:
                raise RuntimeError(f"{self.labels[wrong_way[0]]}: moving away from initialization target")
            stalled = outside_goal & (tracking_error > self.progress_error) & (improvement < 0.003)
            self.stall_time = np.where(stalled, self.stall_time + window, 0.0)
            timed_out = np.flatnonzero(self.stall_time >= self.STALL_TIMEOUT)
            if timed_out.size:
                raise RuntimeError(f"{self.labels[timed_out[0]]}: initialization stalled")
            self.window_error = goal_error
            self.window_started_at = now

        pause_error = self.RESUME_ERROR if self.paused else self.PAUSE_ERROR
        self.paused = float(np.max(tracking_error)) > pause_error
        self._advance_reference(dt)

    def _advance_reference(self, dt):
        # All joints share one progress value. The longest travel sets the limits.
        acceleration = self.MAX_ACCELERATION
        if self.paused:
            phases = ((-acceleration, self.speed / acceleration),)
        else:
            remaining = self.distance - self.travelled
            peak = min(self.MAX_SPEED, math.sqrt(acceleration * remaining + self.speed**2 / 2))
            peak = max(self.speed, peak)
            accelerate = (peak - self.speed) / acceleration
            brake = peak / acceleration
            cruise = max(0.0, (remaining - (self.speed + peak) * accelerate / 2 - peak * brake / 2) / peak)
            phases = ((acceleration, accelerate), (0.0, cruise), (-acceleration, brake))
        for rate, duration in phases:
            step = min(dt, duration)
            self.travelled += (self.speed + rate * step / 2) * step
            self.speed = max(0.0, self.speed + rate * step)
            dt -= step
        if self.distance - self.travelled < 1e-9:
            self.travelled = self.distance
            self.speed = 0.0


class Humanoid:
    # The firmware divides motor-shaft angles by this signed gearbox ratio.
    JOINT_GEAR_RATIO = -15.0
    INIT_SETTLING_TIMEOUT = 2.0
    INIT_STABLE_SAMPLES = 3
    INIT_SETTLING_VELOCITY = 0.05
    INIT_STABLE_TIME = 0.2
    INIT_MAX_TRAVEL = 0.5
    INIT_MAX_SAMPLE_SPREAD = 0.03
    INIT_MAX_POSITION_STEP = 0.20
    INIT_MAX_VELOCITY = 1.0
    INIT_MAX_TRACKING_ERROR = 0.15
    INIT_CAN_TIMEOUT_LIMIT = 10

    def __init__(self, start_peripherals: bool = True):

        # self.left_arm_transport = recoil.Bus("can0")
        # self.right_arm_transport = recoil.Bus("can1")
        self.left_leg_transport = recoil.Bus("can2")
        self.right_leg_transport = recoil.Bus("can3")

        #self.left_leg_transport = recoil.Bus("can0")
        #self.right_leg_transport = recoil.Bus("can1")

        self.joints = [
            # (self.left_arm_transport,   1,  "left_shoulder_pitch_joint"     ),  # noqa: E241
            # (self.left_arm_transport,   3,  "left_shoulder_roll_joint"      ),  # noqa: E241
            # (self.left_arm_transport,   5,  "left_shoulder_yaw_joint"       ),  # noqa: E241
            # (self.left_arm_transport,   7,  "left_elbow_pitch_joint"        ),  # noqa: E241
            # (self.left_arm_transport,   9,  "left_wrist_yaw_joint"          ),  # noqa: E241

            # (self.right_arm_transport,  2,  "right_shoulder_pitch_joint"    ),  # noqa: E241
            # (self.right_arm_transport,  4,  "right_shoulder_roll_joint"     ),  # noqa: E241
            # (self.right_arm_transport,  6,  "right_shoulder_yaw_joint"      ),  # noqa: E241
            # (self.right_arm_transport,  8,  "right_elbow_pitch_joint"       ),  # noqa: E241
            # (self.right_arm_transport,  10, "right_wrist_yaw_joint"         ),  # noqa: E241

            (self.left_leg_transport,   1,  "left_hip_roll_joint"           ),  # noqa: E241
            (self.left_leg_transport,   3,  "left_hip_yaw_joint"            ),  # noqa: E241
            (self.left_leg_transport,   5,  "left_hip_pitch_joint"          ),  # noqa: E241
            (self.left_leg_transport,   7,  "left_knee_pitch_joint"         ),  # noqa: E241
            (self.left_leg_transport,   11, "left_ankle_pitch_joint"        ),  # noqa: E241
            (self.left_leg_transport,   13, "left_ankle_roll_joint"         ),  # noqa: E241

            (self.right_leg_transport,  2,  "right_hip_roll_joint"          ),  # noqa: E241
            (self.right_leg_transport,  4,  "right_hip_yaw_joint"           ),  # noqa: E241
            (self.right_leg_transport,  6,  "right_hip_pitch_joint"         ),  # noqa: E241
            (self.right_leg_transport,  8,  "right_knee_pitch_joint"        ),  # noqa: E241
            (self.right_leg_transport,  12, "right_ankle_pitch_joint"       ),  # noqa: E241
            (self.right_leg_transport,  14, "right_ankle_roll_joint"        ),  # noqa: E241
        ]

        self.imu = None
        self.command_controller = None

        self.state = State.IDLE
        self.next_state = State.IDLE

        self.rl_init_positions = np.array([
            0.0, 0.0, -0.2,
            0.4,
            -0.3, 0.0,
            0.0, 0.0, -0.2,
            0.4,
            -0.3, 0.0
        ], dtype=np.float32)

        self.joint_axis_directions = np.array([
            -1, 1, -1,
            -1,
            -1, 1,
            -1, 1, 1,
            1,
            1, 1
        ], dtype=np.float32)

        self.position_offsets = np.array([
            0.0, 0.0, 0.0,
            0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0,
            0.0, 0.0
        ], dtype=np.float32)

        self.n_lowlevel_states = 4 + 3 + 12 + 12 + 1 + 3
        self.lowlevel_states = np.zeros(self.n_lowlevel_states, dtype=np.float32)

        self.joint_velocity_target = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_position_target = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_position_measured = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_velocity_measured = np.zeros(len(self.joints), dtype=np.float32)

        # used for RL initialization controller
        self.starting_positions = np.zeros_like(self.joint_position_target, dtype=np.float32)
        self.init_ready = False
        self.init_failed = False
        self._init_ramp = None
        self._init_feedback_complete = False
        self._init_settling_at = None
        self._init_stable_since = None
        self._init_stable_samples = 0
        self._init_timeouts = np.zeros(len(self.joints), dtype=int)
        self._init_previous_positions = self.starting_positions.copy()

        self._calibration_loaded = False
        config_path = Path(__file__).resolve().parents[2] / "calibration.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = OmegaConf.load(f)
            position_offsets = np.array(config.get("position_offsets", None), dtype=np.float32)
            if position_offsets.shape != (len(self.joints),) or not np.all(np.isfinite(position_offsets)):
                raise ValueError(f"Invalid calibration data in {config_path}: expected {len(self.joints)} finite offsets")
            self.position_offsets[:] = position_offsets
            self._calibration_loaded = True
        else:
            print(f"Warning: {config_path} not found; using zero joint offsets.")

        # Parse calibration before starting background threads.
        if start_peripherals:
            self.imu = SerialImu(baudrate=Baudrate.BAUD_460800)
            self.imu.run_forever()
            self.command_controller = Se2Gamepad()
            self.command_controller.run()

    def check_initialization_configuration(self) -> None:
        # Calibration tools may construct Humanoid without a saved calibration;
        # locomotion must not enable motion with the fallback zero offsets.
        if not self._calibration_loaded or not np.all(np.isfinite(self.position_offsets)):
            raise JointConfigurationError(
                "Valid calibration.yaml is required; run scripts/calibrate_joints.py first."
            )

    def check_gear_ratios(self, timeout: float = 0.1) -> None:
        errors = []
        for bus, device_id, joint_name in self.joints:
            ratio = bus.read_gear_ratio(device_id, timeout=timeout)
            if ratio is None or not math.isclose(ratio, self.JOINT_GEAR_RATIO, rel_tol=1e-6):
                errors.append(
                    f"{joint_name} ({bus.channel}, ID {device_id}): "
                    f"gear_ratio={ratio}, expected {self.JOINT_GEAR_RATIO}"
                )
        if errors:
            raise JointConfigurationError(
                "Invalid actuator gear ratios:\n" + "\n".join(errors)
                + "\nCheck with: python3 scripts/motor/configure_parameter.py --all"
                + "\nCorrect the actuator parameters, then recalibrate joints before enabling motion."
            )

    def enter_damping(self):
        # Check all actuators before writing any gains or enabling any motor.
        self.check_gear_ratios()
        self.joint_kp = np.zeros((len(self.joints),), dtype=np.float32)
        self.joint_kd = np.zeros((len(self.joints),), dtype=np.float32)
        self.torque_limit = np.zeros((len(self.joints),), dtype=np.float32)

        self.joint_kp[:] = 20
        self.joint_kd[:] = 2
        self.torque_limit[:] = 4

        for i, entry in enumerate(self.joints):
            bus, device_id, joint_name = entry

            print(f"Initializing joint {joint_name}:")
            print(f"  kp: {self.joint_kp[i]}, kd: {self.joint_kd[i]}, torque limit: {self.torque_limit[i]}")

            try:
                # Set the joint mode to idle
                bus.set_mode(device_id, recoil.Mode.IDLE)
                time.sleep(0.001)
                bus.write_position_kp(device_id, self.joint_kp[i])
                time.sleep(0.001)
                bus.write_position_kd(device_id, self.joint_kd[i])
                time.sleep(0.001)
                bus.write_torque_limit(device_id, self.torque_limit[i])
                time.sleep(0.001)
                self._verify_initialization_parameters(i)
                bus.feed(device_id)
                bus.set_mode(device_id, recoil.Mode.DAMPING)
                mode = self._read_init_parameter(i, recoil.Parameter.MODE, timeout=0.02, integer=True)
                if mode != recoil.Mode.DAMPING:
                    raise RuntimeError(f"DAMPING mode rejected (mode=0x{mode:02x})")
            except Exception as error:
                raise RuntimeError(
                    f"{bus.channel} ID {device_id} ({joint_name}): {error}"
                ) from error

        print("Motors enabled")

    def _read_init_parameter(self, joint_id, parameter, timeout, integer=False):
        bus, device_id, _ = self.joints[joint_id]
        read = bus._read_parameter_u32 if integer else bus._read_parameter_f32
        value = read(device_id, parameter, timeout=timeout)
        if value is None or not math.isfinite(value):
            raise TimeoutError(f"{self._joint_label(joint_id)}: no valid readback for parameter 0x{parameter:03x}")
        return value

    def _verify_initialization_parameters(self, joint_id):
        parameter = recoil.Parameter
        fields = (
            ("kp", parameter.POSITION_CONTROLLER_POSITION_KP, self.joint_kp[joint_id]),
            ("kd", parameter.POSITION_CONTROLLER_VELOCITY_KP, self.joint_kd[joint_id]),
            ("torque_limit", parameter.POSITION_CONTROLLER_TORQUE_LIMIT, self.torque_limit[joint_id]),
            ("ki", parameter.POSITION_CONTROLLER_POSITION_KI, None),
            ("torque_filter_alpha", parameter.POSITION_CONTROLLER_TORQUE_FILTER_ALPHA, None),
        )
        values = []
        for name, address, expected in fields:
            value = self._read_init_parameter(joint_id, address, timeout=0.1)
            if expected is not None and not math.isclose(value, expected, rel_tol=1e-5):
                raise RuntimeError(f"{self._joint_label(joint_id)}: {name}={value:g}, expected {expected:g}")
            values.append(f"{name}={value:g}")
        print("  Drive readback: " + ", ".join(values))

    def enter_initialization_position_mode(self, group):
        for joint_id in group:
            bus, device_id, _ = self.joints[joint_id]
            bus.feed(device_id)
            bus.set_mode(device_id, recoil.Mode.POSITION)
            mode = self._read_init_parameter(joint_id, recoil.Parameter.MODE, timeout=0.02, integer=True)
            if mode != recoil.Mode.POSITION:
                raise RuntimeError(f"{self._joint_label(joint_id)}: POSITION mode rejected (mode=0x{mode:02x})")
        self._init_diagnostic_at = time.monotonic()
        self._init_diagnostic_index = 0

    def log_initialization_feedback(self, group, commands, positions, velocities):
        now = time.monotonic()
        if now < self._init_diagnostic_at:
            return
        joint_id = group[self._init_diagnostic_index]
        self._init_diagnostic_index = (self._init_diagnostic_index + 1) % len(group)
        self._init_diagnostic_at = now + 0.1
        parameter = recoil.Parameter
        try:
            mode = self._read_init_parameter(joint_id, parameter.MODE, timeout=0.005, integer=True)
            if mode != recoil.Mode.POSITION:
                raise RuntimeError(f"{self._joint_label(joint_id)}: lost POSITION (mode=0x{mode:02x})")
            fault = self._read_init_parameter(joint_id, parameter.ERROR, timeout=0.005, integer=True)
            torque = self._read_init_parameter(joint_id, parameter.POSITION_CONTROLLER_TORQUE_SETPOINT, timeout=0.005)
            measured = self._read_init_parameter(joint_id, parameter.POSITION_CONTROLLER_TORQUE_MEASURED, timeout=0.005)
        except (TimeoutError, IndexError):
            # SDO replies do not identify the register. End this sample on the
            # first failed read, so a late reply cannot be used for its next field.
            print(f"[init] {self._joint_label(joint_id)}: drive diagnostics unavailable")
            return
        direction = self.joint_axis_directions[joint_id]
        print(f"[init] {self._joint_label(joint_id)} "
              f"cmd={commands[joint_id]:+.3f} pos={positions[joint_id]:+.3f} "
              f"track_err={commands[joint_id] - positions[joint_id]:+.3f} "
              f"goal_err={self.rl_init_positions[joint_id] - positions[joint_id]:+.3f} "
              f"vel={velocities[joint_id]:+.3f} rad/s "
              f"torque_set={direction * torque:+.2f} torque_est={direction * measured:+.2f} N*m "
              f"mode=0x{mode:02x} fault=0x{fault:04x}")

    def stop(self):
        if self.imu is not None:
            self.imu.stop()
        if self.command_controller is not None:
            self.command_controller.stop()

        for entry in self.joints:
            bus, device_id, _ = entry
            bus.set_mode(device_id, recoil.Mode.DAMPING)

        print("Entered damping mode. Press Ctrl+C again to exit.\n")

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Exiting damping mode.")

        for entry in self.joints:
            bus, device_id, _ = entry
            bus.set_mode(device_id, recoil.Mode.IDLE)

        # self.left_arm_transport.stop()
        # self.right_arm_transport.stop()
        self.left_leg_transport.stop()
        self.right_leg_transport.stop()

    def get_observations(self) -> np.ndarray:
        imu_quaternion = self.lowlevel_states[0:4]
        imu_angular_velocity = self.lowlevel_states[4:7]
        joint_positions = self.lowlevel_states[7:19]
        joint_velocities = self.lowlevel_states[19:31]
        mode = self.lowlevel_states[31:32]
        velocity_commands = self.lowlevel_states[32:35]

        imu_quaternion[:] = self.imu.quaternion[:]

        # IMU returns angular velocity in deg/s, we need rad/s
        imu_angular_velocity[:] = np.deg2rad(self.imu.angular_velocity[:])

        joint_positions[:] = self.joint_position_measured[:]
        joint_velocities[:] = self.joint_velocity_measured[:]

        mode[0] = self.command_controller.commands["mode_switch"]
        velocity_commands[0] = self.command_controller.commands["velocity_x"]
        velocity_commands[1] = self.command_controller.commands["velocity_y"]
        velocity_commands[2] = self.command_controller.commands["velocity_yaw"]

        self.next_state = self.command_controller.commands["mode_switch"]

        return self.lowlevel_states

    def update_joint_group(self, joint_id_l, joint_id_r):
        # adjust direction and offset of target values
        position_target_l = (self.joint_position_target[joint_id_l] + self.position_offsets[joint_id_l]) * self.joint_axis_directions[joint_id_l]
        position_target_r = (self.joint_position_target[joint_id_r] + self.position_offsets[joint_id_r]) * self.joint_axis_directions[joint_id_r]

        self.joints[joint_id_l][0].transmit_pdo_2(self.joints[joint_id_l][1], position_target=position_target_l, velocity_target=0.0)
        self.joints[joint_id_r][0].transmit_pdo_2(self.joints[joint_id_r][1], position_target=position_target_r, velocity_target=0.0)

        received = []
        for joint_id in (joint_id_l, joint_id_r):
            bus, device_id, _ = self.joints[joint_id]
            if self.state == State.RL_INIT:
                position, velocity = receive_initialization_feedback(bus, device_id)
            else:
                position, velocity = bus.receive_pdo_2(device_id)

            received.append(position is not None and velocity is not None)
            # adjust direction and offset of measured values
            if position is not None:
                self.joint_position_measured[joint_id] = position * self.joint_axis_directions[joint_id] - self.position_offsets[joint_id]
            if velocity is not None:
                self.joint_velocity_measured[joint_id] = velocity * self.joint_axis_directions[joint_id]
        return received

    def update_joints(self):
        # PDO traffic alone does not feed the firmware watchdog.
        if self.state in (State.RL_INIT, State.RL_RUNNING):
            for bus, device_id, _ in self.joints:
                bus.feed(device_id)
        received = np.zeros(len(self.joints), dtype=bool)
        for left in range(6):
            received[left], received[left + 6] = self.update_joint_group(left, left + 6)
        return received

    def _read_joint_positions(self):
        initial_positions = np.empty(len(self.joints), dtype=np.float32)
        missing_joints = []

        # Read every joint while the motors are still in damping mode.  Never
        # enter position mode with the zero-filled startup values above.
        for joint_id, (bus, device_id, joint_name) in enumerate(self.joints):
            position_measured = bus.read_position_measured(
                device_id,
                timeout=0.1,
            )
            if position_measured is None or not np.isfinite(position_measured):
                missing_joints.append(
                    f"{joint_name} ({bus.channel}, device ID {device_id})"
                )
                continue

            initial_positions[joint_id] = (
                position_measured * self.joint_axis_directions[joint_id]
                - self.position_offsets[joint_id]
            )

        if missing_joints:
            raise RuntimeError(
                "Cannot initialize: no valid position feedback from "
                + ", ".join(missing_joints)
            )
        return initial_positions

    def reset(self):
        initial_positions = self._read_joint_positions()
        self.joint_position_measured[:] = initial_positions
        self.joint_velocity_measured.fill(0.0)
        self.joint_position_target[:] = initial_positions
        self.starting_positions[:] = initial_positions

        # Preload the measured positions as targets before position mode can be
        # selected.  In damping mode these messages cannot produce motion.
        self.update_joints()

        obs = self.get_observations()
        return obs

    def _joint_label(self, joint_id):
        bus, device_id, name = self.joints[joint_id]
        return f"{name} ({bus.channel}, ID {device_id})"

    def _start_initialization(self):
        self.init_ready = False
        self.init_failed = False
        self._init_settling_at = None
        self._init_stable_since = None
        self._init_stable_samples = 0
        self._init_timeouts.fill(0)
        self.check_initialization_configuration()
        self.check_gear_ratios()

        samples = []
        for _ in range(self.INIT_STABLE_SAMPLES):
            samples.append(self._read_joint_positions())
            time.sleep(0.02)
        for joint_id, spread in enumerate(np.ptp(samples, axis=0)):
            if spread > self.INIT_MAX_SAMPLE_SPREAD:
                raise RuntimeError(f"{self._joint_label(joint_id)}: unstable start position, spread={spread:.3f} rad")

        self.starting_positions = samples[-1].copy()
        travel = self.rl_init_positions - self.starting_positions
        print("Joint initialization plan:                  current     target     travel (rad)")
        for joint_id in range(len(self.joints)):
            print(f"{self._joint_label(joint_id):42s} {self.starting_positions[joint_id]:+9.3f} "
                  f"{self.rl_init_positions[joint_id]:+10.3f} {travel[joint_id]:+10.3f}")
        for joint_id in np.flatnonzero(np.abs(travel) > self.INIT_MAX_TRAVEL):
            raise RuntimeError(f"{self._joint_label(joint_id)}: travel={abs(travel[joint_id]):.3f} rad "
                               f"exceeds {self.INIT_MAX_TRAVEL:.3f} rad")

        self.joint_position_target[:] = self.starting_positions
        self.joint_position_measured[:] = self.starting_positions
        self.joint_velocity_measured.fill(0.0)
        # Preload all targets while still in DAMPING, before any POSITION command.
        received = self.update_joints()
        for joint_id in range(len(self.joints)):
            if not received[joint_id]:
                raise RuntimeError(f"{self._joint_label(joint_id)}: no PDO feedback while preloading targets")
            if (abs(self.joint_position_measured[joint_id] - self.starting_positions[joint_id]) > self.INIT_MAX_SAMPLE_SPREAD
                    or abs(self.joint_velocity_measured[joint_id]) > self.INIT_SETTLING_VELOCITY):
                raise RuntimeError(f"{self._joint_label(joint_id)}: joint moved during preflight")
        self._init_previous_positions = self.joint_position_measured.copy()
        self._init_feedback_complete = True
        # Honor a cancel pressed while the preflight reads were in progress.
        if self.command_controller.commands["mode_switch"] == State.IDLE:
            self.state = State.IDLE
            print("Initialization cancelled before entering position mode.")
            return
        self.enter_initialization_position_mode(range(len(self.joints)))
        self._init_ramp = InitializationRamp(
            self.starting_positions, self.rl_init_positions,
            [self._joint_label(joint_id) for joint_id in range(len(self.joints))],
            InitializationRamp.RESUME_ERROR, time.monotonic(),
        )
        print(f"Initializing at up to {self._init_ramp.MAX_SPEED:.2f} rad/s reference speed, "
              f"with {self._init_ramp.TIMEOUT:g} s trajectory timeout.")

    def _fail_initialization(self, reason):
        self.state = State.RL_INIT
        self.init_failed = True
        self.init_ready = False
        print(f"Initialization FAILED: {reason}")
        # One failed CAN send must not prevent attempts to stop the other joints.
        for bus, device_id, _ in self.joints:
            try:
                bus.set_mode(device_id, recoil.Mode.DAMPING)
            except Exception as error:
                print(f"DAMPING send failed for {bus.channel} ID {device_id}: {error}")
        print("DAMPING requested for all joints. RL running is blocked; select idle before retrying initialization.")

    def _check_initialization_feedback(self, received):
        self._init_timeouts[received] = 0
        self._init_timeouts[~received] += 1
        errors = self.joint_position_target - self.joint_position_measured
        for joint_id in range(len(self.joints)):
            label = self._joint_label(joint_id)
            if self._init_timeouts[joint_id] >= self.INIT_CAN_TIMEOUT_LIMIT:
                raise RuntimeError(f"{label}: {self.INIT_CAN_TIMEOUT_LIMIT} consecutive missing PDO replies")
            if not received[joint_id]:
                continue
            jump = abs(self.joint_position_measured[joint_id] - self._init_previous_positions[joint_id])
            velocity = self.joint_velocity_measured[joint_id]
            if jump > self.INIT_MAX_POSITION_STEP:
                raise RuntimeError(f"{label}: position jump={jump:.3f} rad")
            if abs(velocity) > self.INIT_MAX_VELOCITY:
                raise RuntimeError(f"{label}: velocity={velocity:.3f} rad/s")
            if abs(errors[joint_id]) > self.INIT_MAX_TRACKING_ERROR:
                raise RuntimeError(f"{label}: tracking error={errors[joint_id]:+.3f} rad")
            self._init_previous_positions[joint_id] = self.joint_position_measured[joint_id]

        if self._init_ramp.progress < 1.0:
            return
        now = time.monotonic()
        if self._init_settling_at is None:
            self._init_settling_at = now
        # RL readiness requires fresh, slow feedback after the ramp.
        # Position error is reported here and guarded by the tracking limit above.
        settled_joints = received & (np.abs(self.joint_velocity_measured) <= self.INIT_SETTLING_VELOCITY)
        settled = settled_joints.all()
        self._init_stable_samples = self._init_stable_samples + 1 if settled else 0
        if not settled:
            self._init_stable_since = None
        elif self._init_stable_since is None:
            self._init_stable_since = now
        if (self._initialization_is_stable(now)
                and (self.init_ready or now - self._init_settling_at <= self.INIT_SETTLING_TIMEOUT)):
            if not self.init_ready:
                print(f"Initialization ready: max residual position error={np.max(np.abs(errors)):.3f} rad. "
                      "Holding target positions; use the RL running command when ready.")
            self.init_ready = True
        elif not self.init_ready and now - self._init_settling_at >= self.INIT_SETTLING_TIMEOUT:
            details = "; ".join(
                f"{self._joint_label(joint_id)}: "
                + (f"velocity={self.joint_velocity_measured[joint_id]:+.3f} rad/s"
                   if received[joint_id] else "no PDO reply")
                for joint_id in np.flatnonzero(~settled_joints)
            )
            raise RuntimeError(f"Settling timeout: need fresh feedback for {self.INIT_STABLE_TIME:.1f} s "
                               f"at or below {self.INIT_SETTLING_VELOCITY:.3f} rad/s. {details}")

    def _initialization_is_stable(self, now):
        return (self._init_stable_samples >= self.INIT_STABLE_SAMPLES
                and self._init_stable_since is not None
                and now - self._init_stable_since >= self.INIT_STABLE_TIME)

    def step(self, actions: np.ndarray):
        """
        actions: np.ndarray of shape (n_joints, )
        """
        match (self.state):
            case State.IDLE:
                self.joint_position_target[:] = self.joint_position_measured[:]

                if self.next_state == State.RL_INIT:
                    print("Switching to RL initialization mode")
                    self.state = self.next_state
                    try:
                        self._start_initialization()
                    except Exception as error:
                        self._fail_initialization(error)

            case State.RL_INIT:
                if self.next_state == State.IDLE:
                    print("Switching to idle mode")
                    self.state = State.IDLE
                    self.init_ready = False
                    self.init_failed = False

                    try:
                        for bus, device_id, _ in self.joints:
                            bus.set_mode(device_id, recoil.Mode.DAMPING)
                        self.joint_position_target[:] = self.joint_position_measured[:]
                        self.update_joints()
                    except Exception as error:
                        self._fail_initialization(error)
                    return self.get_observations()

            case State.RL_RUNNING:
                for i in range(len(self.joints)):
                    self.joint_position_target[i] = actions[i]

                if self.next_state == State.IDLE:
                    print("Switching to idle mode")
                    self.state = self.next_state

                    for entry in self.joints:
                        bus, device_id, _ = entry
                        bus.set_mode(device_id, recoil.Mode.DAMPING)

        if self.state == State.RL_INIT:
            if not self.init_failed:
                try:
                    if self._init_ramp.progress < 1.0:
                        self._init_ramp.advance(
                            self.joint_position_measured, self._init_feedback_complete,
                            time.monotonic(),
                        )
                        self.joint_position_target[:] = self._init_ramp.command
                    received = self.update_joints()
                    self._init_feedback_complete = received.all()
                    self._check_initialization_feedback(received)
                    if self._init_feedback_complete:
                        self.log_initialization_feedback(
                            range(len(self.joints)), self.joint_position_target,
                            self.joint_position_measured, self.joint_velocity_measured,
                        )
                    now = time.monotonic()
                    if (self.init_ready and self._initialization_is_stable(now)
                            and self.command_controller.commands["mode_switch"] == State.RL_RUNNING):
                        print("Switching to RL running mode")
                        self.state = State.RL_RUNNING
                except Exception as error:
                    self._fail_initialization(error)
        else:
            self.update_joints()

        obs = self.get_observations()

        return obs

    def check_connection(self):
        for entry in self.joints:
            bus, device_id, joint_name = entry
            print(f"Pinging {joint_name} ... ", end="\t")
            result = bus.ping(device_id)
            if result:
                print("OK")
            else:
                print("ERROR")
            time.sleep(0.1)
