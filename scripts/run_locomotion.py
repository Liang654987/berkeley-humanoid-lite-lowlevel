# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import sys
from pathlib import Path

# Allow running this file directly from the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc.udp import UDP
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot import Humanoid
from berkeley_humanoid_lite_lowlevel.robot.humanoid import JointConfigurationError
from berkeley_humanoid_lite_lowlevel.recoil import Mode
from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite_lowlevel.policy.config import Cfg


def cleanup_startup(robot, damping: bool) -> None:
    if robot is None:
        return
    if damping:
        for bus, device_id, joint_name in robot.joints:
            try:
                bus.set_mode(device_id, Mode.DAMPING)
            except Exception as error:
                print(f"Cannot damp {joint_name}: {error}", file=sys.stderr)
    for resource in (
        robot.imu, robot.command_controller,
        robot.left_leg_transport, robot.right_leg_transport,
    ):
        if resource is not None:
            try:
                resource.stop()
            except Exception as error:
                print(f"Startup cleanup failed: {error}", file=sys.stderr)


# Load configuration
cfg = Cfg.from_arguments()

print(f"Policy frequency: {1 / cfg.policy_dt} Hz")

udp = UDP(("0.0.0.0", 11000), ("192.168.3.87", 11000))


# Initialize and start policy controller
controller = RlController(cfg)
controller.load_policy()

rate = RateLimiter(1 / cfg.policy_dt)

robot = None
motor_setup_started = False
try:
    robot = Humanoid()
    robot.check_initialization_configuration()
    motor_setup_started = True
    robot.enter_damping()
    obs = robot.reset()
except KeyboardInterrupt:
    cleanup_startup(robot, damping=motor_setup_started)
    raise SystemExit(130)
except Exception as error:
    print(f"Robot initialization failed: {error}", file=sys.stderr)
    # Gear-ratio rejection happens before enter_damping writes any motor modes.
    cleanup_startup(
        robot, damping=motor_setup_started and not isinstance(error, JointConfigurationError),
    )
    raise SystemExit(1)

try:
    while True:
        actions = controller.update(obs)
        obs = robot.step(actions)
        udp.send_numpy(obs)

        rate.sleep()

except KeyboardInterrupt:
    robot.stop()

print("Stopped.")
