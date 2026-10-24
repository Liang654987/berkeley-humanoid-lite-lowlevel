# Berkeley Humanoid Lite Low-level Control

This submodule repository contains the low-level control code for the Berkeley Humanoid Lite robot.


## Installation

```bash
sudo apt install net-tools can-utils
```

```bash
pip install -r requirements.txt
```

## Getting Started

### Bring up the CAN interface

The low-level computer connects to the joints via CAN.

Run this script to initialize the CAN interface:

```bash
sudo ./scripts/start_can_transports.sh
```

### Verify CAN connection

A Python script is provided to verify the CAN connection to all the joints of the robot:

```bash
python3 ./berkeley_humanoid_lite_lowlevel/robot/check_connection.py
```

### Launch the joystick receiver

The low-level computer also receives commands from the joystick.

To broadcast the joystick commands to other running nodes, run the following command:

```bash
python ./berkeley_humanoid_lite_lowlevel/policy/udp_joystick.py
```

The Xiaoji controller event mapping is applied by
`berkeley_humanoid_lite_lowlevel/policy/gamepad.py`, including the `ABS_*`
axis remapping. The `evdev-keycode-remapper` project uses Linux hwdb
`KEYBOARD_KEY_*` entries and is intended for key events; it cannot remap these
absolute joystick axes. The runtime mapping is therefore required for the
controller's axes and also keeps the mapping in the project configuration.

### Joint Calibration

Because the joint actuators only have single encoder on the motor shaft, we need to calibrate the zero position of the joints after each power cycle.

Run the following command to start the calibration:

```bash
python3 ./scripts/calibrate_joints.py
```

After the script is launched and running, manually move the robot joints to the mechanical position limits. After all the joints are moved, press `q` or `Enter` to stop sampling and save the calibration. Pressing `Ctrl+C` cancels without overwriting the calibration file.

Each actuator read has a timeout. If an actuator does not respond, the script reports its joint name, CAN channel, and device ID instead of waiting indefinitely. Check the reported CAN interface and actuator before running the calibration again.

The calibration data will be saved in the `./calibration.yaml` file.



### Run main controller

The main controller is implemented in C.

To run the controller, run the following command:

```bash
make run
```

Press `LB` + `A` to enter RL init mode. Then, press `RB` + `A` to enter RL running mode.

At any time, press `B` or the thumb buttons to exit RL mode. The joints will enter passive damping mode.

Press `Ctrl` + `C` to terminate the controller. Upon first termination, the joints will enter passive damping mode. Press `Ctrl` + `C` again to completely stop the controller, which joints will return to unpowered idle state.
