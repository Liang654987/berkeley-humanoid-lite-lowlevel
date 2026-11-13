# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Gamepad Controller Module for Berkeley Humanoid Lite

This module implements UDP-based controllers for the Berkeley Humanoid Lite robot,
supporting both gamepad and keyboard input devices. It handles command broadcasting
over UDP for robot control modes and movement velocities.
"""

import struct
import sys
import threading
from typing import Dict

if sys.platform == "linux":
    import fcntl

from inputs import UnpluggedError, get_gamepad


class XInputEntry:
    """
    Constants for gamepad button and axis mappings.

    This class defines the standard mapping for various gamepad controls,
    including analog sticks, triggers, d-pad, and buttons.
    """
    AXIS_X_L = "ABS_X"
    AXIS_Y_L = "ABS_Y"
    AXIS_TRIGGER_L = "ABS_Z"
    AXIS_X_R = "ABS_RX"
    AXIS_Y_R = "ABS_RY"
    AXIS_TRIGGER_R = "ABS_RZ"

    BTN_HAT_X = "ABS_HAT0X"
    BTN_HAT_Y = "ABS_HAT0Y"

    BTN_A = "BTN_SOUTH"
    BTN_B = "BTN_EAST"
    BTN_X = "BTN_NORTH"
    BTN_Y = "BTN_WEST"
    BTN_BUMPER_L = "BTN_TL"
    BTN_BUMPER_R = "BTN_TR"
    BTN_THUMB_L = "BTN_THUMBL"
    BTN_THUMB_R = "BTN_THUMBR"
    BTN_BACK = "BTN_SELECT"
    BTN_START = "BTN_START"


# Raw event names reported by the Xiaoji controller mapped to the standard
# XInput names consumed by Se2Gamepad.
XIAOJI_EVENT_REMAP = {
    ##四个圆键
    "BTN_EAST": "BTN_SOUTH",
    "BTN_C": "BTN_EAST",
    "BTN_SOUTH": "BTN_WEST",
    "BTN_NORTH": "BTN_NORTH",
    ##右摇杆
    "ABS_RZ": "ABS_RY",
    "ABS_Z": "ABS_RX",
    "BTN_START": "BTN_THUMBR",
    ##左摇杆
    "ABS_Y": "ABS_Y",
    "ABS_X": "ABS_X",
    "BTN_SELECT": "BTN_THUMBL",
    ##十字键
    "ABS_HAT0X": "ABS_HAT0X",
    "ABS_HAT0Y": "ABS_HAT0Y",
    ##左肩键
    "BTN_WEST": "BTN_TL",
    "ABS_RX": "ABS_Z",
    "BTN_TL": "BTN_TL2",
    ##右肩键
    "BTN_Z": "BTN_TR",
    "ABS_RY": "ABS_RZ",
    "BTN_TR": "BTN_TR2",
    ##其他
    "BTN_TL2": "BTN_SELECT",
    "BTN_TR2": "BTN_START",
    "BTN_THUMBL": "BTN_C",
    "BTN_MODE": "BTN_MODE",
}


# Linux EV_ABS codes for the three raw stick axes used by this mapping.
LINUX_STICK_AXIS_CODES = {"ABS_X": 0, "ABS_Y": 1, "ABS_Z": 2}
LINUX_ABS_INFO = struct.Struct("=6i")


class Se2Gamepad:
    def __init__(self,
                 stick_sensitivity: float = 1.0,
                 dead_zone: float = 0.01,
                 ) -> None:
        self.stick_sensitivity = stick_sensitivity
        self.dead_zone = dead_zone

        self._stopped = threading.Event()
        self._run_forever_thread = None
        self._last_mode_switch = 0

        self.reset()

        self.commands = {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_yaw": 0.0,
            "mode_switch": 0,
        }

    def reset(self) -> None:
        self._states = {key: 0 for key in XInputEntry.__dict__.values()}
        self._last_mode_switch = 0
        self._stick_axis_limits: dict[str, tuple[int, int, int]] = {}

    def stop(self) -> None:
        print("Gamepad stopping...")
        self._stopped.set()
        # self._run_forever_thread.join()

    def run(self) -> None:
        self._run_forever_thread = threading.Thread(target=self.run_forever)
        self._run_forever_thread.start()

    def run_forever(self) -> None:
        while not self._stopped.is_set():
            self.advance()

    def advance(self) -> None:
        try:
            events = get_gamepad()
        except UnpluggedError:
            return

        # update all events from the joystick
        for event in events:
            mapped_code = XIAOJI_EVENT_REMAP.get(event.code, event.code)
            state = event.state
            if mapped_code in (XInputEntry.AXIS_Y_L, XInputEntry.AXIS_X_R, XInputEntry.AXIS_X_L):
                state = self._normalize_stick_axis(event)
            self._states[mapped_code] = state

        self._update_command_buffer()

    def _normalize_stick_axis(self, event) -> float:
        if sys.platform != "linux":
            return event.state / 32768.0

        if event.code not in self._stick_axis_limits:
            # EVIOCGABS uses the original axis code, before the Xiaoji remap.
            axis = LINUX_STICK_AXIS_CODES[event.code]
            request = 0x80000000 | (LINUX_ABS_INFO.size << 16) | (ord("E") << 8) | (0x40 + axis)
            try:
                with open(event.device.get_char_device_path(), "rb") as device:
                    info = fcntl.ioctl(device, request, bytes(LINUX_ABS_INFO.size))
                _, minimum, maximum, _, flat, _ = LINUX_ABS_INFO.unpack(info)
            except OSError:
                return event.state / 32768.0
            self._stick_axis_limits[event.code] = minimum, maximum, flat

        minimum, maximum, flat = self._stick_axis_limits[event.code]
        center = (minimum + maximum + 1) // 2
        offset = event.state - center
        if abs(offset) <= flat:
            return 0.0
        half_range = center - minimum if offset < 0 else maximum - center
        magnitude = min(1.0, (abs(offset) - flat) / max(half_range - flat, 1))
        return -magnitude if offset < 0 else magnitude

    def _update_command_buffer(self) -> Dict[str, float]:
        self.commands["velocity_x"] = -self._states[XInputEntry.AXIS_Y_L]
        self.commands["velocity_y"] = -self._states[XInputEntry.AXIS_X_R]
        self.commands["velocity_yaw"] = -self._states[XInputEntry.AXIS_X_L]

        mode_switch = 0

        # Enter RL control mode (A + Right Bumper)
        if self._states.get(XInputEntry.BTN_A) and self._states.get(XInputEntry.BTN_BUMPER_R):
            mode_switch = 3

        # Enter init mode (A + Left Bumper)
        if self._states.get(XInputEntry.BTN_A) and self._states.get(XInputEntry.BTN_BUMPER_L):
            mode_switch = 2

        # Enter idle mode (B or Left/Right Thumbstick)
        if self._states.get(XInputEntry.BTN_X) or self._states.get(XInputEntry.BTN_THUMB_L) or self._states.get(XInputEntry.BTN_THUMB_R):
            mode_switch = 1

        self.commands["mode_switch"] = mode_switch
        if mode_switch != self._last_mode_switch:
            mode_names = {
                0: "no mode switch",
                1: "idle mode",
                2: "RL init mode",
                3: "RL control mode",
            }
            print(f"Gamepad: entered {mode_names[mode_switch]}.")
            self._last_mode_switch = mode_switch


if __name__ == "__main__":
    command_controller = Se2Gamepad()
    command_controller.run()

    try:
        while True:
            print(f"""{command_controller.commands.get("velocity_x"):.2f}, {command_controller.commands.get("velocity_y"):.2f}, {command_controller.commands.get("velocity_yaw"):.2f}""")
            pass
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    command_controller.stop()
