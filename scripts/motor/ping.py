# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import berkeley_humanoid_lite_lowlevel.recoil as recoil


args = recoil.util.get_args()
bus = recoil.Bus(channel=args.channel, bitrate=1000000)

device_id = args.id

status = bus.ping(device_id)

if status:
    print("Motor is online")
else:
    print("Motor is offline")

bus.stop()
