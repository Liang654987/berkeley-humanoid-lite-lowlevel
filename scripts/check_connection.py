# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from berkeley_humanoid_lite_lowlevel.robot import Humanoid


robot = Humanoid()

robot.check_connection()

robot.stop()
