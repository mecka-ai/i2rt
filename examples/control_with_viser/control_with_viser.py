"""Start the YAM Viser UI as a client of robot_control.server."""

# ruff: noqa: I001

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from i2rt.utils.nexus_camera import hand_eye_calibration_path
from i2rt.utils.viser_control_interface import ViserControlInterface
from robot_control import RemoteCameraFeed, RobotClient


PORT = 8080
ROBOT_CONTROL_URL = "ws://127.0.0.1:8765"
CAMERA_MOUNT_FRAME = "geom_4_top"
CAMERA_CALIBRATIONS = {
    "left": str(hand_eye_calibration_path("left")),
    "right": str(hand_eye_calibration_path("right")),
}


robot = RobotClient(ROBOT_CONTROL_URL)
camera_feed = RemoteCameraFeed(ROBOT_CONTROL_URL)

ViserControlInterface.from_robot(
    robot,
    ee_site="grasp_site",
    port=PORT,
    camera_calibrations=CAMERA_CALIBRATIONS,
    camera_mount_frame=CAMERA_MOUNT_FRAME,
    camera_feed=camera_feed,
).run()
