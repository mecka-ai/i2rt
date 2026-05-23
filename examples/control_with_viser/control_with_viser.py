"""Start the YAM Viser UI on the CM5."""

# ruff: noqa: I001

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.nexus_camera import hand_eye_calibration_path
from i2rt.utils.viser_control_interface import ViserControlInterface


PORT = 8080
CHANNEL = "can0"
CAMERA_MOUNT_FRAME = "geom_4_top"
CAMERA_CALIBRATIONS = {
    "left": str(hand_eye_calibration_path("left")),
    "right": str(hand_eye_calibration_path("right")),
}


robot = get_yam_robot(
    channel=CHANNEL,
    arm_type=ArmType.YAM,
    gripper_type=GripperType.NO_GRIPPER,
    sim=False,
    clip_motor_torque=2.0,
)

ViserControlInterface.from_robot(
    robot,
    ee_site="grasp_site",
    port=PORT,
    camera_calibrations=CAMERA_CALIBRATIONS,
    camera_mount_frame=CAMERA_MOUNT_FRAME,
).run()
