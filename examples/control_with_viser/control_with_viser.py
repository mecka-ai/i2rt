"""Viser control interface for i2rt robots.

Opens a browser-based 3-D viewer.  The robot stays in read-only mode until
the user confirms visual alignment and clicks "Enable".  Three control modes
are then available: mirror (VIS), IK drag, and per-joint sliders.

Usage:
    python examples/control_with_viser/control_with_viser.py --sim
    python examples/control_with_viser/control_with_viser.py --arm big_yam --gripper linear_4310 --sim
    python examples/control_with_viser/control_with_viser.py --channel can0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.viser_control_interface import ViserControlInterface

if __name__ == "__main__":
    arm_choices = [a.value for a in ArmType]
    gripper_choices = [g.value for g in GripperType]

    parser = argparse.ArgumentParser(description="Viser control interface for i2rt robots")
    parser.add_argument("--arm", type=str, default="yam", choices=arm_choices)
    parser.add_argument("--gripper", type=str, default="linear_4310", choices=gripper_choices)
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel")
    parser.add_argument("--sim", action="store_true", help="Use SimRobot")
    parser.add_argument("--dt", type=float, default=0.02, help="Loop timestep (s)")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--site", type=str, default=None, help="EE site name (auto-detected if omitted)")
    parser.add_argument("--camera-stream-url", type=str, default=None, help="optional MJPEG stream for camera preview")
    parser.add_argument("--camera-calibration", type=str, default=None, help="hand_eye .npz/.json to visualize")
    parser.add_argument("--camera-mount-frame", type=str, default="link3", help="MuJoCo body used for camera mount")
    parser.add_argument("--clip-motor-torque", type=float, default=2.0, help="max absolute motor torque in Nm")
    args = parser.parse_args()

    arm = ArmType.from_string_name(args.arm)
    gripper = GripperType.from_string_name(args.gripper)

    if args.site is not None:
        site = args.site
    elif gripper == GripperType.YAM_TEACHING_HANDLE:
        site = "tcp_site"
    else:
        site = "grasp_site"

    robot = get_yam_robot(
        channel=args.channel,
        arm_type=arm,
        gripper_type=gripper,
        sim=args.sim,
        clip_motor_torque=args.clip_motor_torque,
    )

    iface = ViserControlInterface.from_robot(
        robot,
        ee_site=site,
        dt=args.dt,
        port=args.port,
        camera_stream_url=args.camera_stream_url,
        camera_calibration=args.camera_calibration,
        camera_mount_frame=args.camera_mount_frame,
    )
    iface.run()
