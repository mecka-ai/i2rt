"""Start the YAM Viser UI as a client of robot_control.server."""

# ruff: noqa: I001

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.viser_control_interface import ViserControlInterface
from robot_control import RemoteCameraFeed, RobotClient


DEFAULT_ROBOT_CONTROL_URL = "ws://127.0.0.1:8765"
DEFAULT_PORT = 8080


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Viser against robot_control.server.")
    parser.add_argument(
        "--robot-control-url",
        default=DEFAULT_ROBOT_CONTROL_URL,
        help="robot_control.server WebSocket URL. Use ws://atlascm7660:8765 when running Viser locally.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Viser HTTP port.")
    args = parser.parse_args()

    robot = RobotClient(args.robot_control_url)
    info = robot.get_info()
    camera_feed = RemoteCameraFeed(args.robot_control_url)
    xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)

    ViserControlInterface(
        robot,
        xml_path=xml_path,
        ee_site=info["ee_site"],
        port=args.port,
        camera_calibrations=info["camera"]["calibrations"],
        camera_mount_frame=info["camera_mount_frame"],
        camera_feed=camera_feed,
    ).run()


if __name__ == "__main__":
    main()
