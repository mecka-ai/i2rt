# Viser Control Interface

Browser-based 3D visualization and control interface for i2rt robots, powered by [viser](https://github.com/nerfstudio-project/viser).

![Viser IK Control](assets/viser_ik_control.png)

## Quick Start

Start the robot-side owner first:

```bash
cd /home/radxa/elevator_detection/robot_control
../.venv/bin/python -m robot_control.server --host 0.0.0.0 --port 8765
```

Then start Viser. Viser connects to `ws://127.0.0.1:8765`; it does not open
CAN or the Nexus2 camera directly.

```bash
cd /home/radxa/elevator_detection/i2rt
../.venv/bin/python examples/control_with_viser/control_with_viser.py
```

Then open `http://localhost:8080` in your browser.

## Safety Gate

The interface starts in **read-only mode**. The robot will not accept any commands until you:

1. Visually confirm that the 3D model matches the physical robot pose.
2. Check the **Alignment Confirmed** checkbox.
3. Click **Enable Robot**.

This prevents unexpected motion when the GUI is first opened.

## Control Modes

### VIS (Mirror)

Passively mirrors the robot's current joint state in the 3D viewer. No commands are sent to the robot. Joint sliders update in real time to reflect the live state. Use this mode to observe the robot without any risk of motion.

### IK Control

Drag the 6-DOF transform gizmo to command the end-effector pose. An inverse kinematics solver (via [mink](https://github.com/kevinzakka/mink)) computes the required joint angles each frame and sends them to the robot. The arm joint sliders update to reflect the solved configuration. A gripper slider is available to control the gripper independently while the arm tracks the IK target.

### Joint Sliders

Directly control each joint angle (in degrees) and the gripper position using individual sliders. Changes are sent to the robot immediately each loop iteration. This mode is useful for precise per-joint positioning and testing range of motion.

## Configuration

The current example is fixed to the deployed YAM + no-gripper setup and the
local robot-control service URL. Change `ROBOT_CONTROL_URL` in
`control_with_viser.py` if Viser is running on a different host.
