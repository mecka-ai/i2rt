# Robot-Control Viser Deploy

This setup runs one robot-side owner, `robot_control.server`, which opens CAN
and the Nexus2 camera through `i2rt.utils.nexus_camera`. Viser connects as a
WebSocket client. The Camera panel shows the full Nexus2 frame streamed from
the service, and each calibrated 3D frustum uses its own rectified JPEG stream.

Camera intrinsics come from the synced repo data:

```text
/home/radxa/elevator_detection/calibration/camera_data/kb4_6cam/per_camera_yaml/left_cam.yaml
/home/radxa/elevator_detection/calibration/camera_data/kb4_6cam/per_camera_yaml/right_cam.yaml
```

Do not use `/home/radxa/camera-backend/calibration.json` for this setup. That file is for a different camera serial number.

Deploy local changes from the laptop:

```bash
cd /Users/theol/Documents/github/elevator_detection
rsync -a --filter=':- .gitignore' --exclude='.git/' --exclude='.venv/' --exclude='*/.venv/' --stats ./ radxa@atlascm7660:/home/radxa/elevator_detection/
```

This copies the whole repo while respecting `.gitignore`, so local virtualenvs,
caches, and bytecode are skipped. The CM5 runs the synced code from
`/home/radxa/elevator_detection` using the canonical environment at
`/home/radxa/elevator_detection/robot_control/.venv`.

Create or refresh the robot-control environment:

```bash
ssh radxa@atlascm7660 "cd /home/radxa/elevator_detection/robot_control && /home/radxa/.local/bin/uv venv --allow-existing .venv && /home/radxa/.local/bin/uv pip install --python .venv/bin/python -e ../i2rt -e ."
```

The explicit `/home/radxa/.local/bin/uv` path covers non-interactive shells
where `uv` is not on `PATH`.

Stop existing robot-control/viser processes:

```bash
ssh radxa@atlascm7660 "pkill -f '[r]obot_control.server' || true; pkill -f '[c]ontrol_with_viser.py' || true"
```

Start the robot-control service:

```bash
ssh -tt radxa@atlascm7660 "cd /home/radxa/elevator_detection/robot_control && .venv/bin/robot-control-server --host 0.0.0.0 --port 8765"
```

In a second shell, run Viser on the CM5:

```bash
ssh -tt radxa@atlascm7660 "cd /home/radxa/elevator_detection/robot_control && .venv/bin/yam-viser"
```

Or run Viser locally on the laptop:

```bash
cd /Users/theol/Documents/github/elevator_detection/robot_control
uv venv --allow-existing .venv
uv pip install --python .venv/bin/python -e ../i2rt -e .
.venv/bin/yam-viser --robot-control-url ws://atlascm7660:8765
```

Open:

```text
http://localhost:8080/      # local Viser
http://atlascm7660:8080/    # CM5 Viser
```

No separate browser-visible camera server is needed. Viser never opens the
camera device; it receives `full`, `left_rectified`, and `right_rectified`
JPEG streams from `robot_control.server`. The Camera panel's `Detection boxes`
toggle switches the left/right frustums to `left_detections_overlay` and
`right_detections_overlay`; RF-DETR medium inference starts only while those
overlay streams or raw detection streams are subscribed.

The Camera panel's `Dewarp zoom` slider updates the shared centered rectified
projection used by both left/right frustum streams.

Raw detection JSON is available over the same WebSocket server:

```python
from robot_control import RobotClient

client = RobotClient("ws://atlascm7660:8765")
print(client.get_detections("right"))
for payload in client.stream_detections("left"):
    print(payload["detections"])
```

Hand-eye calibration outputs are discovered from:

```text
/home/radxa/elevator_detection/calibration/camera_data/hand_eye/right/hand_eye.json
/home/radxa/elevator_detection/calibration/camera_data/hand_eye/left/hand_eye.json
```

Run both-camera calibration:

```bash
ssh -t radxa@atlascm7660 "cd /home/radxa/elevator_detection && .venv/bin/python calibration/capture_hand_eye.py"
```
