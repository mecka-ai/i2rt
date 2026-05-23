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

This copies the whole repo while respecting `.gitignore`, so local virtualenvs, caches, and bytecode are skipped. The CM5 runs the synced code from `/home/radxa/elevator_detection` using the canonical environment at `/home/radxa/elevator_detection/.venv/bin/python`.

Stop existing robot-control/viser processes:

```bash
ssh radxa@atlascm7660 "pkill -f '[r]obot_control.server' || true; pkill -f '[c]ontrol_with_viser.py' || true"
```

Start the robot-control service:

```bash
ssh radxa@atlascm7660 "cd /home/radxa/elevator_detection/robot_control && ../.venv/bin/python -u -m robot_control.server --host 0.0.0.0 --port 8765"
```

In a second shell, run Viser:

```bash
ssh radxa@atlascm7660 "cd /home/radxa/elevator_detection/i2rt && ../.venv/bin/python -u examples/control_with_viser/control_with_viser.py"
```

Open:

```text
http://atlascm7660:8080/
```

No separate browser-visible camera server is needed. Viser never opens the
camera device; it receives `full`, `left_rectified`, and `right_rectified`
JPEG streams from `robot_control.server`.

Hand-eye calibration outputs are discovered from:

```text
/home/radxa/artifacts/hand_eye/hand_eye.json
/home/radxa/artifacts/hand_eye_left/hand_eye.json
```

Run both-camera calibration:

```bash
ssh -t radxa@atlascm7660 "cd /home/radxa/elevator_detection && .venv/bin/python calibration/capture_hand_eye.py"
```
