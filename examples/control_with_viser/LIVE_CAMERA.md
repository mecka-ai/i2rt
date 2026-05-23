# Live Camera Viser Deploy

This setup lets `control_with_viser.py` open `/dev/video2` directly through `i2rt.utils.nexus_camera`. The Camera panel shows the full Nexus2 frame, and each calibrated 3D frustum asks that shared camera object for its own cropped, undistorted frame.

Deploy local changes from the laptop:

```bash
cd /Users/theol/Documents/github/elevator_detection
rsync -a --filter=':- .gitignore' --exclude='.git/' --stats ./ radxa@atlascm7660:/home/radxa/elevator_detection/
```

This copies the whole repo while respecting `.gitignore`, so local virtualenvs, caches, and bytecode are skipped. The CM5 uses the synced checkout at `/home/radxa/elevator_detection`; the older `/home/radxa/i2rt` path is not the deployment target for this workflow.

Run viser from the laptop:

```bash
ssh radxa@atlascm7660 "pkill -f '[c]ontrol_with_viser.py' || true; pkill -f '[c]am_server.py' || true; cd /home/radxa/elevator_detection/i2rt && .venv/bin/python -u examples/control_with_viser/control_with_viser.py"
```

Open:

```text
http://atlascm7660:8080/
```

The old camera server is intentionally stopped. Viser owns `/dev/video2`, then passes the live images through its own UI so the sidebar and frustum do not depend on a second browser-visible port.

Hand-eye calibration outputs are discovered from:

```text
/home/radxa/artifacts/hand_eye/hand_eye.json
/home/radxa/artifacts/hand_eye_left/hand_eye.json
```

Run right-camera calibration:

```bash
ssh -t radxa@atlascm7660 "pkill -f '[c]ontrol_with_viser.py' || true; pkill -f '[c]am_server.py' || true; cd /home/radxa/elevator_detection && i2rt/.venv/bin/python calibration/capture_hand_eye.py right"
```

Run left-camera calibration:

```bash
ssh -t radxa@atlascm7660 "pkill -f '[c]ontrol_with_viser.py' || true; pkill -f '[c]am_server.py' || true; cd /home/radxa/elevator_detection && i2rt/.venv/bin/python calibration/capture_hand_eye.py left"
```
