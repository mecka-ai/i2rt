# Live Camera Viser Deploy

This setup lets `control_with_viser.py` open `/dev/video2` directly. The Camera panel shows the full Nexus2 frame, and the 3D frustum uses the calibrated right-camera crop from the same capture stream.

Deploy local changes:

```bash
cd /Users/theol/Documents/github/i2rt
rsync -av i2rt/utils/viser_control_interface.py radxa@atlascm7660:/home/radxa/i2rt/i2rt/utils/viser_control_interface.py
rsync -av examples/control_with_viser/control_with_viser.py radxa@atlascm7660:/home/radxa/i2rt/examples/control_with_viser/control_with_viser.py
```

Run on the CM5:

```bash
ssh radxa@atlascm7660
pkill -f '[c]ontrol_with_viser.py'
pkill -f '[c]am_server.py'
cd /home/radxa/i2rt
.venv/bin/python -u examples/control_with_viser/control_with_viser.py
```

Open:

```text
http://atlascm7660:8080/
```

The old camera server is intentionally stopped. Viser owns `/dev/video2`, then passes the live images through its own UI so the sidebar and frustum do not depend on a second browser-visible port.
