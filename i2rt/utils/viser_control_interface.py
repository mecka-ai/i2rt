"""Viser control interface for i2rt robots.

Starts in DISABLED (read-only) mode, mirroring the robot's joint state in a
browser-based 3-D viewer.  Once the user confirms visual alignment with the
real robot and clicks "Enable", three control modes become available:

  VIS         — continues mirroring without sending any commands.
  IK control  — drag the 6-DOF target frame to control via IK.
  Joint sliders — per-joint angle sliders (degrees).

A PD-gains panel is shown for robots that expose kp/kd (MotorChainRobot).

See examples/control_with_viser/ for a runnable entry-point and README.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from i2rt.motor_drivers.dm_driver import ControlMode, PassiveEncoderInfo
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.utils.nexus_camera import (
    CAMERAS,
    NexusCamera,
    model_from_repo_camera_data,
)

# Teaching-handle button indicator visuals (mirrors mujoco_control_interface.py)
_BTN_OFF_RGB = (89, 89, 89)
_BTN_ON_RGB = (26, 230, 26)
_BTN_RADIUS = 0.022
# World-vertical offsets (meters along +Z) above the TCP. Index 0 = SYNC (top), 1 = RECORD (bottom).
_BTN_Z_OFFSETS = [0.10, 0.04]
_BTN_LABELS = ["SYNC", "RECORD"]
_CAMERA_MOUNT_BODY_ID = 4
_CAMERA_MOUNT_OFFSET_LOCAL = np.array([-0.11963644, 0.04517079, -0.03549660])
_DEFAULT_FRUSTUM_SCALE = 0.12
_DEFAULT_TORQUE_LIMIT_NM = 2.0
_MAX_TORQUE_LIMIT_NM = 4.0
_DEFAULT_PROFILE_MAX_VELOCITY = 0.5
_MAX_PROFILE_MAX_VELOCITY = 4.0
_DEFAULT_PROFILE_ACCELERATION = 1.0


class ViserControlInterface:
    """Browser-based robot visualiser and controller with a safety gate.

    The robot stays in read-only mode until the user confirms that the 3-D
    model matches the physical robot and presses "Enable".  This prevents
    unexpected motion when the GUI is first opened.
    """

    def __init__(
        self,
        robot: Robot,
        xml_path: str,
        camera_calibrations: Dict[str, str],
        ee_site: str = "grasp_site",
        dt: float = 0.02,
        port: int = 8080,
        camera_mount_frame: str = "geom_4_top",
    ) -> None:
        self._robot = robot
        self._ee_site = ee_site
        self._dt = dt
        self._port = port
        self._camera_mount_frame = camera_mount_frame
        self._camera_calibrations = self._load_camera_calibrations(camera_calibrations, camera_mount_frame)
        self._camera_feed = NexusCamera(
            cameras=CAMERAS.keys(),
            models={name: calibration["camera_model"] for name, calibration in self._camera_calibrations.items()},
            start_thread=True,
        )

        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._kin = Kinematics(xml_path, ee_site)

        self._nq = self._model.nq
        self._n_arm = sum(1 for j in range(self._model.njnt) if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)

        self._ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self._ee_site_id == -1:
            available = [mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self._model.nsite)]
            raise ValueError(f"Site {ee_site!r} not found in model. Available: {available}")

        info: Dict[str, Any] = robot.get_robot_info()
        n = robot.num_dofs()
        self._kp: np.ndarray = info.get("kp", np.full(n, 10.0)).copy()
        self._kd: np.ndarray = info.get("kd", np.full(n, 1.0)).copy()
        self._gain_scale = 1.0
        self._gripper_index: Optional[int] = info.get("gripper_index")
        self._gripper_limits: Optional[np.ndarray] = info.get("gripper_limits")
        self._is_sim: bool = info.get("sim", False)
        # tcp_site is exclusive to the teaching handle; covers sim where motor_chain is absent.
        self._with_teaching_handle: bool = ee_site == "tcp_site" or self._has_teaching_handle(robot)

        # Mesh data — filled by _collect_mesh_geoms()
        self._mesh_geom_ids: List[int] = []
        self._mesh_local_verts: Dict[int, np.ndarray] = {}
        self._mesh_local_faces: Dict[int, np.ndarray] = {}

        self._check_data = mujoco.MjData(self._model)
        self._in_collision = False

    @classmethod
    def from_robot(
        cls,
        robot: MotorChainRobot,
        camera_calibrations: Dict[str, str],
        ee_site: str = "grasp_site",
        dt: float = 0.02,
        port: int = 8080,
        camera_mount_frame: str = "geom_4_top",
    ) -> "ViserControlInterface":
        return cls(
            robot,
            robot.xml_path,
            camera_calibrations,
            ee_site,
            dt,
            port,
            camera_mount_frame=camera_mount_frame,
        )

    # ---- MuJoCo helpers -------------------------------------------------------

    def _mirror_robot(self) -> None:
        """Copy robot joint positions into MuJoCo and run forward kinematics."""
        qpos = self._robot.get_joint_pos()
        n = min(len(qpos), self._nq)
        self._data.qpos[:n] = qpos[:n]
        self._denormalize_slide_joints(n)
        self._enforce_eq_constraints()
        mujoco.mj_forward(self._model, self._data)

    def _denormalize_slide_joints(self, n_set: int) -> None:
        self._denormalize_slide_joints_on(self._data, n_set)

    def _denormalize_slide_joints_on(self, data: mujoco.MjData, n_set: int) -> None:
        """Scale normalised [0,1] slide-joint values to physical range (metres)."""
        for j in range(self._model.njnt):
            adr = self._model.jnt_qposadr[j]
            if adr >= n_set:
                continue
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self._model.jnt_range[j]
                data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)

    def _enforce_eq_constraints(self) -> None:
        self._enforce_eq_constraints_on(self._data)

    def _enforce_eq_constraints_on(self, data: mujoco.MjData) -> None:
        """Project qpos to satisfy joint equality constraints (e.g. coupled fingers)."""
        for i in range(self._model.neq):
            if self._model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            adr1 = self._model.jnt_qposadr[self._model.eq_obj1id[i]]
            adr2 = self._model.jnt_qposadr[self._model.eq_obj2id[i]]
            coef = self._model.eq_data[i, :5]
            data.qpos[adr2] = np.polyval(coef[::-1], data.qpos[adr1])

    def _has_self_collision(self, target_q: np.ndarray, n: int) -> bool:
        """Return True if *target_q* would cause self-collision.

        Uses a scratch ``MjData`` so the render state is not corrupted.
        Contacts involving the ground plane or adjacent (parent-child) bodies
        are ignored — only unexpected link-link penetrations count.
        """
        self._check_data.qpos[:n] = target_q[:n]
        self._denormalize_slide_joints_on(self._check_data, n)
        self._enforce_eq_constraints_on(self._check_data)
        mujoco.mj_forward(self._model, self._check_data)
        for i in range(self._check_data.ncon):
            c = self._check_data.contact[i]
            if c.dist >= -1e-3:
                continue
            if (
                self._model.geom_type[c.geom1] == mujoco.mjtGeom.mjGEOM_PLANE
                or self._model.geom_type[c.geom2] == mujoco.mjtGeom.mjGEOM_PLANE
            ):
                continue
            b1 = self._model.geom_bodyid[c.geom1]
            b2 = self._model.geom_bodyid[c.geom2]
            if self._model.body_parentid[b1] == b2 or self._model.body_parentid[b2] == b1:
                continue
            return True
        return False

    def _enter_vis_grav_comp(self) -> None:
        """Restore grav-comp on returning to VIS."""
        if isinstance(self._robot, MotorChainRobot) and self._robot.get_motor_control_mode() == ControlMode.POS_VEL:
            return
        self._robot.enter_gravity_comp_idle()

    def _enter_control_grav_comp(self) -> None:
        """CONTROL mode switches to PD on the next command."""
        self._apply_scaled_gains()
        self._in_collision = False

    def _apply_scaled_gains(self) -> None:
        self._robot.update_kp_kd(self._kp * self._gain_scale, self._kd * self._gain_scale)

    @staticmethod
    def _mat3_to_wxyz(mat3: np.ndarray) -> np.ndarray:
        """Convert a (3,3) or flat-9 rotation matrix to a wxyz quaternion."""
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, mat3.flatten())
        return q

    @staticmethod
    def _wxyz_to_mat3(wxyz: np.ndarray) -> np.ndarray:
        """Convert a wxyz quaternion to a (3,3) rotation matrix."""
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, wxyz)
        return mat.reshape(3, 3)

    def _ee_pose_4x4(self) -> np.ndarray:
        """Return the end-effector pose as a 4x4 homogeneous matrix."""
        site = self._data.site(self._ee_site_id)
        T = np.eye(4)
        T[:3, 3] = site.xpos.copy()
        T[:3, :3] = site.xmat.reshape(3, 3)
        return T

    @classmethod
    def _load_camera_calibrations(
        cls,
        paths: Dict[str, str],
        mount_frame: str,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            camera: cls._load_camera_calibration(camera, path, mount_frame)
            for camera, path in paths.items()
        }

    @staticmethod
    def _load_camera_calibration(
        camera: str,
        path: str,
        mount_frame: str,
    ) -> Dict[str, Any]:
        p = Path(path).expanduser()

        payload = json.loads(p.read_text())
        camera_model = model_from_repo_camera_data(camera)
        T_mount_camera = np.asarray(payload["T_mount_camera"], dtype=float)
        print(
            f"[viser] {camera} camera calibration: frame={mount_frame}, "
            f"offset={T_mount_camera[:3, 3]}"
        )
        print(
            f"[viser] {camera} camera model: KB4/fisheye, "
            f"rectified vfov={np.degrees(camera_model.fov):.1f} deg, "
            f"aspect={camera_model.aspect:.3f}"
        )
        return {
            "mount_frame": mount_frame,
            "T_mount_camera": T_mount_camera,
            "camera_model": camera_model,
        }

    # ---- Mesh extraction ------------------------------------------------------

    def _collect_mesh_geoms(self) -> None:
        """Cache per-geom mesh vertex/face arrays in local (geom) coordinates."""
        for geom_id in range(self._model.ngeom):
            if self._model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = self._model.geom_dataid[geom_id]
            v_adr = self._model.mesh_vertadr[mesh_id]
            v_num = self._model.mesh_vertnum[mesh_id]
            f_adr = self._model.mesh_faceadr[mesh_id]
            f_num = self._model.mesh_facenum[mesh_id]
            self._mesh_geom_ids.append(geom_id)
            self._mesh_local_verts[geom_id] = self._model.mesh_vert[v_adr : v_adr + v_num].copy()
            self._mesh_local_faces[geom_id] = self._model.mesh_face[f_adr : f_adr + f_num].copy()

    # ---- Viser scene ----------------------------------------------------------

    def _setup_scene(self, server: Any) -> Dict[int, Any]:
        """Add robot meshes to the viser scene; return {geom_id: mesh_handle}."""
        self._collect_mesh_geoms()
        handles: Dict[int, Any] = {}
        for geom_id in self._mesh_geom_ids:
            rgba = self._model.geom_rgba[geom_id]
            color = tuple(int(c * 255) for c in rgba[:3])
            handles[geom_id] = server.scene.add_mesh_simple(
                f"robot/geom_{geom_id}",
                self._mesh_local_verts[geom_id],
                self._mesh_local_faces[geom_id],
                color=color,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.zeros(3),
            )
        return handles

    def _update_scene(self, handles: Dict[int, Any]) -> None:
        """Refresh mesh transforms from current MuJoCo forward-kinematics state."""
        for geom_id in self._mesh_geom_ids:
            h = handles[geom_id]
            h.position = self._data.geom_xpos[geom_id].copy()
            h.wxyz = self._mat3_to_wxyz(self._data.geom_xmat[geom_id])

    # ---- Joint-limit helpers --------------------------------------------------

    def _hinge_joint_ranges_deg(self) -> List[tuple]:
        """Return (lo_deg, hi_deg) for each hinge joint in order."""
        ranges = []
        for j in range(self._model.njnt):
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                lo, hi = self._model.jnt_range[j]
                ranges.append((float(np.degrees(lo)), float(np.degrees(hi))))
        return ranges

    # ---- Teaching-handle helpers ---------------------------------------------

    @staticmethod
    def _has_teaching_handle(robot: Robot) -> bool:
        """Return True if robot has a teaching handle (passive encoder on the same CAN bus)."""
        chain = getattr(robot, "motor_chain", None)
        if chain is None:
            return False
        return (
            hasattr(chain, "get_same_bus_device_states")
            and hasattr(chain, "same_bus_device_driver")
            and chain.same_bus_device_driver is not None
        )

    def _get_teaching_handle_state(self) -> Optional[PassiveEncoderInfo]:
        """Read the latest teaching-handle encoder snapshot, or None if unavailable."""
        if self._is_sim:
            return None
        chain = getattr(self._robot, "motor_chain", None)
        if chain is None or not hasattr(chain, "get_same_bus_device_states"):
            return None
        if getattr(chain, "same_bus_device_driver", None) is None:
            return None
        states = chain.get_same_bus_device_states()
        return states[0] if states else None

    def _get_button_states(self) -> Optional[List[bool]]:
        """Read teaching-handle button states from real hardware, or None if unavailable."""
        state = self._get_teaching_handle_state()
        return list(state.io_inputs) if state is not None else None

    # ---- Main -----------------------------------------------------------------

    def run(self) -> None:
        """Open the viser server and run the visualisation / control loop."""
        import viser  # optional dependency — install with: pip install viser

        server = viser.ViserServer(port=self._port)
        print(f"[viser] Server started — open http://localhost:{self._port} in your browser")
        print("[viser] Starting in DISABLED (read-only) mode")
        print("[viser] Confirm robot alignment, then click 'Enable Robot'")

        # ---- Scene objects ----------------------------------------------------
        mesh_handles = self._setup_scene(server)
        ee_frame = server.scene.add_frame("ee_frame", axes_length=0.06, axes_radius=0.004)
        camera_mount_frame = server.scene.add_frame(
            "camera_mount_frame",
            axes_length=0.08,
            axes_radius=0.003,
        )
        camera_sidebar_image = None
        camera_frames: Dict[str, Any] = {}
        camera_frustums: Dict[str, Any] = {}
        frustum_colors = {"left": (255, 170, 60), "right": (40, 200, 255)}
        for camera, calibration in self._camera_calibrations.items():
            camera_frames[camera] = server.scene.add_frame(
                f"calibrated_camera/{camera}/opencv_frame",
                axes_length=0.06,
                axes_radius=0.002,
            )
            camera_model = calibration["camera_model"]
            camera_frustums[camera] = server.scene.add_camera_frustum(
                f"calibrated_camera/{camera}/frustum",
                fov=camera_model.fov,
                aspect=camera_model.aspect,
                scale=_DEFAULT_FRUSTUM_SCALE,
                line_width=2.0,
                color=frustum_colors.get(camera, (40, 200, 255)),
                image=self._camera_feed.latest_rgb(camera),
                format="jpeg",
                jpeg_quality=70,
            )
        ik_ctrl = server.scene.add_transform_controls(
            "/ik_target",
            position=np.zeros(3),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            scale=0.15,
            visible=False,
        )

        # Teaching-handle button spheres: viser icospheres are immutable in colour, so
        # add an "off" (gray) and "on" (green) sphere per position and toggle visibility.
        # Positions are refreshed each frame to track the TCP along world +Z.
        btn_spheres_off: List[Any] = []
        btn_spheres_on: List[Any] = []
        if self._with_teaching_handle:
            for i in range(len(_BTN_LABELS)):
                btn_spheres_off.append(
                    server.scene.add_icosphere(
                        f"/teaching_handle/btn_{i}_off",
                        radius=_BTN_RADIUS,
                        color=_BTN_OFF_RGB,
                        visible=True,
                    )
                )
                btn_spheres_on.append(
                    server.scene.add_icosphere(
                        f"/teaching_handle/btn_{i}_on",
                        radius=_BTN_RADIUS,
                        color=_BTN_ON_RGB,
                        visible=False,
                    )
                )

        # ---- Shared mutable state (read by loop, written by callbacks) --------
        state: Dict[str, Any] = {
            "enabled": False,
            "mode": "vis",
        }

        n_dofs = self._robot.num_dofs()
        info: Dict[str, Any] = self._robot.get_robot_info()
        has_kpkd = "kp" in info
        has_profile_control = isinstance(self._robot, MotorChainRobot) and not self._is_sim

        # ---- GUI — safety gate -----------------------------------------------
        with server.gui.add_folder("Safety"):
            align_cb = server.gui.add_checkbox("Alignment Confirmed", initial_value=False)
            enable_btn = server.gui.add_button("Enable Robot")
            enable_btn.disabled = True
            status_md = server.gui.add_markdown("**Status:** DISABLED (read-only)")

        # ---- GUI — motor command mode ----------------------------------------
        with server.gui.add_folder("Motor Profile"):
            motor_mode_dd = server.gui.add_dropdown(
                "Motor command",
                options=["MIT + gravity comp", "POS-VEL profile"],
                initial_value="POS-VEL profile"
                if info.get("control_mode", ControlMode.MIT) == ControlMode.POS_VEL
                else "MIT + gravity comp",
            )
            torque_limit_slider = server.gui.add_slider(
                "Torque limit (Nm)",
                min=0.0,
                max=_MAX_TORQUE_LIMIT_NM,
                step=0.05,
                initial_value=float(
                    _DEFAULT_TORQUE_LIMIT_NM
                    if np.isinf(info.get("clip_motor_torque", np.inf))
                    else np.clip(info["clip_motor_torque"], 0.0, _MAX_TORQUE_LIMIT_NM)
                ),
            )
            gain_scale_slider = server.gui.add_slider(
                "PD gain scale",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self._gain_scale,
            )
            profile_velocity_slider = server.gui.add_slider(
                "Max velocity (rad/s)",
                min=0.05,
                max=_MAX_PROFILE_MAX_VELOCITY,
                step=0.05,
                initial_value=float(info.get("profile_max_velocity", _DEFAULT_PROFILE_MAX_VELOCITY)),
            )
            profile_accel_slider = server.gui.add_slider(
                "Accel/decel (rad/s^2)",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=float(info.get("profile_acceleration", _DEFAULT_PROFILE_ACCELERATION)),
            )
            motor_mode_dd.disabled = True
            torque_limit_slider.disabled = True
            gain_scale_slider.disabled = True
            profile_velocity_slider.disabled = True
            profile_accel_slider.disabled = True

        # ---- GUI — camera feed -----------------------------------------------
        with server.gui.add_folder("Camera"):
            server.gui.add_markdown(f"**Mount frame:** `{self._camera_mount_frame}`")
            names = ", ".join(sorted(self._camera_calibrations))
            server.gui.add_markdown(f"**Calibrated cameras:** `{names}`")
            frustum_scale_slider = server.gui.add_slider(
                "Frustum length",
                min=0.02,
                max=0.50,
                step=0.01,
                initial_value=_DEFAULT_FRUSTUM_SCALE,
            )
            camera_sidebar_image = server.gui.add_image(
                self._camera_feed.latest_full_rgb(),
                label="video2 full frame",
                format="jpeg",
                jpeg_quality=70,
            )

        # ---- GUI — mode ------------------------------------------------------
        with server.gui.add_folder("Mode"):
            mode_dd = server.gui.add_dropdown(
                "Control mode",
                options=["VIS (mirror)", "IK control", "Joint sliders"],
                initial_value="VIS (mirror)",
            )
            mode_dd.disabled = True

        # ---- GUI — arm joint sliders -----------------------------------------
        joint_ranges = self._hinge_joint_ranges_deg()
        joint_sliders: List[Any] = []
        with server.gui.add_folder("Arm joints (deg)"):
            for i in range(self._n_arm):
                lo, hi = joint_ranges[i] if i < len(joint_ranges) else (-180.0, 180.0)
                s = server.gui.add_slider(f"j{i + 1}", min=lo, max=hi, step=0.1, initial_value=0.0)
                s.disabled = True
                joint_sliders.append(s)

        # ---- GUI — gripper slider --------------------------------------------
        gripper_slider: Optional[Any] = None
        if self._gripper_index is not None and self._gripper_limits is not None:
            with server.gui.add_folder("Gripper"):
                gripper_slider = server.gui.add_slider("Position", min=0.0, max=1.0, step=0.01, initial_value=0.0)
                gripper_slider.disabled = True

        # ---- GUI — teaching-handle indicators -------------------------------
        handle_btn_md: List[Any] = []
        handle_grip_slider: Optional[Any] = None
        if self._with_teaching_handle:
            with server.gui.add_folder("Teaching Handle"):
                for label in _BTN_LABELS:
                    handle_btn_md.append(server.gui.add_markdown(f"**{label}** [○]"))
                handle_grip_slider = server.gui.add_slider(
                    "Gripper Position", min=0.0, max=1.0, step=0.01, initial_value=0.0
                )
                handle_grip_slider.disabled = True

        # ---- GUI — PD gains --------------------------------------------------
        kp_sliders: List[Any] = []
        kd_sliders: List[Any] = []
        apply_btn: Optional[Any] = None
        if has_kpkd:
            with server.gui.add_folder("PD Gains"):
                for i in range(n_dofs):
                    kp_s = server.gui.add_slider(
                        f"kp[{i}]", min=0.0, max=300.0, step=0.5, initial_value=float(self._kp[i])
                    )
                    kd_s = server.gui.add_slider(
                        f"kd[{i}]", min=0.0, max=30.0, step=0.05, initial_value=float(self._kd[i])
                    )
                    kp_sliders.append(kp_s)
                    kd_sliders.append(kd_s)
                apply_btn = server.gui.add_button("Apply Gains")

        # ---- Callbacks -------------------------------------------------------

        def _selected_motor_mode() -> str:
            return ControlMode.POS_VEL if motor_mode_dd.value == "POS-VEL profile" else ControlMode.MIT

        def _set_profile_widgets_enabled() -> None:
            supported = has_profile_control
            enabled = supported and state["enabled"]
            selected = _selected_motor_mode()
            profile_selected = selected == ControlMode.POS_VEL
            mit_selected = selected == ControlMode.MIT
            profile_enabled = enabled and profile_selected
            mit_enabled = enabled and mit_selected
            motor_mode_dd.visible = supported
            torque_limit_slider.visible = supported and mit_selected
            gain_scale_slider.visible = supported and mit_selected
            profile_velocity_slider.visible = supported and profile_selected
            profile_accel_slider.visible = supported and profile_selected
            motor_mode_dd.disabled = not enabled
            torque_limit_slider.disabled = not mit_enabled
            gain_scale_slider.disabled = not mit_enabled
            profile_velocity_slider.disabled = not profile_enabled
            profile_accel_slider.disabled = not profile_enabled
            for slider in kp_sliders + kd_sliders:
                slider.visible = supported and mit_selected
                slider.disabled = not mit_enabled
            if apply_btn is not None:
                apply_btn.visible = supported and mit_selected
                apply_btn.disabled = not mit_enabled

        def _apply_motor_profile_settings() -> None:
            if not has_profile_control:
                return
            self._robot.set_profile_limits(float(profile_velocity_slider.value), float(profile_accel_slider.value))

        _set_profile_widgets_enabled()

        @align_cb.on_update
        def _(_: object) -> None:
            enable_btn.disabled = not align_cb.value

        @enable_btn.on_click
        def _(_: object) -> None:
            state["enabled"] = True
            align_cb.disabled = True
            enable_btn.disabled = True
            status_md.content = "**Status:** ENABLED"
            mode_dd.disabled = False
            _set_profile_widgets_enabled()
            print("[viser] Robot ENABLED — control active")
            # Sync sliders to current robot positions on enable
            q = self._robot.get_joint_pos()
            for i, s in enumerate(joint_sliders):
                if i < len(q):
                    s.value = float(np.degrees(q[i]))
            if gripper_slider is not None and self._gripper_index is not None:
                gripper_slider.value = float(q[self._gripper_index])
            self._apply_scaled_gains()

        @mode_dd.on_update
        def _(_: object) -> None:
            sel = mode_dd.value
            if sel == "VIS (mirror)":
                state["mode"] = "vis"
                ik_ctrl.visible = False
                for s in joint_sliders:
                    s.disabled = True
                if gripper_slider is not None:
                    gripper_slider.disabled = True
                _set_profile_widgets_enabled()
            elif sel == "IK control":
                state["mode"] = "ik"
                ik_ctrl.visible = True
                for s in joint_sliders:
                    s.disabled = True
                if gripper_slider is not None:
                    gripper_slider.disabled = False
                # Snap IK target to current EE pose
                T = self._ee_pose_4x4()
                ik_ctrl.position = T[:3, 3]
                ik_ctrl.wxyz = self._mat3_to_wxyz(T[:3, :3])
                # Sync gripper slider to current position
                if gripper_slider is not None and self._gripper_index is not None:
                    q = self._robot.get_joint_pos()
                    gripper_slider.value = float(q[self._gripper_index])
                _set_profile_widgets_enabled()
            elif sel == "Joint sliders":
                state["mode"] = "joint"
                ik_ctrl.visible = False
                for s in joint_sliders:
                    s.disabled = False
                if gripper_slider is not None:
                    gripper_slider.disabled = False
                # Sync sliders to current robot positions
                q = self._robot.get_joint_pos()
                for i, s in enumerate(joint_sliders):
                    if i < len(q):
                        s.value = float(np.degrees(q[i]))
                if gripper_slider is not None and self._gripper_index is not None:
                    gripper_slider.value = float(q[self._gripper_index])
                _set_profile_widgets_enabled()

        @motor_mode_dd.on_update
        def _(_: object) -> None:
            if not has_profile_control:
                _set_profile_widgets_enabled()
                return
            selected = _selected_motor_mode()
            self._robot.set_motor_control_mode(selected)
            if selected != ControlMode.POS_VEL and not (state["enabled"] and state["mode"] in ("ik", "joint")):
                self._enter_vis_grav_comp()
            print(f"[viser] Motor command mode set to {selected}")
            _set_profile_widgets_enabled()

        @profile_velocity_slider.on_update
        def _(_: object) -> None:
            _apply_motor_profile_settings()

        @profile_accel_slider.on_update
        def _(_: object) -> None:
            _apply_motor_profile_settings()

        if apply_btn is not None:

            @apply_btn.on_click
            def _(_: object) -> None:
                new_kp = np.array([s.value for s in kp_sliders])
                new_kd = np.array([s.value for s in kd_sliders])
                self._kp = new_kp
                self._kd = new_kd
                self._apply_scaled_gains()
                print(
                    f"[viser] Gains applied: scale={self._gain_scale:.2f}, "
                    f"kp={(new_kp * self._gain_scale).tolist()}, "
                    f"kd={(new_kd * self._gain_scale).tolist()}"
                )

        @torque_limit_slider.on_update
        def _(_: object) -> None:
            limit = float(torque_limit_slider.value)
            self._robot.update_clip_motor_torque(limit)

        @gain_scale_slider.on_update
        def _(_: object) -> None:
            self._gain_scale = float(gain_scale_slider.value)
            self._apply_scaled_gains()

        # ---- Main loop -------------------------------------------------------
        prev_controlled = False
        try:
            while True:
                self._mirror_robot()
                self._update_scene(mesh_handles)

                # Update EE frame indicator
                T = self._ee_pose_4x4()
                ee_frame.position = T[:3, 3]
                ee_frame.wxyz = self._mat3_to_wxyz(T[:3, :3])

                R_mount = self._data.xmat[_CAMERA_MOUNT_BODY_ID].reshape(3, 3)
                T_mount = np.eye(4)
                T_mount[:3, :3] = R_mount
                T_mount[:3, 3] = self._data.xpos[_CAMERA_MOUNT_BODY_ID] + R_mount @ _CAMERA_MOUNT_OFFSET_LOCAL
                camera_mount_frame.position = T_mount[:3, 3]
                camera_mount_frame.wxyz = self._mat3_to_wxyz(T_mount[:3, :3])

                for camera, calibration in self._camera_calibrations.items():
                    T_camera = T_mount @ calibration["T_mount_camera"]
                    camera_frames[camera].position = T_camera[:3, 3]
                    camera_frames[camera].wxyz = self._mat3_to_wxyz(T_camera[:3, :3])
                    frustum = camera_frustums[camera]
                    frustum.position = T_camera[:3, 3]
                    frustum.wxyz = self._mat3_to_wxyz(T_camera[:3, :3])
                    if frustum_scale_slider is not None:
                        frustum.scale = frustum_scale_slider.value
                    frustum.image = self._camera_feed.latest_rgb(camera)
                camera_sidebar_image.image = self._camera_feed.latest_full_rgb()

                if self._with_teaching_handle:
                    handle_state = self._get_teaching_handle_state()
                    buttons = list(handle_state.io_inputs) if handle_state is not None else [False, False]
                    for i, md in enumerate(handle_btn_md):
                        pressed = bool(buttons[i]) if i < len(buttons) else False
                        marker = "[●]" if pressed else "[○]"
                        md.content = f"**{_BTN_LABELS[i]}** {marker}"
                    ee_pos = T[:3, 3]
                    for i in range(len(btn_spheres_off)):
                        pressed = bool(buttons[i]) if i < len(buttons) else False
                        sphere_pos = ee_pos + np.array([0.0, 0.0, _BTN_Z_OFFSETS[i]])
                        btn_spheres_off[i].position = sphere_pos
                        btn_spheres_on[i].position = sphere_pos
                        btn_spheres_off[i].visible = not pressed
                        btn_spheres_on[i].visible = pressed
                    if handle_grip_slider is not None and handle_state is not None:
                        handle_grip_slider.value = float(np.clip(1.0 - float(handle_state.position), 0.0, 1.0))

                mode = state["mode"]
                controlled = state["enabled"] and mode in ("ik", "joint")
                if controlled != prev_controlled:
                    if controlled:
                        self._enter_control_grav_comp()
                    else:
                        self._enter_vis_grav_comp()
                    prev_controlled = controlled

                if not state["enabled"]:
                    # Read-only: update sliders to reflect live robot state
                    q = self._robot.get_joint_pos()
                    for i, s in enumerate(joint_sliders):
                        if i < len(q):
                            s.value = float(np.degrees(q[i]))
                    if gripper_slider is not None and self._gripper_index is not None:
                        gripper_slider.value = float(q[self._gripper_index])

                elif mode == "vis":
                    # Mirror only — no commands
                    q = self._robot.get_joint_pos()
                    for i, s in enumerate(joint_sliders):
                        if i < len(q):
                            s.value = float(np.degrees(q[i]))

                elif mode == "ik":
                    # Build target from user-dragged transform control
                    target = np.eye(4)
                    target[:3, 3] = np.asarray(ik_ctrl.position)
                    target[:3, :3] = self._wxyz_to_mat3(np.asarray(ik_ctrl.wxyz))
                    init_q = self._data.qpos[: self._nq].copy()
                    _, ik_q = self._kin.ik(target, self._ee_site, init_q=init_q)
                    cmd = self._robot.get_joint_pos().copy()
                    cmd[: self._n_arm] = ik_q[: self._n_arm]
                    if gripper_slider is not None and self._gripper_index is not None:
                        cmd[self._gripper_index] = float(gripper_slider.value)
                    n = min(len(cmd), self._nq)
                    if self._has_self_collision(cmd, n):
                        if not self._in_collision:
                            print("[viser] Collision detected — command blocked")
                            self._in_collision = True
                    else:
                        self._robot.command_joint_pos(cmd)
                        if self._in_collision:
                            print("[viser] Collision cleared — commands resumed")
                            self._in_collision = False
                    # Reflect solved angles in sliders
                    for i, s in enumerate(joint_sliders):
                        if i < self._n_arm:
                            s.value = float(np.degrees(ik_q[i]))

                elif mode == "joint":
                    # Build command from slider values
                    cmd = self._robot.get_joint_pos().copy()
                    for i, s in enumerate(joint_sliders):
                        if i < self._n_arm:
                            cmd[i] = float(np.radians(s.value))
                    if gripper_slider is not None and self._gripper_index is not None:
                        cmd[self._gripper_index] = float(gripper_slider.value)
                    n = min(len(cmd), self._nq)
                    if self._has_self_collision(cmd, n):
                        if not self._in_collision:
                            print("[viser] Collision detected — command blocked")
                            self._in_collision = True
                    else:
                        self._robot.command_joint_pos(cmd)
                        if self._in_collision:
                            print("[viser] Collision cleared — commands resumed")
                            self._in_collision = False

                time.sleep(self._dt)

        except KeyboardInterrupt:
            pass

        if self._camera_feed is not None:
            self._camera_feed.close()
        print("[viser] Stopped")
