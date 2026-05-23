"""Nexus2 USB camera capture, cropping, and fisheye rectification."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

DEVICE_ID_SUBSTRING = "Atlas_Nexus2"
DEVICE_INDEX_SUFFIX = "video-index0"
REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_CAMERA_DATA_DIR = REPO_ROOT / "calibration" / "camera_data" / "kb4_6cam" / "per_camera_yaml"
REPO_HAND_EYE_DIR = REPO_ROOT / "calibration" / "camera_data" / "hand_eye"
FRAME_SIZE = (4000, 1200)
CAMERA_FPS = 30
DEFAULT_DEWARP_ZOOM = 1.0


@dataclass(frozen=True)
class CameraSpec:
    name: str
    index: int
    x0: int
    x1: int
    image_size: tuple[int, int] = (1920, 1200)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[: self.image_size[1], self.x0 : self.x1]


CAMERAS = {
    "left": CameraSpec("left", index=0, x0=160, x1=2080),
    "right": CameraSpec("right", index=1, x0=2080, x1=4000),
}


def find_nexus_device() -> str:
    override = os.environ.get("I2RT_NEXUS_CAMERA_DEVICE")
    if override:
        return str(Path(override).expanduser().resolve())

    by_id_dir = Path("/dev/v4l/by-id")
    candidates = sorted(
        path
        for path in by_id_dir.glob("*")
        if DEVICE_ID_SUBSTRING in path.name and path.name.endswith(DEVICE_INDEX_SUFFIX)
    )
    if candidates:
        return str(candidates[0].resolve())

    raise RuntimeError(
        f"could not find Nexus2 camera in {by_id_dir}; "
        f"expected name containing {DEVICE_ID_SUBSTRING!r} and ending with {DEVICE_INDEX_SUFFIX!r}"
    )


def camera_spec(name: str) -> CameraSpec:
    return CAMERAS[name]


def hand_eye_output_dir(camera: str) -> Path:
    camera_spec(camera)
    return REPO_HAND_EYE_DIR / camera


def hand_eye_calibration_path(camera: str) -> Path:
    return hand_eye_output_dir(camera) / "hand_eye.json"


@dataclass(frozen=True)
class FisheyeCameraModel:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rectified_camera_matrix: np.ndarray
    image_size: tuple[int, int]
    dewarp_zoom: float

    @classmethod
    def from_intrinsics(
        cls,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        image_size: tuple[int, int],
        dewarp_zoom: float = DEFAULT_DEWARP_ZOOM,
    ) -> "FisheyeCameraModel":
        distortion = np.asarray(distortion, dtype=float).reshape(-1)
        if distortion.size != 4:
            raise ValueError(f"expected a 4-coefficient KB4/fisheye model, got {distortion.size}")

        camera_matrix = np.asarray(camera_matrix, dtype=float)
        dewarp_zoom = float(dewarp_zoom)
        rectified = np.array(
            [
                [camera_matrix[0, 0] * dewarp_zoom, 0.0, image_size[0] / 2.0],
                [0.0, camera_matrix[1, 1] * dewarp_zoom, image_size[1] / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return cls(camera_matrix, distortion, rectified, image_size, dewarp_zoom)

    @property
    def fov(self) -> float:
        return float(2.0 * np.arctan(self.image_size[1] / (2.0 * self.rectified_camera_matrix[1, 1])))

    @property
    def aspect(self) -> float:
        return float(self.image_size[0] / self.image_size[1])

    def with_dewarp_zoom(self, dewarp_zoom: float) -> "FisheyeCameraModel":
        return self.from_intrinsics(self.camera_matrix, self.distortion, self.image_size, dewarp_zoom)

    def undistort_maps(self, cv2: Any) -> tuple[np.ndarray, np.ndarray]:
        return cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.distortion.reshape(4, 1),
            np.eye(3),
            self.rectified_camera_matrix,
            self.image_size,
            cv2.CV_16SC2,
        )

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.fisheye.undistortPoints(points.astype(np.float64), self.camera_matrix, self.distortion)


def model_from_repo_camera_data(camera: str) -> FisheyeCameraModel:
    spec = camera_spec(camera)
    yaml_name = f"{camera}_cam"
    payload = yaml.safe_load((REPO_CAMERA_DATA_DIR / f"{yaml_name}.yaml").read_text())
    record = payload[yaml_name]
    intrinsics = record["intrinsics"]
    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.array(
        [intrinsics["k1"], intrinsics["k2"], intrinsics["k3"], intrinsics["k4"]],
        dtype=np.float64,
    )
    return FisheyeCameraModel.from_intrinsics(camera_matrix, distortion, spec.image_size)


class NexusCamera:
    def __init__(
        self,
        cameras: Iterable[str],
        models: dict[str, FisheyeCameraModel],
        start_thread: bool = False,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self._cameras = tuple(camera_spec(name).name for name in cameras)
        self._device = find_nexus_device()
        self._models = {name: models[name] for name in self._cameras}
        self._dewarp_zoom = float(next(iter(self._models.values())).dewarp_zoom)
        self._maps = self._make_undistort_maps()
        self._cap = self._open_capture()
        self._latest_full_rgb: np.ndarray | None = None
        self._latest_rgb: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self._stop = False
        self._thread: threading.Thread | None = None
        if start_thread:
            _, frame = self.read_full()
            self._publish_frame(frame)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _open_capture(self) -> Any:
        cap = self._cv2.VideoCapture(self._device, self._cv2.CAP_V4L2)
        cap.set(self._cv2.CAP_PROP_FOURCC, self._cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        cap.set(self._cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {self._device}")
        return cap

    def _make_undistort_maps(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            name: self._models[name].undistort_maps(self._cv2)
            for name in self._cameras
        }

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()

    @property
    def device(self) -> str:
        return self._device

    def read_full(self, flush_frames: int = 1) -> tuple[float, np.ndarray]:
        frame = None
        ok = False
        for _ in range(max(1, flush_frames)):
            ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read {self._device}")
        return time.time(), frame

    def read_camera(self, camera: str, flush_frames: int = 1) -> tuple[float, np.ndarray]:
        timestamp, frame = self.read_full(flush_frames)
        image = camera_spec(camera).crop(frame)
        return timestamp, image

    def read_cameras(self, flush_frames: int = 1) -> tuple[float, dict[str, np.ndarray]]:
        timestamp, frame = self.read_full(flush_frames)
        return timestamp, {name: camera_spec(name).crop(frame) for name in self._cameras}

    def latest_full_rgb(self) -> np.ndarray:
        with self._lock:
            assert self._latest_full_rgb is not None
            return self._latest_full_rgb.copy()

    def latest_rgb(self, camera: str) -> np.ndarray:
        with self._lock:
            return self._latest_rgb[camera].copy()

    def camera_model(self, camera: str) -> FisheyeCameraModel:
        with self._lock:
            return self._models[camera]

    def dewarp_zoom(self) -> float:
        with self._lock:
            return self._dewarp_zoom

    def camera_info(self) -> dict[str, object]:
        with self._lock:
            return {
                "dewarp_zoom": self._dewarp_zoom,
                "models": {
                    name: {
                        "fov": self._models[name].fov,
                        "aspect": self._models[name].aspect,
                        "image_size": self._models[name].image_size,
                        "rectified_camera_matrix": self._models[name].rectified_camera_matrix,
                    }
                    for name in self._cameras
                },
            }

    def set_dewarp_zoom(self, dewarp_zoom: float) -> dict[str, object]:
        with self._lock:
            self._models = {
                name: model.with_dewarp_zoom(dewarp_zoom)
                for name, model in self._models.items()
            }
            self._dewarp_zoom = float(dewarp_zoom)
            self._maps = self._make_undistort_maps()
            return self.camera_info()

    def _run(self) -> None:
        period = 1.0 / CAMERA_FPS
        while not self._stop:
            start = time.time()
            _, frame = self.read_full()
            self._publish_frame(frame)
            time.sleep(max(0.0, period - (time.time() - start)))

    def _publish_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            maps = dict(self._maps)
        camera_images = {}
        for name in self._cameras:
            image = camera_spec(name).crop(frame)
            map1, map2 = maps[name]
            image = self._cv2.remap(image, map1, map2, interpolation=self._cv2.INTER_LINEAR)
            camera_images[name] = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        with self._lock:
            self._latest_full_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            self._latest_rgb = camera_images
