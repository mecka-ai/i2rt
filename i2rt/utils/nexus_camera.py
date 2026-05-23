"""Nexus2 USB camera capture, cropping, and fisheye rectification."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

DEFAULT_SOURCE = "/dev/video2"
DEFAULT_INTRINSICS_PATH = Path("/home/radxa/camera-backend/calibration.json")
FRAME_SIZE = (4000, 1200)
FISHEYE_UNDISTORT_BALANCE = 0.5


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
    "left": CameraSpec("left", index=0, x0=0, x1=1920),
    "right": CameraSpec("right", index=1, x0=2080, x1=4000),
}


def camera_spec(name: str) -> CameraSpec:
    try:
        return CAMERAS[name]
    except KeyError as exc:
        raise ValueError(f"unknown Nexus2 camera {name!r}; expected one of {sorted(CAMERAS)}") from exc


def hand_eye_output_dir(camera: str) -> Path:
    camera_spec(camera)
    suffix = "" if camera == "right" else f"_{camera}"
    return Path(f"/home/radxa/artifacts/hand_eye{suffix}")


def hand_eye_calibration_path(camera: str) -> Path:
    return hand_eye_output_dir(camera) / "hand_eye.json"


@dataclass(frozen=True)
class FisheyeCameraModel:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rectified_camera_matrix: np.ndarray
    image_size: tuple[int, int]

    @classmethod
    def from_intrinsics(
        cls,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        image_size: tuple[int, int],
    ) -> "FisheyeCameraModel":
        import cv2

        distortion = np.asarray(distortion, dtype=float).reshape(-1)
        if distortion.size != 4:
            raise ValueError(f"expected a 4-coefficient KB4/fisheye model, got {distortion.size}")

        camera_matrix = np.asarray(camera_matrix, dtype=float)
        rectified = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix,
            distortion.reshape(4, 1),
            image_size,
            np.eye(3),
            balance=FISHEYE_UNDISTORT_BALANCE,
            new_size=image_size,
        )
        rectified = np.asarray(rectified, dtype=float)
        rectified[0, 2] = image_size[0] / 2.0
        rectified[1, 2] = image_size[1] / 2.0
        return cls(camera_matrix, distortion, rectified, image_size)

    @property
    def fov(self) -> float:
        return float(2.0 * np.arctan(self.image_size[1] / (2.0 * self.rectified_camera_matrix[1, 1])))

    @property
    def aspect(self) -> float:
        return float(self.image_size[0] / self.image_size[1])

    def undistort(self, image: np.ndarray) -> np.ndarray:
        import cv2

        map1, map2 = self.undistort_maps(cv2)
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

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


def model_from_intrinsics_file(path: Path, camera: str) -> FisheyeCameraModel:
    spec = camera_spec(camera)
    payload = json.loads(path.expanduser().read_text())
    record = payload["value0"]["intrinsics"][spec.index]
    intrinsics = record["intrinsics"] if "intrinsics" in record else record
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


def model_from_npz(path: Path, camera: str) -> Optional[FisheyeCameraModel]:
    spec = camera_spec(camera)
    data = np.load(path.expanduser())
    if "camera_matrix" not in data or "distortion" not in data:
        return None
    return FisheyeCameraModel.from_intrinsics(data["camera_matrix"], data["distortion"], spec.image_size)


class NexusCamera:
    def __init__(
        self,
        source: str = DEFAULT_SOURCE,
        cameras: Iterable[str] = CAMERAS.keys(),
        models: Optional[dict[str, FisheyeCameraModel]] = None,
        start_thread: bool = False,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self._source = source
        self._cameras = tuple(camera_spec(name).name for name in cameras)
        self._models = models or {}
        self._maps = {
            name: self._models[name].undistort_maps(cv2)
            for name in self._cameras
            if name in self._models
        }
        self._cap = self._open_capture(source)
        self._latest_full_rgb: Optional[np.ndarray] = None
        self._latest_rgb: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _open_capture(self, source: str) -> Any:
        if source.startswith("/dev/video"):
            cap = self._cv2.VideoCapture(int(source.removeprefix("/dev/video")), self._cv2.CAP_V4L2)
            cap.set(self._cv2.CAP_PROP_FOURCC, self._cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
            cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
            cap.set(self._cv2.CAP_PROP_FPS, 30)
            cap.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = self._cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera source {source!r}")
        return cap

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()

    def read_full(self, flush_frames: int = 1) -> tuple[float, np.ndarray]:
        frame = None
        ok = False
        for _ in range(max(1, flush_frames)):
            ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read camera source {self._source!r}")
        return time.time(), frame

    def read_camera(self, camera: str, flush_frames: int = 1, undistort: bool = False) -> tuple[float, np.ndarray]:
        timestamp, frame = self.read_full(flush_frames)
        image = camera_spec(camera).crop(frame)
        if undistort and camera in self._maps:
            map1, map2 = self._maps[camera]
            image = self._cv2.remap(image, map1, map2, interpolation=self._cv2.INTER_LINEAR)
        return timestamp, image

    def latest_full_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_full_rgb is None else self._latest_full_rgb.copy()

    def latest_rgb(self, camera: str) -> Optional[np.ndarray]:
        with self._lock:
            image = self._latest_rgb.get(camera)
            return None if image is None else image.copy()

    def _run(self) -> None:
        while not self._stop:
            try:
                _, frame = self.read_full()
            except RuntimeError:
                time.sleep(0.2)
                continue
            camera_images = {}
            for name in self._cameras:
                image = camera_spec(name).crop(frame)
                if name in self._maps:
                    map1, map2 = self._maps[name]
                    image = self._cv2.remap(image, map1, map2, interpolation=self._cv2.INTER_LINEAR)
                camera_images[name] = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest_full_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                self._latest_rgb = camera_images

    def __enter__(self) -> "NexusCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
