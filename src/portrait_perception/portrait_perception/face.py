"""Face detection (OpenCV YuNet) and head pose from 5 landmarks. No ROS imports."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

MODEL_NAME = 'face_detection_yunet_2023mar.onnx'

# Generic face model, metres. Origin between the eyes; x to image-right, y down, z away from the
# camera when the face looks straight into it. Landmark order matches YuNet:
# right eye (image left), left eye, nose tip, right mouth corner, left mouth corner.
def face_model(interocular: float = 0.063) -> np.ndarray:
    s = interocular / 0.063
    return s * np.array([
        [-0.0315, 0.000, 0.000],
        [0.0315, 0.000, 0.000],
        [0.000, 0.038, -0.030],
        [-0.024, 0.068, -0.008],
        [0.024, 0.068, -0.008],
    ])


@dataclass
class Face:
    score: float
    bbox: np.ndarray            # x, y, w, h (full-resolution pixels)
    landmarks: np.ndarray       # 5x2
    position: np.ndarray | None = None   # eye midpoint in camera optical frame [m]
    yaw: float = 0.0            # deg, + = subject turned to their left (nose to image right)
    pitch: float = 0.0          # deg, + = chin up
    roll: float = 0.0           # deg
    sharpness: float = 0.0

    @property
    def eye_mid(self) -> np.ndarray:
        return self.landmarks[:2].mean(axis=0)


def default_model_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('portrait_perception'), 'models', MODEL_NAME)
    except Exception:  # noqa: BLE001
        return os.path.join(os.path.dirname(__file__), '..', 'models', MODEL_NAME)


class FaceAnalyzer:
    def __init__(self, model_path: str | None = None, detect_width: int = 640,
                 score_threshold: float = 0.7, interocular: float = 0.063):
        self.detector = cv2.FaceDetectorYN.create(model_path or default_model_path(), '', (320, 320),
                                                  score_threshold, 0.3, 50)
        self.detect_width = detect_width
        self.model = face_model(interocular)

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        s = min(1.0, self.detect_width / w)
        small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, dets = self.detector.detect(small)
        faces = []
        if dets is None:
            return faces
        for d in dets:
            bbox = d[:4] / s
            lm = d[4:14].reshape(5, 2) / s
            faces.append(Face(float(d[14]), bbox, lm))
        faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return faces

    def head_pose(self, face: Face, K: np.ndarray, D: np.ndarray | None = None) -> Face:
        D = np.zeros(5) if D is None else D
        lm = face.landmarks.astype(np.float64)
        # Initial guess: frontal face at the depth implied by the eye distance. Iterative PnP from
        # this guess avoids the mirrored / flipped solutions a 5-point closed form can return.
        iod = max(1.0, float(np.linalg.norm(lm[1] - lm[0])))
        z0 = K[0, 0] * float(np.linalg.norm(self.model[1] - self.model[0])) / iod
        mid = lm[:2].mean(axis=0)
        t0 = np.array([(mid[0] - K[0, 2]) * z0 / K[0, 0], (mid[1] - K[1, 2]) * z0 / K[1, 1], z0])
        ok, rvec, tvec = cv2.solvePnP(self.model, lm, K, D, np.zeros((3, 1)), t0.reshape(3, 1).copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return face
        # Sanity: reject implausible solutions and fall back to the eye-distance depth.
        if not (0.5 * z0 < tvec[2, 0] < 1.6 * z0):
            tvec = t0.reshape(3, 1)
            rvec = np.zeros((3, 1))
        R, _ = cv2.Rodrigues(rvec)
        face.position = tvec.ravel()
        n = R @ np.array([0.0, 0.0, -1.0])        # face normal, towards the camera
        face.yaw = math.degrees(math.atan2(n[0], -n[2]))
        face.pitch = math.degrees(math.atan2(-n[1], -n[2]))
        face.roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        return face

    @staticmethod
    def sharpness(gray: np.ndarray, bbox) -> float:
        x, y, w, h = [int(v) for v in bbox]
        H, W = gray.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return 0.0
        return float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())

    def analyze(self, bgr: np.ndarray, K: np.ndarray, D=None) -> Face | None:
        faces = self.detect(bgr)
        if not faces:
            return None
        f = self.head_pose(faces[0], K, D)
        f.sharpness = self.sharpness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), f.bbox)
        return f
