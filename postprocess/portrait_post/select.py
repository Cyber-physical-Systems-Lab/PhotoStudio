"""Pick the best frame of a burst: sharp, eyes open, face towards the camera."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

YUNET = 'face_detection_yunet_2023mar.onnx'


def find_face_model() -> str | None:
    for c in (os.environ.get('PORTRAIT_YUNET', ''),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', YUNET)):
        if c and os.path.exists(c):
            return c
    return None


@dataclass
class FrameScore:
    path: str
    image: np.ndarray
    bbox: np.ndarray | None = None
    landmarks: np.ndarray | None = None
    sharpness: float = 0.0
    eyes_open: float = 0.5
    frontal: float = 0.5
    total: float = 0.0
    notes: dict = field(default_factory=dict)


class FaceTools:
    def __init__(self, model_path: str | None = None):
        model_path = model_path or find_face_model()
        if model_path is None:
            raise RuntimeError('YuNet model not found; set PORTRAIT_YUNET')
        self.det = cv2.FaceDetectorYN.create(model_path, '', (320, 320), 0.6, 0.3, 20)
        self.mesh = None
        try:
            import mediapipe as mp
            self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                        refine_landmarks=True)
        except Exception:  # noqa: BLE001 - optional
            self.mesh = None

    def detect(self, img):
        h, w = img.shape[:2]
        s = min(1.0, 800 / w)
        small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, d = self.det.detect(small)
        if d is None or len(d) == 0:
            return None, None
        d = max(d, key=lambda r: r[2] * r[3])
        return d[:4] / s, d[4:14].reshape(5, 2) / s

    def eye_aspect(self, img) -> float | None:
        """Mean eye aspect ratio from MediaPipe Face Mesh (about 0.25-0.35 open, < 0.15 closed)."""
        if self.mesh is None:
            return None
        res = self.mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        P = lambda i: np.array([lm[i].x * w, lm[i].y * h])  # noqa: E731

        def ear(a, b, c, d, e, f):
            return (np.linalg.norm(P(b) - P(f)) + np.linalg.norm(P(c) - P(e))) / (2 * np.linalg.norm(P(a) - P(d)) + 1e-6)
        return float((ear(33, 160, 158, 133, 153, 144) + ear(362, 385, 387, 263, 373, 380)) / 2)


def score_frames(paths, tools: FaceTools) -> list[FrameScore]:
    scores = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        fs = FrameScore(p, img)
        fs.bbox, fs.landmarks = tools.detect(img)
        if fs.bbox is None:
            fs.total = -1.0
            scores.append(fs)
            continue
        x, y, w, h = fs.bbox.astype(int)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[max(0, y):y + h, max(0, x):x + w]
        fs.sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
        ear = tools.eye_aspect(img)
        if ear is not None:
            fs.eyes_open = float(np.clip((ear - 0.12) / 0.14, 0, 1))
            fs.notes['ear'] = ear
        lm = fs.landmarks
        eye_mid = lm[:2].mean(axis=0)
        iod = np.linalg.norm(lm[1] - lm[0]) + 1e-6
        # nose offset from the eye midline, normalised: 0 frontal, ~0.5 strong profile
        fs.frontal = float(np.clip(1 - abs(lm[2, 0] - eye_mid[0]) / iod / 0.5, 0, 1))
        scores.append(fs)
    valid = [s for s in scores if s.bbox is not None]
    if valid:
        smax = max(s.sharpness for s in valid) + 1e-6
        for s in valid:
            s.total = 0.5 * s.sharpness / smax + 0.3 * s.eyes_open + 0.2 * s.frontal
    return sorted(scores, key=lambda s: s.total, reverse=True)
