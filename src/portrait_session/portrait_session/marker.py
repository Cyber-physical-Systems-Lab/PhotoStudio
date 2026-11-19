"""AprilTag held in front of the eyes: exact face distance at the start of a session. No ROS imports.

The single-view face distance comes from an assumed eye distance (63 mm) and is 10-20 % off for
many people. The participant briefly holds a cube with a known tag (AprilTag 36h11 id 4, 31.5 mm
black square) in front of their eyes; its pose gives the true distance, so the session can set the
person's scale before the arm moves.
"""
from __future__ import annotations

import cv2
import numpy as np


class MarkerLocator:
    def __init__(self, marker_id: int = 4, size_m: float = 0.0315, dictionary: str = 'DICT_APRILTAG_36h11'):
        self.marker_id = marker_id
        self.size = size_m
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)))
        h = size_m / 2
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)

    def detect(self, bgr, K, D):
        """Tag centre in the camera optical frame [m], or None."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None or self.marker_id not in ids.ravel():
            return None
        c = corners[list(ids.ravel()).index(self.marker_id)].reshape(4, 2).astype(float)
        ok, _, tvec = cv2.solvePnP(self.obj, c, np.asarray(K, float), np.asarray(D, float),
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return tvec.ravel() if ok else None


def stable_point(points, n: int = 5, max_spread_m: float = 0.02):
    """Median of the last n points once they agree within max_spread_m, else None."""
    if len(points) < n:
        return None
    P = np.asarray(points[-n:], float)
    m = np.median(P, axis=0)
    if np.max(np.linalg.norm(P - m, axis=1)) > max_spread_m:
        return None
    return m


def eyes_behind_marker(tag_cam, eye_offset_m: float = 0.05):
    """Face centre (between the eyes) in the camera frame: the tag is held this far in front of it."""
    t = np.asarray(tag_cam, float)
    return t * (1.0 + eye_offset_m / np.linalg.norm(t))


def face_scale(true_dist: float, pnp_dist: float):
    """Person scale for FaceTracker (true / eye-distance estimate), or None if implausible."""
    if pnp_dist <= 1e-6:
        return None
    s = true_dist / pnp_dist
    return s if 0.7 < s < 1.4 else None
