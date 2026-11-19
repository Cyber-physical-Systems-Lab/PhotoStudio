import cv2
import numpy as np

from portrait_session.framing import FaceTracker
from portrait_session.marker import MarkerLocator, eyes_behind_marker, face_scale, stable_point


def synthetic_view(dist_m, size_m=0.0315, f=1100.0, W=1920, H=1080):
    """Tag 4 facing the camera at dist_m on the optical axis, rendered with a pinhole model."""
    side_px = int(round(f * size_m / dist_m))
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 4, side_px * 10 // 8)
    img = np.full((H, W), 255, np.uint8)
    # generateImageMarker includes a 1-bit black border inside side; the outer black square = side_px
    tag = cv2.resize(tag, (side_px * 10 // 8, side_px * 10 // 8), interpolation=cv2.INTER_NEAREST)
    tag = cv2.resize(tag, (side_px, side_px), interpolation=cv2.INTER_NEAREST)
    y0, x0 = H // 2 - side_px // 2, W // 2 - side_px // 2
    img[y0:y0 + side_px, x0:x0 + side_px] = tag
    img = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), K


def test_marker_distance_from_synthetic_view():
    img, K = synthetic_view(1.0)
    p = MarkerLocator().detect(img, K, np.zeros(5))
    assert p is not None
    assert abs(np.linalg.norm(p) - 1.0) < 0.03


def test_stable_point_needs_agreeing_frames():
    pts = [np.array([0, 0, 1.0])] * 4
    assert stable_point(pts, 5) is None
    pts.append(np.array([0, 0, 1.005]))
    assert np.allclose(stable_point(pts, 5), [0, 0, 1.0])
    pts.append(np.array([0, 0.1, 1.0]))          # someone moved the cube: no agreement
    assert stable_point(pts, 5) is None


def test_eyes_behind_marker_and_scale():
    eyes = eyes_behind_marker([0, 0, 0.95], 0.05)
    assert np.allclose(eyes, [0, 0, 1.0])
    assert abs(face_scale(1.0, 0.9) - 1.0 / 0.9) < 1e-9
    assert face_scale(1.0, 0.5) is None           # implausible: ignore


def test_locked_scale_survives_triangulation():
    t = FaceTracker()
    t.set_scale(1.1)
    t.add([0, 0, 0], np.eye(3), [0, 0, 1.0])
    t.add([0.3, 0, 0], np.eye(3), [-0.3, 0, 1.0])   # wide baseline would normally re-estimate scale
    assert t.scale == 1.1
