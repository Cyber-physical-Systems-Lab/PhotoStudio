import math
import os

import cv2
import numpy as np

from portrait_perception.face import Face, FaceAnalyzer, face_model

MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_detection_yunet_2023mar.onnx')


def test_pnp_recovers_distance_and_yaw():
    a = FaceAnalyzer(MODEL)
    K = np.array([[914.0, 0, 640], [0, 914.0, 360], [0, 0, 1]])
    M = face_model()
    rng = np.random.default_rng(0)
    for yaw in (-30, -15, 0, 15, 30):
        # yaw > 0 = subject turned to their left = nose moves to image right
        R, _ = cv2.Rodrigues(np.array([0.0, -math.radians(yaw), 0.0]))
        P = (R @ M.T).T + [0.02, -0.01, 0.7]
        uv = (K @ P.T).T
        uv = uv[:, :2] / uv[:, 2:]
        f = a.head_pose(Face(1.0, np.zeros(4), uv + rng.normal(0, 1.0, uv.shape)), K)
        assert abs(f.yaw - yaw) < 4, (yaw, f.yaw)
        assert abs(f.position[2] - 0.7) < 0.03
