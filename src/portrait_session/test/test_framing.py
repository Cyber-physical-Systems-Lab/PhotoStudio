import math

import numpy as np

from portrait_session.framing import framing_error, pose_correction, shot_pose


def project(T_cam, p, f=914.0, w=1280, h=720):
    pc = np.linalg.inv(T_cam) @ np.append(p, 1.0)
    return np.array([f * pc[0] / pc[2] + w / 2, f * pc[1] / pc[2] + h / 2]), pc[2]


def test_shot_pose_places_eyes_on_the_eye_line():
    face = np.array([1.0, 0.05, 0.78])
    for az in (-22, -15, 0, 10):
        T = shot_pose(face, az, 0.65, 0.03, 0.38, 914.0, 720)
        uv, z = project(T, face)
        du, dv = framing_error(uv, 1280, 720, 0.38)
        assert abs(du) < 0.01 and abs(dv) < 0.01, (az, du, dv)
        assert abs(np.linalg.norm(T[:3, 3] - face) - math.hypot(0.65, 0.03)) < 1e-6
        # camera on the subject's left (-y) for negative azimuth
        assert np.sign(T[1, 3] - face[1]) == np.sign(az) or az == 0


def test_camera_is_level():
    T = shot_pose([1.0, 0, 0.8], -15, 0.65, 0.03, 0.38, 914.0, 720)
    assert abs(T[2, 0]) < 1e-9          # image x axis horizontal (no roll)


def test_pose_correction_directions():
    assert pose_correction(-12.0, 0, 0, expected_yaw=-15, yaw_tol=10, pitch_tol=12) is None
    # subject turned 20 deg further to their left than wanted -> ask them to turn right
    assert pose_correction(5.0, 0, 0, -15, 10, 12) == 'turn_right'
    assert pose_correction(-30.0, 0, 0, -15, 10, 12) == 'turn_left'
    assert pose_correction(-15.0, 20, 0, -15, 10, 12) == 'chin_down'
    assert pose_correction(-15.0, -20, 0, -15, 10, 12) == 'chin_up'
    assert pose_correction(-15.0, 0, 15, -15, 10, 12) == 'level_head'


def test_position_correction():
    from portrait_session.framing import position_correction
    mark, tol = [1.0, 0.0], [0.10, 0.12]
    assert position_correction([1.02, 0.03, 0.5], mark, tol) is None       # any height is fine
    assert position_correction([0.85, 0.0, 0.9], mark, tol) == 'step_back'
    assert position_correction([1.2, 0.0, 0.9], mark, tol) == 'step_forward'
    # face on the robot's +y side = the person's right: they should step to their left
    assert position_correction([1.0, 0.2, 0.8], mark, tol) == 'step_left'
    assert position_correction([1.0, -0.2, 0.8], mark, tol) == 'step_right'
    assert position_correction([0.95, 0.0, 0.8], mark, tol, force=True) == 'step_back'


def test_face_tracker_learns_face_size():
    from portrait_session.framing import FaceTracker, look_at
    face = np.array([0.9, 0.1, 0.6])
    true_scale = 0.85          # small face: single-view distance is 1/0.85 too far
    tr = FaceTracker()
    for cam in ([0.35, 0.0, 0.8], [0.35, -0.2, 0.7], [0.3, 0.1, 0.6]):
        T = look_at(cam, face + [0.03, -0.02, 0.05])
        p_cam = T[:3, :3].T @ (face - T[:3, 3]) / true_scale
        tr.add(T[:3, 3], T[:3, :3], p_cam)
    assert tr.scale_known
    assert abs(tr.scale - true_scale) < 0.01
    assert np.linalg.norm(tr.position() - face) < 0.005
