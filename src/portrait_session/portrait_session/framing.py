"""Portrait framing geometry. No ROS imports."""
from __future__ import annotations

import math

import numpy as np


def look_at(eye, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Optical-frame pose (z forward, x right, y down) at eye looking at target, zero roll."""
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(world_up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def shot_pose(face_base, azimuth_deg: float, standoff: float, camera_above: float,
              eye_line_fraction: float, fy: float, image_height: int) -> np.ndarray:
    """Camera pose for one shot.

    The subject faces the robot (-x in the base frame). Positive azimuth moves the camera to the
    subject's right side (+y), giving a three-quarter view. The camera is tilted down so the eyes
    land at `eye_line_fraction` of the image height from the top.
    """
    a = math.radians(azimuth_deg)
    face = np.asarray(face_base, float)
    eye = face + standoff * np.array([-math.cos(a), math.sin(a), 0.0]) + np.array([0, 0, camera_above])
    T = look_at(eye, face)
    tilt = math.atan(((0.5 - eye_line_fraction) * image_height) / fy)
    T[:3, :3] = T[:3, :3] @ rot_x(-tilt)
    return T


def framing_error(eye_mid_px, width: int, height: int, eye_line_fraction: float) -> tuple[float, float]:
    """Normalised (du, dv) of the eye midpoint from where it should be."""
    du = (eye_mid_px[0] - width / 2) / width
    dv = (eye_mid_px[1] - eye_line_fraction * height) / height
    return du, dv


def pose_correction(yaw: float, pitch: float, roll: float, expected_yaw: float,
                    yaw_tol: float, pitch_tol: float, roll_tol: float = 10.0) -> str | None:
    """Phrase key for the most important head-pose correction, or None when it is good.

    yaw > 0 means the subject is turned to their left as seen by the camera.
    """
    err = yaw - expected_yaw
    if abs(err) > yaw_tol:
        return 'turn_right' if err > 0 else 'turn_left'
    if abs(roll) > roll_tol:
        return 'level_head'
    if pitch > pitch_tol:
        return 'chin_down'
    if pitch < -pitch_tol:
        return 'chin_up'
    return None


def position_correction(face_base, mark_xy, tol_xy, force: bool = False) -> str | None:
    """Phrase key asking the person to step towards the marked spot, or None.

    The person faces the robot (-x). Their left is the robot's -y side. Height is never corrected:
    the arm adapts to it. With `force` (pose unreachable) the largest offset is corrected even when it
    is inside the tolerance, as long as it is at least a third of it.
    """
    dx = float(face_base[0] - mark_xy[0])
    dy = float(face_base[1] - mark_xy[1])
    sx, sy = abs(dx) / tol_xy[0], abs(dy) / tol_xy[1]
    if max(sx, sy) < (0.33 if force else 1.0):
        return None
    if sx >= sy:
        return 'step_back' if dx < 0 else 'step_forward'
    return 'step_left' if dy > 0 else 'step_right'


class FaceTracker:
    """Face position that does not depend on the person's face size.

    A single view gives the direction to the face exactly but the distance only through an assumed
    eye distance (63 mm), which is off by 10-20 % for smaller or larger faces. The arm sees the face
    from several known camera positions, so the rays are triangulated; the ratio between triangulated
    and assumed-size distance is this person's scale, which then corrects every new single view.
    """

    def __init__(self, min_angle_deg: float = 4.0, max_rays: int = 12):
        self.min_cos = np.cos(np.radians(min_angle_deg))
        self.max_rays = max_rays
        self.rays = []          # (origin, unit direction, pnp distance)
        self.scale = 1.0
        self.scale_known = False
        self.scale_locked = False   # measured with the marker: triangulation must not override it

    def set_scale(self, s: float):
        self.scale, self.scale_known, self.scale_locked = float(s), True, True

    def reset_position(self):
        """The person moved (stepped): forget the rays, keep the learned scale."""
        self.rays = []

    def add(self, cam_origin, cam_R, p_cam):
        p_cam = np.asarray(p_cam, float)
        r = float(np.linalg.norm(p_cam))
        if r < 1e-6:
            return
        o = np.asarray(cam_origin, float)
        d = np.asarray(cam_R, float) @ (p_cam / r)
        self.rays.append((o, d, r))
        self.rays = self.rays[-self.max_rays:]
        self._update_scale()

    def _update_scale(self):
        if self.scale_locked or len(self.rays) < 2:
            return
        ds = np.array([d for _, d, _ in self.rays])
        if np.min(ds @ ds.T) > self.min_cos:          # all rays nearly parallel: no depth info
            return
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for o, d, _ in self.rays:
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ o
        x = np.linalg.solve(A, b)
        ratios = [float(np.dot(x - o, d)) / r for o, d, r in self.rays]
        s = float(np.median(ratios))
        if 0.7 < s < 1.4:
            self.scale, self.scale_known = s, True

    def position(self):
        """Latest ray, at its single-view distance corrected by the learned scale."""
        if not self.rays:
            return None
        o, d, r = self.rays[-1]
        return o + d * r * self.scale
