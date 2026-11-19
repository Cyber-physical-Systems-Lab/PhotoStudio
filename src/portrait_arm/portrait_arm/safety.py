"""Safety checks shared by every backend (sim, fake and the real Nova 5).

Every motion is a joint-space straight line (what the Dobot JointMovJ command does too),
so the whole path is sampled and checked, not just the goal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kinematics import Kinematics


@dataclass
class SafetyLimits:
    box_min: np.ndarray                 # camera position limits, robot base frame [m]
    box_max: np.ndarray
    link_x_max: float                   # no joint origin / camera may pass this x (towards subject)
    link_z_min: float                   # nothing below this z (table / pedestal)
    min_face_distance: float            # camera lens to face centre [m]
    max_cart_speed: float               # camera speed cap [m/s]
    max_joint_speed: float              # per-joint speed cap [rad/s]
    max_joint_step: float               # largest single joint change for a free move [rad]
    face_timeout: float                 # stop if the face is older than this while moving [s]
    min_move_time: float = 0.8

    @staticmethod
    def from_params(p: dict) -> 'SafetyLimits':
        return SafetyLimits(
            box_min=np.array(p['box_min'], float),
            box_max=np.array(p['box_max'], float),
            link_x_max=float(p['link_x_max']),
            link_z_min=float(p['link_z_min']),
            min_face_distance=float(p['min_face_distance']),
            max_cart_speed=float(p['max_cart_speed']),
            max_joint_speed=math.radians(float(p['max_joint_speed_deg'])),
            max_joint_step=math.radians(float(p['max_joint_step_deg'])),
            face_timeout=float(p['face_timeout']),
        )


class SafetyChecker:
    def __init__(self, kin: Kinematics, limits: SafetyLimits, samples_per_rad: float = 20.0):
        self.kin = kin
        self.lim = limits
        self.samples_per_rad = samples_per_rad

    # -------------------------------------------------------------- single configuration
    def check_config(self, q, face_pos=None) -> tuple[bool, str]:
        q = np.asarray(q, float)
        if np.any(q < self.kin.lower - 1e-6) or np.any(q > self.kin.upper + 1e-6):
            bad = [self.kin.joint_names[i] for i in np.where((q < self.kin.lower - 1e-6) |
                                                              (q > self.kin.upper + 1e-6))[0]]
            return False, f'joint limit: {",".join(bad)}'
        joints, T_cam = self.kin.frames(q)
        cam = T_cam[:3, 3]
        if np.any(cam < self.lim.box_min) or np.any(cam > self.lim.box_max):
            return False, f'camera {np.round(cam, 3).tolist()} outside workspace box'
        pts = [Tj[:3, 3] for Tj, _ in joints] + [cam]
        for i, p in enumerate(pts):
            if p[0] > self.lim.link_x_max:
                return False, f'link {i} x={p[0]:.3f} beyond keep-out plane {self.lim.link_x_max:.3f}'
            if i >= 2 and p[2] < self.lim.link_z_min:
                return False, f'link {i} z={p[2]:.3f} below floor limit {self.lim.link_z_min:.3f}'
        if face_pos is not None:
            for i, p in enumerate(pts[3:] if len(pts) > 3 else pts):
                d = float(np.linalg.norm(p - np.asarray(face_pos)))
                if d < self.lim.min_face_distance:
                    return False, f'{d:.3f} m from the face (min {self.lim.min_face_distance:.2f} m)'
        return True, 'ok'

    # -------------------------------------------------------------- whole joint path
    def check_path(self, q0, q1, face_pos=None, free_move: bool = True) -> tuple[bool, str]:
        q0 = np.asarray(q0, float)
        q1 = np.asarray(q1, float)
        dq = q1 - q0
        if free_move and np.max(np.abs(dq)) > self.lim.max_joint_step:
            j = int(np.argmax(np.abs(dq)))
            return False, (f'{self.kin.joint_names[j]} would move {math.degrees(abs(dq[j])):.0f} deg '
                           f'in one step (max {math.degrees(self.lim.max_joint_step):.0f})')
        n = max(2, int(np.max(np.abs(dq)) * self.samples_per_rad) + 2)
        for s in np.linspace(0.0, 1.0, n):
            ok, why = self.check_config(q0 + s * dq, face_pos)
            if not ok:
                return False, f'path at {s:.0%}: {why}'
        return True, 'ok'

    # -------------------------------------------------------------- timing
    def move_duration(self, q0, q1) -> float:
        q0 = np.asarray(q0, float)
        q1 = np.asarray(q1, float)
        t_joint = float(np.max(np.abs(q1 - q0))) / self.lim.max_joint_speed
        # Cartesian length of the camera path (sampled)
        pts = [self.kin.fk(q0 + s * (q1 - q0))[:3, 3] for s in np.linspace(0, 1, 11)]
        length = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:]))
        t_cart = length / self.lim.max_cart_speed
        # minimum-jerk peak speed is 1.875x the mean: stretch so the peak respects the cap
        return max(self.lim.min_move_time, 1.875 * max(t_joint, t_cart))


def min_jerk(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s ** 3 * (10 - 15 * s + 6 * s * s)
