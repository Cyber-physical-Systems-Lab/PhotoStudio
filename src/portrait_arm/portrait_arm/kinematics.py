"""Pure-numpy forward/inverse kinematics for the Nova 5, read from the URDF.

No ROS imports here so the module can be unit tested and reused by the Isaac Sim
script and offline tools.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- transforms
def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw (Rz(y) @ Ry(p) @ Rx(r))."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    sy = -R[2, 0]
    p = math.asin(max(-1.0, min(1.0, sy)))
    if abs(math.cos(p)) > 1e-6:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        r = math.atan2(-R[1, 2], R[1, 1])
        y = 0.0
    return r, p, y


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def make_tf(R: np.ndarray | None = None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def rotation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Axis-angle vector (in base frame) rotating R_cur into R_des."""
    R_err = R_des @ R_cur.T
    cos_a = max(-1.0, min(1.0, (np.trace(R_err) - 1) / 2))
    a = math.acos(cos_a)
    if a < 1e-9:
        return np.zeros(3)
    if abs(a - math.pi) < 1e-6:
        # 180 degrees: pick axis from the symmetric part
        w, v = np.linalg.eigh((R_err + np.eye(3)) / 2)
        return v[:, np.argmax(w)] * a
    v = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
    return v * a / (2 * math.sin(a))


def quat_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def matrix_from_quat(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def look_at(eye, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera optical-frame pose (z forward, x right, y down) at `eye` looking at `target`,
    with zero roll (image x axis horizontal)."""
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    up = np.asarray(world_up, float)
    x = np.cross(z, up)          # right = forward x up  (for y-down optical frame)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, -1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)           # down
    return make_tf(np.column_stack([x, y, z]), eye)


# ----------------------------------------------------------------------------- URDF chain
@dataclass
class Joint:
    name: str
    jtype: str
    origin: np.ndarray            # 4x4 parent->joint frame
    axis: np.ndarray
    lower: float = -math.pi
    upper: float = math.pi
    velocity: float = 1.0


@dataclass
class Chain:
    joints: list[Joint] = field(default_factory=list)
    tip_offset: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def active(self) -> list[Joint]:
        return [j for j in self.joints if j.jtype in ('revolute', 'continuous')]

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.active]

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.active])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.active])


def load_chain(urdf_path: str, base: str = 'base_link', tip: str = 'tool_link') -> Chain:
    root = ET.parse(urdf_path).getroot()
    by_child = {}
    for j in root.findall('joint'):
        by_child[j.find('child').get('link')] = j
    seq = []
    link = tip
    while link != base:
        if link not in by_child:
            raise ValueError(f'No path from {base} to {tip} in {urdf_path}')
        j = by_child[link]
        seq.append(j)
        link = j.find('parent').get('link')
    seq.reverse()
    chain = Chain()
    for j in seq:
        o = j.find('origin')
        xyz = [float(v) for v in (o.get('xyz', '0 0 0').split() if o is not None else [0, 0, 0])]
        rpy = [float(v) for v in (o.get('rpy', '0 0 0').split() if o is not None else [0, 0, 0])]
        ax = j.find('axis')
        axis = np.array([float(v) for v in ax.get('xyz').split()]) if ax is not None else np.array([0, 0, 1.0])
        lim = j.find('limit')
        jt = Joint(j.get('name'), j.get('type'), make_tf(rpy_to_matrix(*rpy), xyz), axis)
        if lim is not None:
            jt.lower = float(lim.get('lower', -math.pi))
            jt.upper = float(lim.get('upper', math.pi))
            jt.velocity = float(lim.get('velocity', 1.0))
        chain.joints.append(jt)
    return chain


class Kinematics:
    """FK / damped-least-squares IK for a serial chain plus a fixed tool offset.

    `base` is an optional fixed transform in front of the chain: with base = Rz(-yaw) every pose
    (FK, IK targets, joint frames) is expressed in a "layout" frame whose +x points along `yaw` in
    the robot base frame, so the portrait code can keep assuming the subject is at +x.
    """

    def __init__(self, chain: Chain, tool: np.ndarray | None = None,
                 lower: np.ndarray | None = None, upper: np.ndarray | None = None,
                 base: np.ndarray | None = None):
        self.chain = chain
        self.tool = np.eye(4) if tool is None else np.asarray(tool, float)
        self.base = np.eye(4) if base is None else np.asarray(base, float)
        self.lower = chain.lower if lower is None else np.asarray(lower, float)
        self.upper = chain.upper if upper is None else np.asarray(upper, float)
        self.n = len(chain.active)

    @property
    def joint_names(self):
        return self.chain.joint_names

    def frames(self, q) -> list[np.ndarray]:
        """World pose and world axis of every active joint, plus the tool pose (last)."""
        T = self.base.copy()
        out = []
        k = 0
        for j in self.chain.joints:
            T = T @ j.origin
            if j.jtype in ('revolute', 'continuous'):
                out.append((T.copy(), T[:3, :3] @ j.axis))
                T = T @ make_tf(axis_angle(j.axis, q[k]))
                k += 1
        return out, T @ self.tool

    def fk(self, q) -> np.ndarray:
        return self.frames(q)[1]

    def jacobian(self, q) -> tuple[np.ndarray, np.ndarray]:
        js, T_tip = self.frames(q)
        p_tip = T_tip[:3, 3]
        J = np.zeros((6, self.n))
        for i, (Tj, ax) in enumerate(js):
            J[:3, i] = np.cross(ax, p_tip - Tj[:3, 3])
            J[3:, i] = ax
        return J, T_tip

    def ik(self, T_des: np.ndarray, q0, max_iter: int = 300, tol_pos: float = 1e-4,
           tol_rot: float = 1e-3, damping: float = 0.02, rot_weight: float = 0.5):
        """Damped least squares, seeded at q0. Returns (q, ok, pos_err, rot_err)."""
        q = np.array(q0, float)
        W = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])
        for _ in range(max_iter):
            J, T = self.jacobian(q)
            e_p = T_des[:3, 3] - T[:3, 3]
            e_r = rotation_error(T[:3, :3], T_des[:3, :3])
            ep, er = np.linalg.norm(e_p), np.linalg.norm(e_r)
            if ep < tol_pos and er < tol_rot:
                return q, True, ep, er
            e = W @ np.concatenate([e_p, e_r])
            Jw = W @ J
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + (damping ** 2) * np.eye(6), e)
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lower, self.upper)
        T = self.fk(q)
        ep = np.linalg.norm(T_des[:3, 3] - T[:3, 3])
        er = np.linalg.norm(rotation_error(T[:3, :3], T_des[:3, :3]))
        return q, (ep < 1e-3 and er < 1e-2), ep, er


def layout_base(yaw_deg: float) -> np.ndarray:
    """Base transform for Kinematics(base=...): layout +x = robot base direction `yaw_deg`."""
    a = math.radians(-yaw_deg)
    return make_tf(np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]))
