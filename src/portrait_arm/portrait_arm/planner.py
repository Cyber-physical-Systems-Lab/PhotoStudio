"""Pick a safe joint goal for a camera pose: IK from several seeds, keep the safe one that moves least."""
from __future__ import annotations

import numpy as np

from .kinematics import Kinematics
from .safety import SafetyChecker


def plan_camera_pose(kin: Kinematics, safety: SafetyChecker, T_goal, q_now, seeds, face_pos=None):
    """Returns (q_goal, None) or (None, reason). Seeds: current joints first, then named poses."""
    best, best_cost, reasons = None, np.inf, []
    tried = []
    for seed in [q_now] + list(seeds):
        q, ok, ep, er = kin.ik(T_goal, seed)
        if not ok:
            reasons.append(f'no IK from seed (pos err {ep * 1000:.0f} mm)')
            continue
        if any(np.max(np.abs(q - t)) < 1e-3 for t in tried):
            continue
        tried.append(q)
        safe, why = safety.check_path(q_now, q, face_pos, free_move=True)
        if not safe:
            reasons.append(why)
            continue
        cost = float(np.max(np.abs(q - q_now)))
        if cost < best_cost:
            best, best_cost = q, cost
    if best is None:
        return None, (reasons[0] if reasons else 'no IK solution')
    return best, None
