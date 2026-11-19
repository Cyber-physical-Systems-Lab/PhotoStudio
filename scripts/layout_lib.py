"""Portrait layout from a measured face position and the arm's current pose (no robot motion).

The portrait code works in a "layout" frame whose +x points from the robot to the subject
(arm_node `layout_yaw_deg`). Given the face position in the robot BASE frame and the joints the
operator left the arm in, `solve_layout` picks the yaw, solves IK for the search/scan poses and the
shot poses (also for faces a little off the seat) seeded at those joints, so every pose stays in
the operator's arm/wrist configuration, builds joint limits and a workspace box around them, and
checks everything with the arm node's own SafetyChecker. `layout_params` turns the result into a
ROS parameter file that overrides portrait.yaml.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'src/portrait_arm'), os.path.join(ROOT, 'src/portrait_session')]
from portrait_arm.kinematics import Kinematics, layout_base, load_chain, make_tf, rpy_to_matrix  # noqa: E402
from portrait_arm.safety import SafetyChecker, SafetyLimits  # noqa: E402
from portrait_session.framing import shot_pose  # noqa: E402
from portrait_session.styles import DEFAULT_STYLE, SESSION_STYLES, session_style  # noqa: E402

URDF = os.path.join(ROOT, 'src/portrait_description/urdf/nova5_portrait.urdf')
DEFAULT_CONFIG = os.path.join(ROOT, 'src/portrait_bringup/config/portrait.yaml')
NAMED = ['home', 'scan_left', 'scan_right', 'scan_low', 'scan_high']


def load_config(path=DEFAULT_CONFIG):
    c = yaml.safe_load(open(path))
    return c['arm_node']['ros__parameters'], c['session_node']['ros__parameters']


def camera_kinematics(P, base=None, lower=None, upper=None):
    mount = make_tf(rpy_to_matrix(*np.radians(P['camera_mount_rpy_deg'])), P['camera_mount_xyz'])
    chain = load_chain(URDF, 'base_link', 'tool_link')
    return Kinematics(chain, mount, lower, upper, base=base), chain


def family_bounds(seed_deg):
    """IK search bounds that keep the seed's elbow (joint 3) and wrist (joint 5) configuration."""
    s = np.asarray(seed_deg, float)
    lo = np.array([s[0] - 100, -135, -160, s[3] - 180, -170, s[5] - 180])
    hi = np.array([s[0] + 100, 135, 160, s[3] + 180, 170, s[5] + 180])
    for j in (2, 4):                        # same sign as the seed, at least 15 deg from 0
        if s[j] < 0:
            hi[j] = -15.0
        else:
            lo[j] = 15.0
    return lo, hi


def solve_layout(face_base, seed_deg, P, SP, standoff=None, hfov=82.0, scan_dz=(-0.03, 0.20),
                 face_spread=(0.15, 0.15, 0.08), margin=25.0, log=print):
    """Returns (result dict or None, list of problems)."""
    face_base = np.asarray(face_base, float)
    yaw = math.degrees(math.atan2(face_base[1], face_base[0]))
    base = layout_base(yaw)
    face = (base @ np.r_[face_base, 1.0])[:3]
    f_lo, f_hi = family_bounds(seed_deg)
    kin_free, chain = camera_kinematics(P, base, np.radians(f_lo), np.radians(f_hi))
    seed = np.radians(seed_deg)
    s = SP.get('standoff_m', 1.0) if standoff is None else standoff
    above = SP.get('camera_above_eyes_m', 0.10)
    eyeline = SP.get('eye_line_fraction', 0.38)
    W, H = 1920, 1080
    fy = (W / 2) / math.tan(math.radians(hfov) / 2)
    log(f'face: base {np.round(face_base, 3).tolist()} m, {np.hypot(*face_base[:2]):.2f} m from the base axis, '
        f'layout yaw {yaw:.1f} deg')

    targets = {
        'home': shot_pose(face, -15.0, s, above, 0.45, fy, H),
        'scan_left': shot_pose(face, -25.0, s, above, 0.45, fy, H),
        'scan_right': shot_pose(face, -5.0, s, above, 0.45, fy, H),
        'scan_low': shot_pose(face + [0, 0, scan_dz[0]], -15.0, s, above, 0.45, fy, H),
        'scan_high': shot_pose(face + [0, 0, scan_dz[1]], -15.0, s, above, 0.45, fy, H),
    }
    shots = {}
    dX, dY, dZ = face_spread
    for a in SP.get('shot_azimuths_deg', [-15.0, -22.0, -8.0]):
        for ox in (-dX, 0.0, dX):
            for oy in (-dY, 0.0, dY):
                for oz in (-dZ, 0.0, dZ):
                    tag = '' if (ox, oy, oz) == (0, 0, 0) else f' face{ox:+.2f},{oy:+.2f},{oz:+.2f}'
                    shots[f'shot_{a:+.0f}{tag}'] = shot_pose(face + [ox, oy, oz], a, s, above, eyeline, fy, H)

    problems, sol, q_prev = [], {}, seed
    for name, T in {**targets, **shots}.items():
        q, ok, ep, _ = kin_free.ik(T, q_prev)
        if not ok:
            q, ok, ep, _ = kin_free.ik(T, seed)
        if ok:
            sol[name] = q
            q_prev = q
        elif ' face' not in name:
            problems.append(f'{name} unreachable (camera {np.round(T[:3, 3], 2).tolist()}, {ep * 1000:.0f} mm off)')
    if problems:
        return None, problems + ['the seat is probably too far from (or too close to) the arm, or the start pose is '
                                 'in an awkward configuration: move the arm (or the seat) and calibrate again']

    allq = np.degrees(np.array(list(sol.values()) + [seed]))
    lo = np.maximum(np.maximum(np.floor(allq.min(0) - margin), np.degrees(chain.lower)), f_lo)
    hi = np.minimum(np.minimum(np.ceil(allq.max(0) + margin), np.degrees(chain.upper)), f_hi)
    cams = np.array([kin_free.fk(q)[:3, 3] for q in sol.values()])
    min_face = P['safety']['min_face_distance']
    x_cap = face[0] - min_face - 0.10                      # box edge stays >= 0.55 m from the face
    box_min = np.round(cams.min(0) - [0.12, 0.20, 0.12], 2)
    box_max = np.round(np.minimum(cams.max(0) + [0.12, 0.20, 0.15], [x_cap, 9, 9]), 2)
    box_min[2] = max(box_min[2], P['safety']['link_z_min'] + 0.15)
    lim = dict(P['safety'])
    lim.update(box_min=box_min.tolist(), box_max=box_max.tolist(), link_x_max=float(box_max[0]))
    kin, _ = camera_kinematics(P, base, np.radians(lo), np.radians(hi))
    checker = SafetyChecker(kin, SafetyLimits.from_params(lim))
    for name in NAMED + [n for n in sol if n.startswith('shot_') and ' face' not in n]:
        ok, why = checker.check_config(sol[name], face)
        if not ok:
            problems.append(f'{name}: {why}')
    ok, why = checker.check_path(seed, sol['home'], face, free_move=True)
    if not ok:
        problems.append(f'path from the current pose to home: {why}')
    if problems:
        return None, problems
    step = float(np.max(np.abs(np.degrees(sol['home'] - seed))))
    log(f'all poses reachable and safe; first move (current pose -> home) changes a joint by at most {step:.1f} deg')
    return {
        'yaw_deg': round(yaw, 2), 'face_layout': np.round(face, 3).tolist(), 'face_base': np.round(face_base, 3).tolist(),
        'joint_lower_deg': np.round(lo, 1).tolist(), 'joint_upper_deg': np.round(hi, 1).tolist(),
        'named': {n: np.round(np.degrees(sol[n]), 2).tolist() for n in NAMED},
        'box_min': box_min.tolist(), 'box_max': box_max.tolist(), 'link_x_max': float(box_max[0]),
        'first_move_deg': round(step, 1), 'seed_deg': [round(float(v), 2) for v in seed_deg],
    }, []


def apply_style(SP, style):
    """Session parameters with a style preset (angles, camera height, framing, voice) applied."""
    SP = dict(SP)
    SP.update(session_style(style))
    return SP


def layout_params(r, note='', style=None):
    """ROS 2 parameter file content (overrides portrait.yaml) for a solve_layout result."""
    arm = {
        'layout_yaw_deg': r['yaw_deg'],
        'joint_lower_deg': r['joint_lower_deg'], 'joint_upper_deg': r['joint_upper_deg'],
        'named_poses': {'names': NAMED, **r['named']},
        'safety': {'box_min': r['box_min'], 'box_max': r['box_max'], 'link_x_max': r['link_x_max']},
        'nominal_face_position': r['face_layout'],
    }
    session = {'mark_position': r['face_layout'][:2], 'mark_height': r['face_layout'][2],
               **(session_style(style) if style else {})}
    head = ('# Portrait layout generated by scripts/calibrate_layout.py; loaded on top of portrait.yaml.\n'
            f'# {note}\n# face (robot base frame) {r["face_base"]} m, start joints {r["seed_deg"]} deg\n')
    return head + yaml.safe_dump({'arm_node': {'ros__parameters': arm},
                                  'session_node': {'ros__parameters': session}}, sort_keys=False,
                          default_flow_style=None)


def merged_config(config_path, layout_path=None):
    """portrait.yaml with a layout file merged on top (for tools that read the YAML directly)."""
    c = yaml.safe_load(open(config_path))
    if layout_path and os.path.exists(layout_path):
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v
        merge(c, yaml.safe_load(open(layout_path)))
    return c
