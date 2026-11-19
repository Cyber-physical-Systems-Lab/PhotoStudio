import math
import os

import numpy as np
import pytest
import yaml

from portrait_arm.kinematics import Kinematics, load_chain, look_at, make_tf, rpy_to_matrix
from portrait_arm.safety import SafetyChecker, SafetyLimits

HERE = os.path.dirname(__file__)
URDF = os.path.join(HERE, '..', '..', 'portrait_description', 'urdf', 'nova5_portrait.urdf')
CFG = os.path.join(HERE, '..', '..', 'portrait_bringup', 'config', 'portrait.yaml')


@pytest.fixture(scope='module')
def setup():
    P = yaml.safe_load(open(CFG))['arm_node']['ros__parameters']
    chain = load_chain(URDF, 'base_link', 'tool_link')
    mount = make_tf(rpy_to_matrix(*np.radians(P['camera_mount_rpy_deg'])), P['camera_mount_xyz'])
    from portrait_arm.kinematics import layout_base
    kin = Kinematics(chain, mount, np.radians(P['joint_lower_deg']), np.radians(P['joint_upper_deg']),
                     base=layout_base(P.get('layout_yaw_deg', 0.0)))
    safety = SafetyChecker(kin, SafetyLimits.from_params(P['safety']))
    return P, kin, safety


def test_zero_pose_matches_touchdrive_urdf(setup):
    _, kin, _ = setup
    flange = Kinematics(load_chain(URDF, 'base_link', 'tool_link'))
    T = flange.fk(np.zeros(6))
    # Nova 5 upright: flange 1.096 m above the base, offset 0.223 m along -y
    assert T[2, 3] == pytest.approx(1.096, abs=0.01)
    assert T[1, 3] == pytest.approx(-0.223, abs=0.01)


def test_ik_roundtrip(setup):
    P, kin, _ = setup
    home = np.radians(P['named_poses']['home'])
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = home + rng.uniform(-0.2, 0.2, 6)
        T = kin.fk(q)
        q2, ok, ep, er = kin.ik(T, home)
        assert ok, (ep, er)
        assert np.allclose(kin.fk(q2), T, atol=1e-3)


def test_named_poses_safe_and_look_at_subject(setup):
    P, kin, safety = setup
    face = np.array(P['nominal_face_position'])
    for name in P['named_poses']['names']:
        q = np.radians(P['named_poses'][name])
        ok, why = safety.check_config(q, face)
        assert ok, f'{name}: {why}'
        T = kin.fk(q)
        to_face = (face - T[:3, 3]) / np.linalg.norm(face - T[:3, 3])
        assert np.dot(T[:3, 2], to_face) > math.cos(math.radians(25)), name


def test_rejects_too_close_to_face(setup):
    P, kin, safety = setup
    home = np.radians(P['named_poses']['home'])
    face = kin.fk(home)[:3, 3] + np.array([0.3, 0, 0])      # face 30 cm in front of the lens
    ok, why = safety.check_config(home, face)
    assert not ok and 'face' in why


def test_rejects_outside_box_and_big_steps(setup):
    P, kin, safety = setup
    home = np.radians(P['named_poses']['home'])
    x_out = P['safety']['box_max'][0] + 0.10                # beyond the box / keep-out plane
    z = 0.5 * (P['safety']['box_min'][2] + P['safety']['box_max'][2])
    T = look_at([x_out, 0.0, z], [x_out + 0.6, 0, z])
    q, ok, _, _ = kin.ik(T, home)
    ok_path, why = safety.check_path(home, q, None)
    assert not ok_path
    far = home.copy()
    far[0] += math.radians(120)
    ok_path, why = safety.check_path(home, far, None)
    assert not ok_path and 'one step' in why


def test_speed_caps(setup):
    P, kin, safety = setup
    home = np.radians(P['named_poses']['home'])
    q1 = home + np.radians([20, 0, 0, 0, 0, 0])
    T = safety.move_duration(home, q1)
    assert T >= 1.875 * math.radians(20) / safety.lim.max_joint_speed - 1e-6
    d = np.linalg.norm(kin.fk(q1)[:3, 3] - kin.fk(home)[:3, 3])
    assert d / T <= safety.lim.max_cart_speed / 1.875 + 1e-6


def test_layout_base_rotates_fk_and_ik():
    from portrait_arm.kinematics import layout_base
    chain = load_chain(URDF, 'base_link', 'tool_link')
    raw = Kinematics(chain)
    lay = Kinematics(chain, base=layout_base(-90.0))      # subject along base -y
    q = np.radians([-82.69, -50.67, -109.12, 63.83, -89.67, 5.54])
    p_base = raw.fk(q)[:3, 3]
    p_lay = lay.fk(q)[:3, 3]
    # base -y becomes layout +x, base +x becomes layout +y
    assert np.allclose(p_lay, [-p_base[1], p_base[0], p_base[2]], atol=1e-9)
    q2, ok, _, _ = lay.ik(lay.fk(q), q + 0.1)
    assert ok and np.allclose(lay.fk(q2), lay.fk(q), atol=1e-3)


def test_current_real_pose_is_valid_in_config(setup):
    """The arm's pose after the 2026-11-18 grasp must pass the configured limits (start-up check)."""
    P, kin, checker = setup
    q = np.radians([-82.69, -50.67, -109.12, 63.83, -89.67, 5.54])
    ok, why = checker.check_config(q, np.array(P['nominal_face_position']))
    assert ok, why
