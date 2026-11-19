#!/usr/bin/env python3
"""Calibrate the portrait layout for wherever the arm has been pointed. NO robot motion.

1. Point the arm (camera in the gripper) roughly at the seat with DobotStudio Pro, then release
   the controller (disconnect DobotStudio Pro / TCP mode) and stop any portrait launch.
2. The person sits where the portrait will be taken and holds the AprilTag cube (36h11 id 4)
   in front of their eyes, facing the camera.
3. This tool reads the joints (feedback port only, no commands), finds the tag, works out the
   seat in the robot frame and the layout yaw, solves every search/shot pose from the current
   pose with the arm node's safety checks, and writes layout/current.yaml. The launch file loads
   that file on top of portrait.yaml.

    scripts/calibrate_layout.py                 # voice prompts, writes layout/current.yaml
    scripts/calibrate_layout.py --dry-run       # check only, write nothing
"""
import argparse
import datetime as dt
import os
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from layout_lib import SESSION_STYLES, DEFAULT_STYLE, apply_style, ROOT, DEFAULT_CONFIG, camera_kinematics, layout_params, load_config, solve_layout
sys.path.insert(0, os.path.join(ROOT, 'src/portrait_arm'))
from portrait_arm.dobot_v3 import DobotV3Client  # noqa: E402
from portrait_arm.kinematics import Kinematics, load_chain  # noqa: E402
from portrait_session.marker import MarkerLocator, eyes_behind_marker, stable_point  # noqa: E402


def say(text, voice=True):
    print('SAY:', text, flush=True)
    if voice:
        subprocess.run(['espeak-ng', '-v', 'en-gb', '-s', '150', text],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)


def read_robot(ip):
    """Joints [deg] and controller flange position [m] from the realtime feedback port (read-only)."""
    c = DobotV3Client(ip, log=lambda s: None)
    c._fb_sock = socket.create_connection((ip, 30004), timeout=3)
    c._running = True
    threading.Thread(target=c._feedback_loop, daemon=True).start()
    t0 = time.time()
    while c.feedback() is None:
        if time.time() - t0 > 3:
            sys.exit(f'no feedback from the robot at {ip}:30004')
        time.sleep(0.02)
    time.sleep(0.2)
    q, (p, _), fb = np.degrees(c.joints_rad()), c.tool_vector(), c.feedback()
    mode = int(fb['robot_mode'])
    c._running = False
    c._fb_sock.close()
    return q, p, mode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default='192.168.0.37')
    ap.add_argument('--device', default='/dev/video0')
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--out', default=os.path.join(ROOT, 'layout', 'current.yaml'))
    ap.add_argument('--timeout', type=float, default=30.0, help='seconds to wait for the marker')
    ap.add_argument('--hfov', type=float, default=82.0, help='BRIO horizontal FOV without a calibration file')
    ap.add_argument('--no-voice', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='check only, do not write the layout file')
    ap.add_argument('--style', choices=sorted(SESSION_STYLES), default=DEFAULT_STYLE,
                    help='portrait style for this place: natural (relaxed, recognisable background) or '
                         'professional (formal, clean background); sets angles, camera height, voice and look')
    a = ap.parse_args()
    P, SP = load_config(a.config)
    SP = apply_style(SP, a.style)
    print(f'style: {a.style} (angles {SP["shot_azimuths_deg"]} deg, camera {SP["camera_above_eyes_m"]:.2f} m above the eyes)')
    voice = not a.no_voice
    M = SP.get('marker', {})

    # 1. Robot pose (read-only) and the same kinematics check the arm node does before moving.
    q, p_ctrl, mode = read_robot(a.ip)
    kin_flange = Kinematics(load_chain(os.path.join(ROOT, 'src/portrait_description/urdf/nova5_portrait.urdf'),
                                       'base_link', 'tool_link'))
    err = np.linalg.norm(kin_flange.fk(np.radians(q))[:3, 3] - p_ctrl) * 1000
    print(f'robot mode {mode}, joints {np.round(q, 2).tolist()} deg, kinematics difference {err:.1f} mm')
    if err > 10:
        sys.exit('URDF and controller disagree by more than 10 mm: check Tool(0)/User(0) before continuing')
    kin_cam, _ = camera_kinematics(P)                 # robot base frame (no layout yaw)
    T_cam = kin_cam.fk(np.radians(q))

    # 2. Marker in front of the eyes.
    cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f'cannot open {a.device} (is a portrait launch still running?)')
    for _ in range(10):
        ok, img = cap.read()
    H, W = img.shape[:2]
    f = (W / 2) / np.tan(np.radians(a.hfov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    loc = MarkerLocator(M.get('id', 4), M.get('size_m', 0.0315), M.get('dictionary', 'DICT_APRILTAG_36h11'))
    say('Calibration. Please sit where you want your portrait taken, and hold the marker cube in front of your '
        'eyes, facing the camera.', voice)
    pts, found, shot, t_end, last_hint = [], None, None, time.time() + a.timeout, time.time()
    while time.time() < t_end and found is None:
        ok, img = cap.read()
        if not ok:
            continue
        pc = loc.detect(img, K, np.zeros(5))
        if pc is None:
            if time.time() - last_hint > 10:
                say('I cannot see the marker yet. Hold it in front of your eyes, facing the camera.', voice)
                last_hint = time.time()
            continue
        pts.append(pc)
        shot = img
        found = stable_point(pts, M.get('frames', 5))
    cap.release()
    if found is None:
        say('I could not see the marker. Calibration stopped.', voice)
        sys.exit(f'marker not found in {a.timeout:.0f} s ({len(pts)} detections): is the camera pointing at the seat?')
    eyes_cam = eyes_behind_marker(found, M.get('eye_offset_m', 0.05))
    eyes_base = (T_cam @ np.r_[eyes_cam, 1.0])[:3]
    say('Thank you, you can put the marker down.', voice)
    print(f'marker {np.linalg.norm(found):.3f} m from the camera; eyes at {np.round(eyes_base, 3).tolist()} m (robot base)')

    # 3. Layout: poses, limits and box from the current pose, checked like the arm node checks them.
    r, problems = solve_layout(eyes_base, q, P, SP)
    if r is None:
        say('This position does not work for the arm. Please see the screen.', voice)
        print('NOT OK:\n  ' + '\n  '.join(problems))
        sys.exit(1)
    stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    text = layout_params(r, f'calibrated {stamp}, style {a.style}', style=a.style)
    if a.dry_run:
        print(text)
        say('Calibration OK. Dry run, nothing written.', voice)
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if os.path.exists(a.out):
        os.replace(a.out, a.out + '.prev')
    open(a.out, 'w').write(text)
    cv2.imwrite(os.path.join(os.path.dirname(a.out), 'calibration.jpg'), cv2.resize(shot, (1280, 720)))
    print(f'wrote {a.out} (previous one kept as {os.path.basename(a.out)}.prev)')
    print(f'first move after the launch (current pose -> home): at most {r["first_move_deg"]} deg per joint')
    say('Calibration done.', voice)


if __name__ == '__main__':
    main()
