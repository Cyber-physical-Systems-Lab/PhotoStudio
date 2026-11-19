#!/usr/bin/env python3
"""Move through the named poses and print the single-view face estimate at each (diagnostics)."""
import time

import numpy as np
import rclpy
from portrait_interfaces.msg import FaceObservation
from portrait_interfaces.srv import MoveNamed

rclpy.init()
n = rclpy.create_node('probe_face')
obs = []
n.create_subscription(FaceObservation, '/portrait/face', lambda m: obs.append((time.time(), m)), 10)
cli = n.create_client(MoveNamed, '/arm/move_named')
cli.wait_for_service()
for name in ['home', 'scan_left', 'scan_right', 'scan_low', 'scan_high', 'home']:
    f = cli.call_async(MoveNamed.Request(name=name))
    rclpy.spin_until_future_complete(n, f)
    t0 = time.time() + 3.0
    while time.time() < t0 + 5.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    sel = [m for t, m in obs if t > t0 and m.detected and m.has_base_position]
    if not sel:
        print(name, 'no face')
        continue
    P = np.array([[m.position_base.x, m.position_base.y, m.position_base.z] for m in sel])
    D = np.array([np.linalg.norm([m.position_camera.x, m.position_camera.y, m.position_camera.z]) for m in sel])
    print(f'{name:10s} n={len(sel)} face_base={np.round(np.median(P, 0), 3).tolist()} spread={np.round(P.std(0), 3).tolist()} '
          f'dist={np.median(D):.3f} yaw={np.median([m.yaw_deg for m in sel]):+.0f}', flush=True)
