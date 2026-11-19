#!/usr/bin/env python3
"""Save one /portrait/debug_image frame and print the matching face observation (diagnostics)."""
import sys

import cv2
import numpy as np
import rclpy
from portrait_interfaces.msg import FaceObservation
from sensor_msgs.msg import Image

rclpy.init()
n = rclpy.create_node('grab_debug')
got = {}
n.create_subscription(Image, '/portrait/debug_image', lambda m: got.setdefault('img', m), 2)
n.create_subscription(FaceObservation, '/portrait/face', lambda m: got.setdefault('face', m), 2)
while rclpy.ok() and len(got) < 2:
    rclpy.spin_once(n, timeout_sec=0.5)
m = got['img']
img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else '/tmp/debug.png', img)
f = got['face']
print('detected', f.detected, 'bbox', np.round(f.bbox, 1).tolist(), 'pcam',
      [round(f.position_camera.x, 3), round(f.position_camera.y, 3), round(f.position_camera.z, 3)])
