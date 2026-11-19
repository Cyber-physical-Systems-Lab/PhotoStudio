"""USB webcam node (OpenCV). Publishes /camera/image_raw (bgr8) and /camera/camera_info.

The camera is read at the full capture resolution (4K on the BRIO); the stream used for face
tracking and framing is published downscaled (publish_width x publish_height, 1080p) so perception
stays fast, and /camera/save_still writes the latest full-resolution frame for the actual shots.
Kept deliberately small so a DSLR node (gphoto2) can replace it with the same topics.
"""
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from portrait_interfaces.srv import SaveStill
from sensor_msgs.msg import CameraInfo, Image


def load_calibration(path):
    fs = cv2.FileStorage(os.path.expanduser(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f'cannot read calibration {path}')
    K = fs.getNode('camera_matrix').mat()
    D = fs.getNode('distortion_coefficients').mat().ravel()
    w = int(fs.getNode('image_width').real())
    h = int(fs.getNode('image_height').real())
    return K, D, w, h


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 3840)            # capture resolution (stills)
        self.declare_parameter('height', 2160)
        self.declare_parameter('publish_width', 1920)    # streamed resolution (tracking, framing)
        self.declare_parameter('publish_height', 1080)
        self.declare_parameter('still_jpeg_quality', 97)
        self.declare_parameter('fps', 30)
        self.declare_parameter('fourcc', 'MJPG')
        self.declare_parameter('hfov_deg', 70.0)
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('frame_id', 'camera_optical_frame')
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        dev = g('device')
        self.cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open camera {dev}')
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*g('fourcc')))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, g('width'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, g('height'))
        self.cap.set(cv2.CAP_PROP_FPS, g('fps'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ch = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w, h = min(g('publish_width'), cw), min(g('publish_height'), ch)
        self.pub_size = None if (w, h) == (cw, ch) else (w, h)
        self.get_logger().info(f'Camera {dev} opened at {cw}x{ch} (stills); streaming {w}x{h}')
        if cw < g('width'):
            self.get_logger().warn(f'asked for {g("width")}x{g("height")} stills but got {cw}x{ch} '
                                   '(USB 2 port or cable?)')
        self.full_lock = threading.Lock()
        self.full = None            # (frame, wall time) at capture resolution
        self.jpeg_q = int(g('still_jpeg_quality'))

        self.info = CameraInfo()
        self.info.width, self.info.height = w, h
        self.info.distortion_model = 'plumb_bob'
        if g('calibration_file'):
            K, D, cw, ch = load_calibration(g('calibration_file'))
            if (cw, ch) != (w, h):  # scale intrinsics to the running resolution
                K = K.copy()
                K[0] *= w / cw
                K[1] *= h / ch
            self.info.d = [float(v) for v in D[:5]] + [0.0] * max(0, 5 - len(D))
        else:
            f = (w / 2) / math.tan(math.radians(g('hfov_deg')) / 2)
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            self.info.d = [0.0] * 5
            self.get_logger().warn('No calibration file: using nominal intrinsics from hfov_deg')
        self.info.k = [float(v) for v in K.ravel()]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [float(v) for v in (K[0, 0], 0.0, K[0, 2], 0.0, 0.0, K[1, 1], K[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0)]
        self.frame_id = g('frame_id')

        self.img_pub = self.create_publisher(Image, '/camera/image_raw', 2)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 2)
        self.create_service(SaveStill, '/camera/save_still', self.on_save_still)
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        fails = 0
        while self.running and rclpy.ok():
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails % 30 == 1:
                    self.get_logger().warn('camera read failed')
                time.sleep(0.03)
                continue
            fails = 0
            with self.full_lock:
                self.full = (frame, time.time())
            if self.pub_size is not None:
                frame = cv2.resize(frame, self.pub_size, interpolation=cv2.INTER_AREA)
            stamp = self.get_clock().now().to_msg()
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = 'bgr8'
            msg.step = msg.width * 3
            msg.data = frame.tobytes()
            self.img_pub.publish(msg)
            self.info.header.stamp = stamp
            self.info.header.frame_id = self.frame_id
            self.info_pub.publish(self.info)

    def on_save_still(self, req, res):
        t_end = time.time() + 2.0
        while True:
            with self.full_lock:
                full = self.full
            if full is not None and full[1] > req.not_before:
                break
            if time.time() > t_end:
                res.success, res.message = False, 'no new frame within 2 s'
                return res
            time.sleep(0.005)
        img, t = full
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q] if req.path.lower().endswith(('.jpg', '.jpeg')) else []
        ok = cv2.imwrite(req.path, img, params)
        res.success, res.message = bool(ok), ('saved' if ok else f'cannot write {req.path}')
        res.stamp, res.width, res.height = t, img.shape[1], img.shape[0]
        return res

    def destroy_node(self):
        self.running = False
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
