"""Face perception node.

In : /camera/image_raw, /camera/camera_info, /portrait/camera_pose
Out: /portrait/face (FaceObservation), /portrait/debug_image (annotated, half size)
"""
import threading

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from portrait_interfaces.msg import FaceObservation
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from .face import FaceAnalyzer


def image_to_bgr(msg: Image) -> np.ndarray:
    buf = np.frombuffer(msg.data, np.uint8)
    if msg.encoding in ('bgr8', 'rgb8'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if msg.encoding == 'rgb8' else img
    if msg.encoding in ('bgra8', 'rgba8'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if msg.encoding == 'rgba8' else cv2.COLOR_BGRA2BGR)
    if msg.encoding == 'mono8':
        return cv2.cvtColor(buf.reshape(msg.height, msg.step)[:, :msg.width], cv2.COLOR_GRAY2BGR)
    raise ValueError(f'unsupported encoding {msg.encoding}')


def bgr_to_image(img: np.ndarray, header) -> Image:
    m = Image()
    m.header = header
    m.height, m.width = img.shape[:2]
    m.encoding = 'bgr8'
    m.step = m.width * 3
    m.data = np.ascontiguousarray(img).tobytes()
    return m


def quat_from_matrix(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return float(x), float(y), float(z), float(w)


def pose_to_tf(p) -> np.ndarray:
    x, y, z, w = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.declare_parameter('detect_width', 640)
        self.declare_parameter('score_threshold', 0.7)
        self.declare_parameter('interocular_m', 0.063)
        self.declare_parameter('max_rate_hz', 15.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.analyzer = FaceAnalyzer(None, g('detect_width'), g('score_threshold'), g('interocular_m'))
        self.K = None
        self.D = None
        self.pose_buf = []
        self.lock = threading.Lock()
        self.latest = None
        self.event = threading.Event()
        self.period = 1.0 / g('max_rate_hz')

        # reliable (default) QoS: large images are dropped under best-effort
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 2)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, 2)
        self.create_subscription(PoseStamped, '/portrait/camera_pose', self.on_pose, 10)
        self.face_pub = self.create_publisher(FaceObservation, '/portrait/face', 10)
        self.dbg_pub = self.create_publisher(Image, '/portrait/debug_image', 2)
        threading.Thread(target=self.worker, daemon=True).start()

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.D = np.array(msg.d, float) if len(msg.d) else np.zeros(5)

    def on_pose(self, msg: PoseStamped):
        # Keep ~3 s of camera poses so every image is transformed with the pose at its own stamp
        # (the arm may have moved between exposure and processing).
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pose_buf.append((t, pose_to_tf(msg.pose)))
        if len(self.pose_buf) > 200:
            del self.pose_buf[:len(self.pose_buf) - 200]

    def pose_at(self, stamp):
        t = stamp.sec + stamp.nanosec * 1e-9
        if not self.pose_buf:
            return None
        tb, T = min(self.pose_buf, key=lambda e: abs(e[0] - t))
        return T if abs(tb - t) < 0.2 else None

    def on_image(self, msg: Image):
        with self.lock:
            if self.latest is not None and self.latest.header.stamp == msg.header.stamp:
                return
            self.latest = msg
        self.event.set()

    def worker(self):
        import time
        while rclpy.ok():
            if not self.event.wait(0.5):
                continue
            self.event.clear()
            with self.lock:
                msg, self.latest = self.latest, None
            if msg is None or self.K is None:
                continue
            t0 = time.time()
            try:
                self.process(msg)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'perception failed: {e}')
            dt = time.time() - t0
            if dt < self.period:
                time.sleep(self.period - dt)

    def process(self, msg: Image):
        bgr = image_to_bgr(msg)
        T_cam = self.pose_at(msg.header.stamp)
        face = self.analyzer.analyze(bgr, self.K, self.D)
        out = FaceObservation()
        out.header = msg.header
        out.image_width, out.image_height = msg.width, msg.height
        dbg = cv2.resize(bgr, (msg.width // 2, msg.height // 2))
        if face is not None and face.position is not None:
            out.detected = True
            out.score = face.score
            out.bbox = [float(v) for v in face.bbox]
            bx, by, bw, bh = face.bbox
            m = 0.02 * max(msg.width, msg.height)
            out.complete = bool(bx > m and by > m and bx + bw < msg.width - m and by + bh < msg.height - m)
            out.landmarks = [float(v) for v in face.landmarks.ravel()]
            out.position_camera.x, out.position_camera.y, out.position_camera.z = (float(v) for v in face.position)
            out.yaw_deg, out.pitch_deg, out.roll_deg = face.yaw, face.pitch, face.roll
            out.sharpness = face.sharpness
            if T_cam is not None:
                pb = T_cam @ np.append(face.position, 1.0)
                out.has_base_position = True
                out.camera_pose.position.x, out.camera_pose.position.y, out.camera_pose.position.z = \
                    (float(v) for v in T_cam[:3, 3])
                qx, qy, qz, qw = quat_from_matrix(T_cam[:3, :3])
                out.camera_pose.orientation.x, out.camera_pose.orientation.y = qx, qy
                out.camera_pose.orientation.z, out.camera_pose.orientation.w = qz, qw
                out.position_base.x, out.position_base.y, out.position_base.z = (float(v) for v in pb[:3])
            x, y, w, h = (face.bbox / 2).astype(int)
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (u, v) in (face.landmarks / 2).astype(int):
                cv2.circle(dbg, (u, v), 3, (0, 0, 255), -1)
            cv2.putText(dbg, f'd={face.position[2]:.2f}m yaw={face.yaw:+.0f} pitch={face.pitch:+.0f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(dbg, 'no face', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        h2, w2 = dbg.shape[:2]
        cv2.line(dbg, (w2 // 2, 0), (w2 // 2, h2), (255, 255, 0), 1)
        self.face_pub.publish(out)
        self.dbg_pub.publish(bgr_to_image(dbg, msg.header))


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
