"""Portrait session state machine.

    IDLE -> GREET -> SEARCH -> FRAME -> GUIDE -> SHOOT (x poses) -> DONE -> IDLE

ROS only captures: raw shots + session.json + capture_complete.json go to
<output_dir>/<name>/<stamp>_raw/. The portrait itself is made by the separate, non-ROS
post-processing service (postprocess/, `python -m portrait_post.watch`).
                          \\________ face lost: STOP arm, ask to come back, reacquire ________/

Start a session:  ros2 run portrait_session start --name "Ada Lovelace"
             or:  ros2 service call /portrait/start portrait_interfaces/srv/StartSession "{name: Ada}"
Abort:            ros2 service call /portrait/abort std_srvs/srv/Trigger
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from portrait_interfaces.msg import ArmStatus, FaceObservation
from portrait_interfaces.srv import MoveCamera, MoveNamed, SaveStill, Say, StartSession
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .marker import MarkerLocator, eyes_behind_marker, face_scale, stable_point

from .framing import FaceTracker, framing_error, look_at, pose_correction, position_correction, shot_pose


class Abort(Exception):
    pass


class FaceLost(Exception):
    pass


class Unreachable(Exception):
    pass


def quat_from_matrix(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def image_to_bgr(msg: Image) -> np.ndarray:
    buf = np.frombuffer(msg.data, np.uint8)
    ch = 4 if msg.encoding in ('bgra8', 'rgba8') else 3
    img = buf.reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    if msg.encoding == 'rgb8':
        return img[:, :, ::-1].copy()
    if msg.encoding == 'rgba8':
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if msg.encoding == 'bgra8':
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def median_face(obs: list) -> FaceObservation:
    """Median of several FaceObservations (positions and head angles); pixels from the last one."""
    out = FaceObservation()
    last = obs[-1]
    out.header, out.detected, out.score = last.header, True, last.score
    out.bbox, out.landmarks = last.bbox, last.landmarks
    out.camera_pose = last.camera_pose
    out.complete = all(o.complete for o in obs)
    out.image_width, out.image_height = last.image_width, last.image_height
    out.sharpness = last.sharpness
    for fld in ('position_camera', 'position_base'):
        arr = np.median([[getattr(o, fld).x, getattr(o, fld).y, getattr(o, fld).z] for o in obs], axis=0)
        p = getattr(out, fld)
        p.x, p.y, p.z = (float(v) for v in arr)
    out.has_base_position = all(o.has_base_position for o in obs)
    out.yaw_deg = float(np.median([o.yaw_deg for o in obs]))
    out.pitch_deg = float(np.median([o.pitch_deg for o in obs]))
    out.roll_deg = float(np.median([o.roll_deg for o in obs]))
    return out


def slug(name: str) -> str:
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', name.strip()).strip('_')
    return s or 'guest'


class SessionNode(Node):
    def __init__(self):
        super().__init__('session_node')
        from .single import ensure_single
        ensure_single(self)
        P = {
            'output_dir': '~/Nova5Photo/portraits', 'language': 'en', 'standoff_m': 0.65,
            'camera_above_eyes_m': 0.03, 'eye_line_fraction': 0.38,
            'shot_azimuths_deg': [20.0, 30.0, 10.0], 'shot_counts': [2, 2, 1], 'shot_interval_s': 0.6,
            'framing_tolerance': 0.05, 'max_framing_iterations': 3, 'yaw_tolerance_deg': 10.0,
            'pitch_tolerance_deg': 12.0, 'max_corrections': 3, 'face_lost_timeout_s': 1.0,
            'reacquire_timeout_s': 12.0, 'search_poses': ['home', 'scan_left', 'scan_right', 'scan_high'],
            'auto_start_name': '', 'mark_position': [1.0, 0.0], 'mark_tolerance_m': [0.10, 0.12],
            'settle_s': 0.4, 'observations_per_measure': 3, 'frame_timeout_s': 2.0,
            'marker.enabled': False, 'marker.id': 4, 'marker.size_m': 0.0315,
            'marker.dictionary': 'DICT_APRILTAG_36h11', 'marker.timeout_s': 5.0, 'marker.frames': 5,
            'marker.eye_offset_m': 0.05,
            'expressions': ['smile', 'expr_big_smile', 'expr_thumbs_up'],
            'style': 'natural',           # natural | professional; set per place by calibrate_layout.py
            'mark_height': -1.0,          # calibrated face height (layout frame); < 0 = not checked
            'height_tolerance_m': 0.20,   # ask to sit down / sit up beyond this
        }
        for k, v in P.items():
            self.declare_parameter(k, v)
        self.locator = MarkerLocator(self.p('marker.id'), self.p('marker.size_m'), self.p('marker.dictionary'))
        cb = ReentrantCallbackGroup()
        self.cli_cam = self.create_client(MoveCamera, '/arm/move_camera', callback_group=cb)
        self.cli_named = self.create_client(MoveNamed, '/arm/move_named', callback_group=cb)
        self.cli_stop = self.create_client(Trigger, '/arm/stop', callback_group=cb)
        self.cli_say = self.create_client(Say, '/voice/say', callback_group=cb)
        self.cli_still = self.create_client(SaveStill, '/camera/save_still', callback_group=cb)
        self.create_subscription(FaceObservation, '/portrait/face', self.on_face, 10, callback_group=cb)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 2,
                                 callback_group=cb)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, 2, callback_group=cb)
        self.create_subscription(PoseStamped, '/portrait/camera_pose', self.on_pose, 10, callback_group=cb)
        self.create_subscription(ArmStatus, '/arm/status', self.on_status, 10, callback_group=cb)
        self.create_service(StartSession, '/portrait/start', self.on_start, callback_group=cb)
        self.create_service(Trigger, '/portrait/abort', self.on_abort, callback_group=cb)
        self.state_pub = self.create_publisher(String, '/portrait/session_state', 10)
        self.result_pub = self.create_publisher(String, '/portrait/result', 10)
        self.name_pub = self.create_publisher(String, '/voice/set_name', 10)
        self.lang_pub = self.create_publisher(String, '/voice/set_language', 10)

        self.lock = threading.Lock()
        self.face: FaceObservation | None = None
        self.face_rx = 0.0           # wall time of the last detection
        self.face_any_rx = 0.0       # wall time of the last processed frame
        self.last_detected = False   # did the last processed frame contain a face
        self.frame: Image | None = None
        self.frame_rx = 0.0
        self.info: CameraInfo | None = None
        self.cam_pose = None
        self.arm: ArmStatus | None = None
        self.thread: threading.Thread | None = None
        self.abort_flag = False
        self.watch_face = False
        self.face_lost = False
        self.state = 'IDLE'
        self.tracker = FaceTracker()
        self.create_timer(0.1, self.monitor, callback_group=cb)
        self.publish_state('IDLE')
        if self.p('auto_start_name'):
            threading.Thread(target=self.auto_start, daemon=True).start()

    def p(self, n):
        return self.get_parameter(n).value

    # ------------------------------------------------------------------ inputs
    def on_face(self, m: FaceObservation):
        with self.lock:
            self.face_any_rx = time.time()
            self.last_detected = bool(m.detected)
            if m.detected:
                self.face = m
                self.face_rx = time.time()

    def on_image(self, m: Image):
        with self.lock:
            self.frame = m
            self.frame_rx = time.time()

    def on_info(self, m):
        self.info = m

    def on_pose(self, m):
        self.cam_pose = m

    def on_status(self, m):
        self.arm = m

    # ------------------------------------------------------------------ services
    def on_start(self, req, res):
        if self.thread is not None and self.thread.is_alive():
            res.accepted, res.message = False, f'busy ({self.state})'
            return res
        ready, why = self.ready()
        if not ready:
            res.accepted, res.message = False, f'not ready: {why}'
            return res
        self.abort_flag = False
        self.thread = threading.Thread(target=self.run_session, args=(req.name or 'guest',
                                       req.language or self.p('language')), daemon=True)
        self.thread.start()
        res.accepted, res.message = True, 'session started'
        return res

    def on_abort(self, req, res):
        self.abort_flag = True
        self.stop_arm()
        res.success, res.message = True, 'aborting'
        return res

    def auto_start(self):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < 300:
            ok, why = self.ready()
            if ok:
                time.sleep(2.0)
                self.get_logger().info(f'auto start for {self.p("auto_start_name")}')
                self.abort_flag = False
                self.thread = threading.Thread(target=self.run_session, args=(
                    self.p('auto_start_name'), self.p('language')), daemon=True)
                self.thread.start()
                return
            time.sleep(1.0)
        self.get_logger().error(f'auto start gave up: {why}')

    def ready(self):
        for c, n in ((self.cli_cam, 'arm'), (self.cli_named, 'arm'), (self.cli_say, 'voice')):
            if not c.service_is_ready():
                return False, f'{n} service not available'
        if self.arm is None or not self.arm.connected:
            return False, 'arm not connected'
        if self.info is None or self.frame is None:
            return False, 'no camera images'
        if time.time() - self.face_any_rx > 2.0:
            return False, 'perception not running'
        return True, ''

    # ------------------------------------------------------------------ helpers
    def publish_state(self, s, detail=''):
        self.state = s
        self.state_pub.publish(String(data=json.dumps({'state': s, 'detail': detail})))
        self.get_logger().info(f'[{s}] {detail}')

    def call(self, client, req, timeout):
        fut = client.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                raise Abort(f'{client.srv_name} timed out')
            time.sleep(0.01)
        return fut.result()

    def say(self, key='', text='', wait=True):
        r = Say.Request()
        r.key, r.text, r.wait = key, text, wait
        self.call(self.cli_say, r, 60.0)

    def stop_arm(self):
        if self.cli_stop.service_is_ready():
            self.cli_stop.call_async(Trigger.Request())

    def check_abort(self):
        if self.abort_flag:
            raise Abort('aborted by operator')
        if self.face_lost:
            raise FaceLost()

    def move_named(self, name):
        self.check_abort()
        r = MoveNamed.Request()
        r.name = name
        res = self.call(self.cli_named, r, 120.0)
        self.check_abort()
        if not res.success:
            raise Abort(f'move {name} failed: {res.message}')

    def move_camera(self, T, require_face=True):
        self.check_abort()
        r = MoveCamera.Request()
        r.target = Pose()
        r.target.position.x, r.target.position.y, r.target.position.z = (float(v) for v in T[:3, 3])
        qx, qy, qz, qw = quat_from_matrix(T[:3, :3])
        r.target.orientation.x, r.target.orientation.y = float(qx), float(qy)
        r.target.orientation.z, r.target.orientation.w = float(qz), float(qw)
        r.require_face = require_face
        r.speed_scale = 1.0
        res = self.call(self.cli_cam, r, 120.0)
        self.check_abort()
        return res.success, res.message

    def fresh_face(self, after: float, timeout: float):
        """First detection received after wall time `after`."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.check_abort()
            with self.lock:
                if self.face is not None and self.face_rx > after and self.face.has_base_position:
                    return self.face
            time.sleep(0.03)
        return None

    def settled_face(self, timeout=3.0, recenter=True):
        """Wait for the arm to settle, then average a few complete detections (median).

        A face cut by the image border gives wrong landmarks; if that is all we see, turn the camera
        towards it first (direction is still reliable) and measure again.
        """
        time.sleep(self.p('settle_s'))
        obs, partial = [], None
        t_end = time.time() + timeout + 1.5 * self.p('observations_per_measure')
        after = time.time()
        while len(obs) < self.p('observations_per_measure') and time.time() < t_end:
            f = self.fresh_face(after, max(0.1, t_end - time.time()))
            if f is None:
                break
            after = self.face_rx
            if f.complete:
                obs.append(f)
            else:
                partial = f
        if not obs:
            if partial is not None and recenter:
                self.get_logger().info('face cut by the image border: turning the camera towards it')
                self.center_on(partial)
                return self.settled_face(timeout, recenter=False)
            return None
        f = median_face(obs)
        if f.has_base_position:
            # Correct the size-dependent single-view distance with rays from several arm poses.
            c = f.camera_pose
            R = quat_to_matrix(c.orientation)
            self.tracker.add([c.position.x, c.position.y, c.position.z], R,
                             [f.position_camera.x, f.position_camera.y, f.position_camera.z])
            p = self.tracker.position()
            f.position_base.x, f.position_base.y, f.position_base.z = (float(v) for v in p)
        return f

    # ------------------------------------------------------------------ marker (face distance)
    def measure_marker(self):
        """Ask for the tag in front of the eyes; its position in the camera frame (eyes), or None."""
        self.publish_state('MARKER')
        self.say('hold_marker')                       # the timeout starts once the prompt is spoken
        K = np.array(self.info.k, float).reshape(3, 3)
        D = np.array(self.info.d, float) if len(self.info.d) else np.zeros(5)
        pts, last_rx, found = [], None, None
        t_end = time.time() + self.p('marker.timeout_s')
        while time.time() < t_end and found is None:
            fr, rx = self.frame, self.frame_rx
            if fr is None or rx == last_rx:
                time.sleep(0.01)
                continue
            last_rx = rx
            p = self.locator.detect(image_to_bgr(fr), K, D)
            if p is not None:
                pts.append(p)
                found = stable_point(pts, self.p('marker.frames'))
        self.say('marker_done' if found is not None else 'marker_missed')
        if found is None:
            self.get_logger().warn(f'marker not measured ({len(pts)} detections in '
                                   f'{self.p("marker.timeout_s"):.0f} s): using the face size estimate')
            return None
        return eyes_behind_marker(found, self.p('marker.eye_offset_m'))

    def calibrate_with_marker(self, face, log):
        """Set this person's face scale from the marker (exact distance) before the arm moves."""
        if not self.p('marker.enabled'):
            return face
        eyes = self.measure_marker()
        if eyes is None:
            return face
        f = self.settled_face(3.0)                     # marker down: same camera pose, face visible
        if f is None or not f.has_base_position:
            return face
        pnp = float(np.linalg.norm([f.position_camera.x, f.position_camera.y, f.position_camera.z]))
        true = float(np.linalg.norm(eyes))
        s = face_scale(true, pnp)
        self.get_logger().info(f'marker: eyes at {true:.3f} m, face-size estimate {pnp:.3f} m -> scale '
                               f'{s if s is None else round(s, 3)}')
        log['marker'] = {'eyes_distance_m': true, 'face_estimate_m': pnp, 'scale': s}
        if s is None:
            return f
        self.tracker.set_scale(s)
        p = self.tracker.position()
        f.position_base.x, f.position_base.y, f.position_base.z = (float(v) for v in p)
        return f

    def center_on(self, f):
        """Keep the camera where it is and point it at the (partially visible) face."""
        c = f.camera_pose
        o = np.array([c.position.x, c.position.y, c.position.z])
        p = np.array([f.position_camera.x, f.position_camera.y, f.position_camera.z])
        d = quat_to_matrix(c.orientation) @ (p / max(1e-6, np.linalg.norm(p)))
        ok, msg = self.move_camera(look_at(o, o + 0.75 * d), require_face=False)
        if not ok:
            self.get_logger().warn(f'could not re-centre: {msg}')

    def monitor(self):
        """Stop the arm the moment the face disappears while we depend on it."""
        if not self.watch_face or self.face_lost:
            return
        if time.time() - self.face_rx > self.p('face_lost_timeout_s'):
            self.face_lost = True
            self.stop_arm()
            self.get_logger().warn('face lost: arm stopped')

    def face_pos(self, f: FaceObservation):
        return np.array([f.position_base.x, f.position_base.y, f.position_base.z])

    # ------------------------------------------------------------------ session
    def run_session(self, name, lang):
        self.tracker = FaceTracker()
        self.face_lost = False
        self.watch_face = False
        stamp = dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        out_root = os.path.join(os.path.expanduser(self.p('output_dir')), slug(name))
        raw_dir = os.path.join(out_root, f'{stamp}_raw')
        os.makedirs(raw_dir, exist_ok=True)
        log = {'name': name, 'started': stamp, 'events': [], 'shots': []}
        self.lang_pub.publish(String(data=lang))
        self.name_pub.publish(String(data=name.split()[0] if name.strip() else ''))
        time.sleep(0.2)
        try:
            self.publish_state('GREET', name)
            self.move_named('home')
            self.say('greeting')
            face = self.search()
            self.say('found')
            face = self.guide_position(face)
            face = self.calibrate_with_marker(face, log)
            style = self.p('style')
            log['style'] = style                  # post-processing picks its look from this
            self.say(f'style_{style}')
            self.watch_face = True
            self.shoot_all(face, raw_dir, log)
            self.watch_face = False
            self.publish_state('DONE')
            self.say('done', wait=False)
            self.move_named('home')
            # Hand over to the separate post-processing service (not ROS): it watches for this marker.
            log['complete'] = True
            log['face_scale'] = self.tracker.scale if self.tracker.scale_known else None
            self.write_log(raw_dir, log)
            with open(os.path.join(raw_dir, 'capture_complete.json'), 'w') as f:
                json.dump({'name': name, 'shots': len(log['shots']), 'raw_dir': raw_dir}, f)
            self.result_pub.publish(String(data=raw_dir))
            self.publish_state('IDLE', f'finished: {raw_dir}')
        except Abort as e:
            self.watch_face = False
            self.stop_arm()
            log['events'].append(f'abort: {e}')
            self.publish_state('ABORTED', str(e))
            try:
                self.face_lost = False
                self.abort_flag = False
                self.say('aborted')
                self.move_named('home')
            except Exception as e2:  # noqa: BLE001
                self.get_logger().error(f'could not return home: {e2}')
            self.publish_state('IDLE', 'after abort')
        finally:
            self.watch_face = False
            self.write_log(raw_dir, log)

    @staticmethod
    def write_log(raw_dir, log):
        with open(os.path.join(raw_dir, 'session.json'), 'w') as f:
            json.dump(log, f, indent=2, default=float)

    def search(self):
        self.publish_state('SEARCH')
        f = self.settled_face(2.5)
        if f is not None:
            return f
        self.say('searching')
        for _ in range(2):
            for pose in self.p('search_poses'):
                self.move_named(pose)
                f = self.settled_face(2.5)
                if f is not None:
                    return f
        raise Abort('no face found')

    def guide_position(self, face, force=False):
        """Ask the person to step towards the mark if they are far off it.

        The place was calibrated for a seated (or standing) face height: someone standing up at a
        seat is asked to sit down (the arm's poses were only checked around the calibrated height).
        """
        for _ in range(self.p('max_corrections')):
            mh = self.p('mark_height')
            dz = self.face_pos(face)[2] - mh
            if mh >= 0 and abs(dz) > self.p('height_tolerance_m'):
                self.get_logger().info(f'face {dz:+.2f} m from the calibrated height -> '
                                       f'{"sit_down" if dz > 0 else "sit_up"}')
                self.say('sit_down' if dz > 0 else 'sit_up')
                self.tracker.reset_position()
                time.sleep(2.5)                   # sitting down takes a moment
                f = self.settled_face(6.0)
                if f is not None:
                    face = f
                continue
            key = position_correction(self.face_pos(face), np.array(self.p('mark_position')),
                                      self.p('mark_tolerance_m'), force)
            self.get_logger().info(f'face at {np.round(self.face_pos(face), 2).tolist()} -> {key}')
            if key is None:
                return face
            self.say(key)
            self.tracker.reset_position()
            time.sleep(1.5)
            f = self.settled_face(4.0)
            if f is None:
                raise FaceLost()
            face, force = f, False
        return face

    def recover(self):
        """Face lost: the arm is already stopped.

        Our own move can hide the face (the arm or gripper in front of the lens, an odd wrist
        angle): first go back to the last pose the face was seen from. Only if the face is not
        there either, ask the subject to come back. Sets self.moved_back.
        """
        self.publish_state('LOST')
        self.watch_face = False
        self.face_lost = False
        self.moved_back = False
        T = getattr(self, 'last_good_T', None)
        if T is not None:
            ok, msg = self.move_camera(T, require_face=False)
            if ok:
                f = self.fresh_face(time.time() + self.p('settle_s'), 3.0)
                if f is not None:
                    self.get_logger().warn('face lost during our own move; back at the previous pose')
                    self.moved_back = True
                    f = self.settled_face(3.0) or f
                    self.watch_face = True
                    return f
            else:
                self.get_logger().warn(f'could not return to the previous pose: {msg}')
        self.say('lost')
        self.tracker.reset_position()
        f = self.fresh_face(time.time(), self.p('reacquire_timeout_s'))
        if f is None:
            raise Abort('subject did not come back')
        f = self.settled_face(4.0) or f
        self.say('found')
        self.watch_face = True
        return f

    def shoot_all(self, face, raw_dir, log):
        azs = list(self.p('shot_azimuths_deg'))
        counts = list(self.p('shot_counts'))
        guided = False
        for i, (az, n) in enumerate(zip(azs, counts)):
            for attempt in range(3):
                try:
                    face = self.frame_shot(face, az)
                    if not guided:
                        face = self.guide(face, az)
                        guided = True
                        self.say('look_at_camera')
                    # One expression / gesture cue per angle, just spoken (not checked).
                    expr = list(self.p('expressions'))
                    if expr:
                        self.say(expr[i % len(expr)])
                    self.say('countdown')
                    self.capture(raw_dir, i, az, n, log)
                    break
                except FaceLost:
                    log['events'].append(f'face lost at pose {i}')
                    face = self.recover()
                    if self.moved_back and attempt >= 1:
                        # The way to this angle keeps hiding the face: skip it.
                        self.get_logger().warn(f'skipping pose {i} (azimuth {az:+.0f}): face hidden on the way')
                        log['events'].append(f'pose {i} skipped: face hidden by the move')
                        break
                except Unreachable as e:
                    if attempt == 0:
                        self.get_logger().warn(f'pose {i} unreachable ({e}); asking the person to move')
                        # Sit still: only ask the person to move if they are off the seat; the
                        # arm has already tried other distances and angles in frame_shot.
                        moved = self.guide_position(face, force=False)
                        if moved is not face:
                            face = moved
                            continue
                    self.get_logger().warn(f'skipping pose {i} (azimuth {az:+.0f}): {e}')
                    log['events'].append(f'pose {i} unreachable: {e}')
                    break
            else:
                raise Abort('face kept getting lost')
            if i + 1 < len(azs):
                self.say('next_angle', wait=False)
        if not log['shots']:
            raise Abort('no shot pose was reachable; check the marked spot and the workspace box')

    def current_camera_T(self):
        m = self.cam_pose
        if m is None:
            return None
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(m.pose.orientation)
        T[:3, 3] = [m.pose.position.x, m.pose.position.y, m.pose.position.z]
        return T

    def frame_shot(self, face, az):
        self.publish_state('FRAME', f'azimuth {az:+.0f} deg')
        self.last_good_T = self.current_camera_T()     # the face is visible from here
        fy = self.info.k[4]
        H, W = self.info.height, self.info.width
        tol = self.p('framing_tolerance')
        for it in range(self.p('max_framing_iterations')):
            # Candidates differ in height, size and position: if the ideal pose is out of reach,
            # step back a little, then soften the angle, before giving up on this pose.
            so = self.p('standoff_m')
            msg = ''
            for standoff, a in ((so, az), (so + 0.10, az), (so, az / 2), (so + 0.10, 0.0)):
                T = shot_pose(self.face_pos(face), a, standoff, self.p('camera_above_eyes_m'),
                              self.p('eye_line_fraction'), fy, H)
                ok, msg = self.move_camera(T)
                if ok:
                    break
                if 'face not visible' in msg or 'face lost' in msg:
                    raise FaceLost()
            else:
                raise Unreachable(msg)
            f = self.settled_face(3.0)
            if f is None:
                raise FaceLost()
            face = f
            eye = np.array(face.landmarks[:4]).reshape(2, 2).mean(axis=0)
            du, dv = framing_error(eye, W, H, self.p('eye_line_fraction'))
            c = face.camera_pose.position
            dist = float(np.linalg.norm(self.face_pos(face) - np.array([c.x, c.y, c.z])))
            self.get_logger().info(f'framing it {it}: du={du:+.3f} dv={dv:+.3f} dist={dist:.2f} m '
                                   f'face {np.round(self.face_pos(face), 2).tolist()} scale {self.tracker.scale:.2f}')
            if abs(du) < tol and abs(dv) < tol and abs(dist - standoff) < 0.08:
                return face
        self.get_logger().warn('framing not within tolerance after max iterations, continuing')
        return face

    def guide(self, face, az):
        self.publish_state('GUIDE')
        for _ in range(self.p('max_corrections')):
            key = pose_correction(face.yaw_deg, face.pitch_deg, face.roll_deg, az,
                                  self.p('yaw_tolerance_deg'), self.p('pitch_tolerance_deg'))
            self.get_logger().info(f'head pose yaw={face.yaw_deg:+.1f} (want {az:+.0f}) '
                                   f'pitch={face.pitch_deg:+.1f} roll={face.roll_deg:+.1f} -> {key}')
            if key is None:
                self.say('perfect')
                return face
            self.say(key)
            time.sleep(1.5)
            f = self.settled_face(3.0)
            if f is None:
                raise FaceLost()
            face = f
        # head may have moved: re-frame once
        return self.frame_shot(face, az)

    def capture(self, raw_dir, pose_idx, az, n, log):
        self.publish_state('SHOOT', f'pose {pose_idx} azimuth {az:+.0f}, {n} shots')
        t_ready = time.time()
        for k in range(n):
            self.check_abort()
            t_end = time.time() + self.p('frame_timeout_s')
            while True:
                with self.lock:
                    fr, rx = self.frame, self.frame_rx
                    face = self.face
                if rx > t_ready:
                    break
                if time.time() > t_end:
                    raise Abort('no camera frames')
                time.sleep(0.01)
            # Only keep the shot if perception still sees the face on a frame at least this new.
            t_chk = time.time() + self.p('frame_timeout_s') + 1.0
            while True:
                with self.lock:
                    seen, det = self.face_any_rx, self.last_detected
                if seen > rx:
                    break
                if time.time() > t_chk:
                    raise FaceLost()
                time.sleep(0.02)
            if not det:
                raise FaceLost()
            base = f'shot_p{pose_idx}_{k}_az{int(round(az)):+03d}'
            size = None
            if self.cli_still.service_is_ready():
                # Full-resolution still (4K) from the camera node, taken after the checked frame.
                rq = SaveStill.Request()
                fn = base + '.jpg'
                rq.path, rq.not_before = os.path.join(raw_dir, fn), float(t_ready)
                r = self.call(self.cli_still, rq, 10.0)
                if not r.success:
                    raise Abort(f'still capture failed: {r.message}')
                size = [r.width, r.height]
            else:                                   # simulation: save the streamed frame
                fn = base + '.png'
                img = image_to_bgr(fr)
                cv2.imwrite(os.path.join(raw_dir, fn), img)
                size = [img.shape[1], img.shape[0]]
            log['shots'].append({
                'file': fn, 'pose': pose_idx, 'azimuth_deg': az, 'size': size,
                'face': None if face is None else {
                    'bbox': list(face.bbox), 'landmarks': list(face.landmarks),
                    'yaw': face.yaw_deg, 'pitch': face.pitch_deg, 'roll': face.roll_deg,
                    'sharpness': face.sharpness},
                'camera_pose': None if self.cam_pose is None else [
                    self.cam_pose.pose.position.x, self.cam_pose.pose.position.y, self.cam_pose.pose.position.z],
            })
            self.get_logger().info(f'saved {fn}')
            t_ready = time.time()
            time.sleep(self.p('shot_interval_s'))


def main():
    rclpy.init()
    node = SessionNode()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
