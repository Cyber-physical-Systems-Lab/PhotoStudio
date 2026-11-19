"""Nova 5 arm node: one interface for Isaac Sim, a fake arm and the real robot.

Services
  /arm/move_camera  portrait_interfaces/MoveCamera   camera optical-frame pose in base frame
  /arm/move_named   portrait_interfaces/MoveNamed    named joint pose from the config
  /arm/stop         std_srvs/Trigger                 stop now, hold position
Topics
  /joint_states            sensor_msgs/JointState   (for robot_state_publisher / RViz)
  /portrait/camera_pose    geometry_msgs/PoseStamped
  /arm/status              portrait_interfaces/ArmStatus
  /portrait/face (in)      portrait_interfaces/FaceObservation  (face watchdog + distance)

Safety (all backends): narrowed joint limits, camera workspace box, keep-out plane towards the
subject, minimum face distance along the whole path, joint and camera speed caps, largest joint
step per move, stop when the face is lost during a face-guided move, runtime monitor that stops
the arm if it ever leaves the box or exceeds the speed cap. Real robot with gripper.check: the
camera is held by a Robotiq 2F-85; the node only READS the gripper status (never commands it),
refuses to start unless it reports an object held, and stops the arm if that changes.

All poses (services, /portrait/camera_pose) are in the layout frame (layout_yaw_deg), +x towards
the subject; only the controller cross-check uses the raw robot base frame.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from portrait_interfaces.msg import ArmStatus, FaceObservation
from portrait_interfaces.srv import MoveCamera, MoveNamed
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from .kinematics import (Kinematics, layout_base, load_chain, make_tf, matrix_from_quat, quat_from_matrix,
                         rpy_to_matrix)
from .planner import plan_camera_pose
from .safety import SafetyChecker, SafetyLimits, min_jerk


# =============================================================================== backends
class StreamingBackend:
    """Interpolates on the PC and streams joint targets (Isaac Sim, or a purely fake arm)."""

    def __init__(self, node: 'ArmNode', fake: bool):
        self.node = node
        self.fake = fake
        self.lock = threading.Lock()
        self.q_meas = None
        self.q_meas_time = 0.0
        self.q_cmd = None
        self.traj = None          # (q0, q1, t0, T)
        self.name = 'fake' if fake else 'sim'
        if fake:
            self.q_cmd = node.named['home'].copy()
        else:
            self.pub = node.create_publisher(JointState, node.p('sim.command_topic'), 10)
            node.create_subscription(JointState, node.p('sim.state_topic'), self._on_state, 10)

    def _on_state(self, msg: JointState):
        try:
            idx = [msg.name.index(n) for n in self.node.kin.joint_names]
        except ValueError:
            return
        with self.lock:
            self.q_meas = np.array([msg.position[i] for i in idx])
            self.q_meas_time = time.time()
            if self.q_cmd is None:
                self.q_cmd = self.q_meas.copy()

    def connect(self):
        return True

    @property
    def connected(self) -> bool:
        return self.fake or (time.time() - self.q_meas_time < 0.5)

    def joints(self):
        with self.lock:
            if self.fake:
                return None if self.q_cmd is None else self.q_cmd.copy()
            if time.time() - self.q_meas_time > 0.5:
                return None
            return self.q_meas.copy()

    def velocities(self):
        return None

    def start_move(self, q0, q1, duration):
        with self.lock:
            self.traj = (np.array(q0), np.array(q1), time.time(), duration)

    def tick(self):
        with self.lock:
            if self.traj is not None:
                q0, q1, t0, T = self.traj
                s = min_jerk((time.time() - t0) / T)
                self.q_cmd = q0 + s * (q1 - q0)
            q_cmd = None if self.q_cmd is None else self.q_cmd.copy()
        if q_cmd is not None and not self.fake:
            msg = JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = list(self.node.kin.joint_names)
            msg.position = [float(v) for v in q_cmd]
            self.pub.publish(msg)

    def done(self) -> bool:
        with self.lock:
            if self.traj is None:
                return True
            q0, q1, t0, T = self.traj
            if time.time() - t0 < T:
                return False
            q = self.q_cmd if self.fake else self.q_meas
            if q is not None and np.max(np.abs(q - q1)) < math.radians(0.5):
                self.traj = None
                return True
            return False

    def stop(self):
        with self.lock:
            hold = self.q_cmd if self.fake or self.q_meas is None else self.q_meas
            self.q_cmd = None if hold is None else hold.copy()
            self.traj = None

    def status(self):
        return True, ''

    def shutdown(self):
        pass


class RealBackend:
    """Dobot Nova 5 over TCP/IP (V3 protocol)."""

    def __init__(self, node: 'ArmNode'):
        from .dobot_v3 import MODE_IDLE, MODE_NAMES, DobotV3Client
        self.node = node
        self.name = 'real'
        self.MODE_IDLE, self.MODE_NAMES = MODE_IDLE, MODE_NAMES
        self.client = DobotV3Client(node.p('real.robot_ip'), log=lambda s: node.get_logger().debug(s),
                                    warn=lambda s: node.get_logger().warn(s))
        self.goal = None
        self.move_t0 = 0.0
        self._connected = False
        self.grip_ok, self.grip_msg = True, ''
        self._grip_index = None
        self._grip_running = False

    def connect(self):
        n = self.node
        n.get_logger().info(f'Connecting to Nova 5 at {n.p("real.robot_ip")} ...')
        self.client.connect()
        self.client.prepare(n.p('real.speed_factor'), n.p('real.collision_level'))
        self._connected = True
        if n.p('real.verify_kinematics'):
            self.verify_kinematics()
        if n.p('gripper.check'):
            self.start_gripper_watch()
        return True

    # Robotiq 2F-85 on the tool RS-485, reached through the controller's Modbus bridge.
    def read_gripper(self):
        import re
        r = self.client.dash.call(f'GetInRegs({self._grip_index},2000,3)')
        v = [int(x) for x in re.search(r'\{(.*?)\}', r).group(1).split(',')]
        st = v[0] >> 8
        return {'gACT': st & 1, 'gSTA': (st >> 4) & 3, 'gOBJ': (st >> 6) & 3,
                'fault': v[1] >> 8, 'pos': v[2] >> 8}

    def start_gripper_watch(self):
        import re
        n = self.node
        # Close Modbus links left open by earlier tools/crashes (the controller has only a few, and stale
        # ones were seen together with bursts of failed reads), then open ours.
        for i in range(5):
            self.client.dash.call(f'ModbusClose({i})', check=False)
        r = self.client.dash.call(f'ModbusCreate(127.0.0.1,60000,{int(n.p("gripper.slave_id"))},1)')
        self._grip_index = int(re.search(r'\{(\d+)\}', r).group(1))
        g = self.read_gripper()
        if not (g['gACT'] and g['gSTA'] == 3 and g['gOBJ'] == 2):
            raise RuntimeError(f'gripper is not holding the camera ({g}); grasp it before starting')
        n.get_logger().info(f'Gripper holds the camera (position {g["pos"]}); watching it at 4 Hz')
        self._grip_running = True
        threading.Thread(target=self._gripper_loop, daemon=True).start()

    def _gripper_loop(self):
        fails = 0
        while self._grip_running:
            try:
                g = self.read_gripper()
                fails = 0
                ok = g['gOBJ'] == 2 and g['gACT'] == 1        # grip reported lost: stop at once
                self.grip_ok, self.grip_msg = ok, '' if ok else f'gripper lost the camera ({g})'
            except Exception as e:  # noqa: BLE001
                # Modbus reads fail now and then, sometimes in short bursts: stop only if the status
                # stays unreadable for 8 reads in a row (~2 s). A reported lost grip stops at once.
                fails += 1
                self.node.get_logger().warn(f'gripper status read failed ({fails}/8): {e}')
                if fails >= 8:
                    self.grip_ok, self.grip_msg = False, f'gripper status unreadable: {e}'
            if not self.grip_ok and self.node.moving:
                self.node.request_stop(self.grip_msg)
            time.sleep(0.25)

    def verify_kinematics(self):
        """Refuse to move if our URDF FK and the controller disagree on the flange position."""
        n = self.node
        time.sleep(0.2)
        q = self.client.joints_rad()
        tv = self.client.tool_vector()
        if q is None or tv is None:
            raise RuntimeError('no feedback to verify kinematics')
        p_ctrl = tv[0]
        p_fk = n.kin_flange.fk(q)[:3, 3]
        err = float(np.linalg.norm(p_ctrl - p_fk))
        n.get_logger().info(
            f'Kinematics check: controller flange {np.round(p_ctrl, 4).tolist()} m, URDF '
            f'{np.round(p_fk, 4).tolist()} m, difference {err * 1000:.1f} mm')
        if err > n.p('real.verify_tolerance_m'):
            raise RuntimeError(
                f'URDF and controller disagree by {err * 1000:.1f} mm. Check Tool(0)/User(0), the base '
                'frame offset and the URDF before moving the real arm.')

    @property
    def connected(self):
        return self._connected and self.client.feedback() is not None

    def joints(self):
        return self.client.joints_rad()

    def velocities(self):
        fb = self.client.feedback()
        return None if fb is None else np.radians(np.array(fb['qd_actual'], float))

    def start_move(self, q0, q1, duration):
        self.goal = np.array(q1)
        self.move_t0 = time.time()
        self.client.joint_movj(q1, self.node.p('real.speed_j'), self.node.p('real.acc_j'))

    def tick(self):
        pass

    def done(self):
        if self.goal is None:
            return True
        q = self.client.joints_rad()
        if q is None or time.time() - self.move_t0 < 0.3:
            return False
        if self.client.robot_mode() == self.MODE_IDLE and np.max(np.abs(q - self.goal)) < math.radians(0.3):
            self.goal = None
            return True
        return False

    def stop(self):
        self.goal = None
        self.client.stop()

    def status(self):
        fb = self.client.feedback()
        if fb is None:
            return False, 'no feedback'
        mode = int(fb['robot_mode'])
        if mode in (9, 11) or int(fb['error_status']):
            return False, f'robot mode {self.MODE_NAMES.get(mode, mode)}'
        if not self.grip_ok:
            return False, self.grip_msg
        return True, self.MODE_NAMES.get(mode, str(mode))

    def shutdown(self):
        self._grip_running = False
        try:
            self.client.stop()
            if self._grip_index is not None:
                self.client.dash.call(f'ModbusClose({self._grip_index})', check=False)
        finally:
            self.client.close()


# =============================================================================== node
class ArmNode(Node):
    def __init__(self):
        super().__init__('arm_node', automatically_declare_parameters_from_overrides=True,
                         allow_undeclared_parameters=True)
        defaults = {
            'backend': 'sim', 'rate_hz': 50.0,
            'camera_mount_xyz': [0.0, -0.06, 0.05], 'camera_mount_rpy_deg': [0.0, 0.0, 0.0],
            'urdf_path': '',
            'nominal_face_position': [1.0, 0.0, 0.78], 'layout_yaw_deg': 0.0,
            'gripper.check': False, 'gripper.slave_id': 9,
            'sim.command_topic': '/isaac/joint_command', 'sim.state_topic': '/isaac/joint_states',
            'real.robot_ip': '192.168.5.1', 'real.speed_factor': 10, 'real.speed_j': 30,
            'real.acc_j': 20, 'real.collision_level': 3, 'real.verify_kinematics': True,
            'real.verify_tolerance_m': 0.01,
        }
        for k, v in defaults.items():
            if not self.has_parameter(k):
                self.declare_parameter(k, v)
        from .single import ensure_single
        ensure_single(self)

        urdf = self.p('urdf_path') or (get_package_share_directory('portrait_description')
                                       + '/urdf/nova5_portrait.urdf')
        chain = load_chain(urdf, 'base_link', 'tool_link')
        lower = np.radians(self.p('joint_lower_deg')) if self.has_parameter('joint_lower_deg') else None
        upper = np.radians(self.p('joint_upper_deg')) if self.has_parameter('joint_upper_deg') else None
        mount = make_tf(rpy_to_matrix(*np.radians(self.p('camera_mount_rpy_deg'))), self.p('camera_mount_xyz'))
        self.kin = Kinematics(chain, mount, lower, upper, base=layout_base(self.p('layout_yaw_deg')))
        self.kin_flange = Kinematics(chain, None, lower, upper)
        self.limits = SafetyLimits.from_params({k: self.p('safety.' + k) for k in (
            'box_min', 'box_max', 'link_x_max', 'link_z_min', 'min_face_distance', 'max_cart_speed',
            'max_joint_speed_deg', 'max_joint_step_deg', 'face_timeout')})
        self.safety = SafetyChecker(self.kin, self.limits)
        self.nominal_face = np.array(self.p('nominal_face_position'), float)

        # Named poses, all validated now.
        self.named = {}
        for name in self.p('named_poses.names'):
            q = np.radians(self.p(f'named_poses.{name}'))
            ok, why = self.safety.check_config(q, self.nominal_face)
            if not ok:
                raise RuntimeError(f'named pose "{name}" is unsafe: {why}')
            self.named[name] = q

        backend = self.p('backend')
        if backend == 'real':
            self.backend = RealBackend(self)
        elif backend in ('sim', 'fake'):
            self.backend = StreamingBackend(self, fake=(backend == 'fake'))
        else:
            raise RuntimeError(f'unknown backend {backend}')
        self.get_logger().info(f'Arm backend: {backend}')

        self.face_pos = None
        self.face_time = 0.0
        self.move_lock = threading.Lock()
        self.abort_reason = None
        self.moving = False
        self.require_face = False
        self.fault = ''

        svc = ReentrantCallbackGroup()
        self.create_service(MoveCamera, '/arm/move_camera', self.on_move_camera, callback_group=svc)
        self.create_service(MoveNamed, '/arm/move_named', self.on_move_named, callback_group=svc)
        self.create_service(Trigger, '/arm/stop', self.on_stop, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(FaceObservation, '/portrait/face', self.on_face, 10,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/portrait/camera_pose', 10)
        self.status_pub = self.create_publisher(ArmStatus, '/arm/status', 10)
        self.create_timer(1.0 / self.p('rate_hz'), self.tick, callback_group=MutuallyExclusiveCallbackGroup())

        self.backend.connect()

    def p(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ callbacks
    def on_face(self, msg: FaceObservation):
        if msg.detected and msg.has_base_position:
            self.face_pos = np.array([msg.position_base.x, msg.position_base.y, msg.position_base.z])
            self.face_time = time.time()

    def face_for_checks(self):
        if self.face_pos is not None and time.time() - self.face_time < 5.0:
            return self.face_pos
        return self.nominal_face

    def on_stop(self, req, res):
        self.request_stop('stop requested')
        res.success = True
        res.message = 'stopped'
        return res

    def request_stop(self, reason: str):
        if self.moving:
            self.get_logger().warn(f'STOP: {reason}')
        self.abort_reason = reason
        self.backend.stop()

    def on_move_named(self, req, res):
        if req.name not in self.named:
            res.success, res.message = False, f'unknown pose {req.name}; known: {list(self.named)}'
            return res
        res.success, res.message = self.execute(self.named[req.name], require_face=False,
                                                free_move=False, speed_scale=1.0)
        return res

    def on_move_camera(self, req, res):
        q0 = self.backend.joints()
        if q0 is None:
            res.success, res.message = False, 'no joint feedback'
            return res
        p = req.target.position
        o = req.target.orientation
        T = make_tf(matrix_from_quat(o.x, o.y, o.z, o.w), [p.x, p.y, p.z])
        q1, why = plan_camera_pose(self.kin, self.safety, T, q0, self.named.values(), self.face_for_checks())
        if q1 is None:
            res.success, res.message = False, f'unreachable: {why}'
            return res
        scale = req.speed_scale if req.speed_scale > 0 else 1.0
        res.success, res.message = self.execute(q1, require_face=req.require_face, free_move=True,
                                                speed_scale=min(1.0, scale))
        return res

    # ------------------------------------------------------------------ motion
    def execute(self, q1, require_face: bool, free_move: bool, speed_scale: float):
        if not self.move_lock.acquire(timeout=0.1):
            return False, 'busy: another move is running'
        try:
            # Feedback can miss a beat (real controller): wait briefly for a fresh state.
            t_wait = time.time() + 1.5
            while True:
                ok, why = self.backend.status()
                q0 = self.backend.joints()
                if (ok and q0 is not None) or time.time() > t_wait:
                    break
                time.sleep(0.05)
            if not ok:
                return False, f'robot not ready: {why}'
            if q0 is None:
                return False, 'no joint feedback'
            if require_face and time.time() - self.face_time > self.limits.face_timeout:
                return False, 'face not visible'
            ok, why = self.safety.check_path(q0, q1, self.face_for_checks(), free_move=free_move)
            if not ok:
                self.get_logger().warn(f'Rejected move: {why}')
                return False, f'unsafe: {why}'
            T = self.safety.move_duration(q0, q1) / max(0.05, speed_scale)
            self.abort_reason = None
            self.require_face = require_face
            self.moving = True
            self.backend.start_move(q0, q1, T)
            deadline = time.time() + 3.0 * T + 5.0
            while rclpy.ok():
                if self.abort_reason:
                    return False, f'aborted: {self.abort_reason}'
                if self.backend.done():
                    return True, f'reached in {T:.1f} s'
                if time.time() > deadline:
                    self.request_stop('move timed out')
                    return False, 'timeout'
                time.sleep(0.02)
            return False, 'shutdown'
        except Exception as e:  # noqa: BLE001 - report every driver error to the caller
            self.request_stop(f'error: {e}')
            return False, f'error: {e}'
        finally:
            self.moving = False
            self.require_face = False
            self.move_lock.release()

    # ------------------------------------------------------------------ periodic
    def tick(self):
        self.backend.tick()
        q = self.backend.joints()
        now = self.get_clock().now().to_msg()
        st = ArmStatus()
        st.header.stamp = now
        st.backend = self.backend.name
        st.connected = bool(self.backend.connected)
        ok, msg = self.backend.status()
        st.enabled = ok
        st.moving = self.moving
        if q is None:
            st.message = 'waiting for joint feedback'
            self.status_pub.publish(st)
            return

        T = self.kin.fk(q)
        # Runtime monitor: backend fault, leaving the box (with margin) or overspeed -> stop.
        if self.moving and not ok:
            self.request_stop(msg)
        if self.moving:
            cam = T[:3, 3]
            margin = 0.02
            if np.any(cam < self.limits.box_min - margin) or np.any(cam > self.limits.box_max + margin):
                self.request_stop(f'camera left the workspace box at {np.round(cam, 3).tolist()}')
            qd = self.backend.velocities()
            if qd is not None and np.max(np.abs(qd)) > 1.5 * self.limits.max_joint_speed:
                self.request_stop(f'joint speed {math.degrees(np.max(np.abs(qd))):.0f} deg/s over the cap')
            if self.require_face and time.time() - self.face_time > self.limits.face_timeout:
                self.request_stop('face lost during move')

        js = JointState()
        js.header.stamp = now
        js.name = list(self.kin.joint_names)
        js.position = [float(v) for v in q]
        self.js_pub.publish(js)

        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = 'base_link'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in T[:3, 3])
        qx, qy, qz, qw = quat_from_matrix(T[:3, :3])
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        self.pose_pub.publish(ps)

        st.joints = js.position
        st.camera_pose = ps.pose
        st.fault = not ok
        st.message = msg or ('moving' if self.moving else 'idle')
        self.status_pub.publish(st)

    def destroy_node(self):
        try:
            self.backend.shutdown()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = ArmNode()
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(node)
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
