"""Minimal Dobot TCP/IP (V3 protocol) client for the Nova 5.

Written fresh for this project; the port layout, command names and the 1440-byte feedback
layout follow Dobot's TCP-IP-Python-V3 SDK and the dobot_bringup_v3 driver that the
TouchDrive project used on the same arm.

Ports: 29999 dashboard (enable, speed, errors, stop), 30003 motion queue (JointMovJ),
30004 realtime feedback (1440-byte packets every 8 ms).
"""
from __future__ import annotations

import math
import socket
import threading
import time

import numpy as np

FEEDBACK_DTYPE = np.dtype([
    ('len', np.int64), ('digital_input_bits', np.uint64), ('digital_output_bits', np.uint64),
    ('robot_mode', np.uint64), ('time_stamp', np.uint64), ('time_stamp_reserve_bit', np.uint64),
    ('test_value', np.uint64), ('test_value_keep_bit', np.float64), ('speed_scaling', np.float64),
    ('linear_momentum_norm', np.float64), ('v_main', np.float64), ('v_robot', np.float64),
    ('i_robot', np.float64), ('i_robot_keep_bit1', np.float64), ('i_robot_keep_bit2', np.float64),
    ('tool_accelerometer_values', np.float64, (3,)), ('elbow_position', np.float64, (3,)),
    ('elbow_velocity', np.float64, (3,)), ('q_target', np.float64, (6,)),
    ('qd_target', np.float64, (6,)), ('qdd_target', np.float64, (6,)),
    ('i_target', np.float64, (6,)), ('m_target', np.float64, (6,)),
    ('q_actual', np.float64, (6,)), ('qd_actual', np.float64, (6,)),
    ('i_actual', np.float64, (6,)), ('actual_TCP_force', np.float64, (6,)),
    ('tool_vector_actual', np.float64, (6,)), ('TCP_speed_actual', np.float64, (6,)),
    ('TCP_force', np.float64, (6,)), ('Tool_vector_target', np.float64, (6,)),
    ('TCP_speed_target', np.float64, (6,)), ('motor_temperatures', np.float64, (6,)),
    ('joint_modes', np.float64, (6,)), ('v_actual', np.float64, (6,)),
    ('hand_type', np.uint8, (4,)), ('user', np.uint8), ('tool', np.uint8),
    ('run_queued_cmd', np.uint8), ('pause_cmd_flag', np.uint8), ('velocity_ratio', np.uint8),
    ('acceleration_ratio', np.uint8), ('jerk_ratio', np.uint8), ('xyz_velocity_ratio', np.uint8),
    ('r_velocity_ratio', np.uint8), ('xyz_acceleration_ratio', np.uint8),
    ('r_acceleration_ratio', np.uint8), ('xyz_jerk_ratio', np.uint8), ('r_jerk_ratio', np.uint8),
    ('brake_status', np.uint8), ('enable_status', np.uint8), ('drag_status', np.uint8),
    ('running_status', np.uint8), ('error_status', np.uint8), ('jog_status', np.uint8),
    ('robot_type', np.uint8), ('drag_button_signal', np.uint8), ('enable_button_signal', np.uint8),
    ('record_button_signal', np.uint8), ('reappear_button_signal', np.uint8),
    ('jaw_button_signal', np.uint8), ('six_force_online', np.uint8), ('reserve2', np.uint8, (82,)),
    ('m_actual', np.float64, (6,)), ('load', np.float64), ('center_x', np.float64),
    ('center_y', np.float64), ('center_z', np.float64), ('user6', np.float64, (6,)),
    ('tool6', np.float64, (6,)), ('trace_index', np.float64), ('six_force_value', np.float64, (6,)),
    ('target_quaternion', np.float64, (4,)), ('actual_quaternion', np.float64, (4,)),
    ('reserve3', np.uint8, (24,)),
])
assert FEEDBACK_DTYPE.itemsize == 1440, FEEDBACK_DTYPE.itemsize
FEEDBACK_MAGIC = 0x0123456789ABCDEF

# robot_mode values (Dobot TCP/IP protocol)
MODE_NAMES = {1: 'init', 2: 'brake_open', 3: 'power_off', 4: 'disabled', 5: 'enabled_idle',
              6: 'backdrive', 7: 'running', 8: 'single_move', 9: 'error', 10: 'pause',
              11: 'collision'}
MODE_IDLE = 5
MODE_RUNNING = 7


class DobotError(RuntimeError):
    pass


class _CmdPort:
    def __init__(self, ip: str, port: int, timeout: float, log):
        self.ip, self.port, self.log = ip, port, log
        self.sock = socket.create_connection((ip, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.lock = threading.Lock()

    def call(self, cmd: str, check: bool = True) -> str:
        with self.lock:
            self.sock.sendall(cmd.encode('utf-8'))
            buf = b''
            while not buf.rstrip().endswith(b';'):
                chunk = self.sock.recv(1024)
                if not chunk:
                    raise DobotError(f'{self.ip}:{self.port} closed the connection')
                buf += chunk
        reply = buf.decode('utf-8', 'replace').strip()
        self.log(f'[{self.port}] {cmd} -> {reply}')
        if check:
            code = reply.split(',', 1)[0]
            if code.strip() != '0':
                raise DobotError(f'{cmd} failed: {reply}')
        return reply

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class DobotV3Client:
    def __init__(self, ip: str = '192.168.5.1', timeout: float = 3.0, log=print, warn=None):
        self.ip = ip
        self.timeout = timeout
        self.log = log
        self.warn = warn or log
        self.fb_port = 30004
        self.dash: _CmdPort | None = None
        self.move: _CmdPort | None = None
        self._fb_sock: socket.socket | None = None
        self._fb_thread: threading.Thread | None = None
        self._fb_lock = threading.Lock()
        self._fb = None
        self._fb_time = 0.0
        self._running = False

    # ---------------------------------------------------------------- connection
    def connect(self):
        self.dash = _CmdPort(self.ip, 29999, self.timeout, self.log)
        self.move = _CmdPort(self.ip, 30003, self.timeout, self.log)
        self._fb_sock = socket.create_connection((self.ip, self.fb_port), timeout=self.timeout)
        self._running = True
        self._fb_thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self._fb_thread.start()
        t0 = time.time()
        while self.feedback() is None:
            if time.time() - t0 > 3.0:
                raise DobotError('no feedback packets on port 30004')
            time.sleep(0.02)

    def close(self):
        self._running = False
        for p in (self.dash, self.move):
            if p:
                p.close()
        if self._fb_sock:
            try:
                self._fb_sock.close()
            except OSError:
                pass

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._fb_sock.recv(n - len(buf))
            if not chunk:
                raise DobotError('feedback socket closed')
            buf += chunk
        return bytes(buf)

    def _feedback_loop(self):
        magic = FEEDBACK_MAGIC.to_bytes(8, 'little')
        while self._running:
            try:
                pkt = self._recv_exact(1440)
                if pkt[48:56] != magic:
                    # Misaligned: resynchronise on the magic test_value (offset 48).
                    data = pkt + self._recv_exact(1440)
                    idx = data.find(magic)
                    if idx < 48:
                        continue
                    start = idx - 48
                    pkt = data[start:] + self._recv_exact(start)
                    if pkt[48:56] != magic:
                        continue
                fb = np.frombuffer(pkt, dtype=FEEDBACK_DTYPE)[0]
                with self._fb_lock:
                    self._fb = fb
                    self._fb_time = time.time()
            except (OSError, DobotError) as e:
                if not self._running:
                    break
                # The stream dropped: feedback() goes stale (the arm node then reports the robot as
                # not ready and stops any move); reconnect, but never resume motion from here.
                self.warn(f'feedback stream lost ({e}); reconnecting')
                self._reconnect_feedback()

    def _reconnect_feedback(self):
        try:
            self._fb_sock.close()
        except OSError:
            pass
        while self._running:
            time.sleep(0.5)
            try:
                self._fb_sock = socket.create_connection((self.ip, self.fb_port), timeout=self.timeout)
                self.warn('feedback stream reconnected')
                return
            except OSError as e:
                self.log(f'feedback reconnect failed: {e}')

    # ---------------------------------------------------------------- feedback
    def feedback(self, max_age: float = 0.5):
        with self._fb_lock:
            if self._fb is None or time.time() - self._fb_time > max_age:
                return None
            return self._fb

    def joints_rad(self):
        fb = self.feedback()
        return None if fb is None else np.radians(np.array(fb['q_actual'], float))

    def tool_vector(self):
        """(x, y, z) in metres and (rx, ry, rz) in degrees, as reported by the controller."""
        fb = self.feedback()
        if fb is None:
            return None
        v = np.array(fb['tool_vector_actual'], float)
        return v[:3] / 1000.0, v[3:]

    def robot_mode(self) -> int:
        fb = self.feedback()
        return -1 if fb is None else int(fb['robot_mode'])

    # ---------------------------------------------------------------- dashboard
    def prepare(self, speed_factor: int, collision_level: int | None):
        self.dash.call('ClearError()', check=False)
        self.dash.call('EnableRobot()')
        self.dash.call('User(0)', check=False)
        self.dash.call('Tool(0)', check=False)
        self.dash.call(f'SpeedFactor({int(speed_factor)})')
        if collision_level is not None:
            self.dash.call(f'SetCollisionLevel({int(collision_level)})', check=False)

    def stop(self):
        """Stop motion and clear the controller's motion queue (robot stays enabled)."""
        if self.dash:
            self.dash.call('ResetRobot()', check=False)

    def disable(self):
        if self.dash:
            self.dash.call('DisableRobot()', check=False)

    def error_id(self) -> str:
        return self.dash.call('GetErrorID()', check=False)

    # ---------------------------------------------------------------- motion
    def joint_movj(self, q_rad, speed_j: int, acc_j: int):
        d = [math.degrees(v) for v in q_rad]
        cmd = 'JointMovJ({:f},{:f},{:f},{:f},{:f},{:f},SpeedJ={:d},AccJ={:d})'.format(
            *d, int(speed_j), int(acc_j))
        self.move.call(cmd)
