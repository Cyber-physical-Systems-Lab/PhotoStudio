"""Isaac Sim scene for the Nova 5 portrait photographer.

Run through sim/run_isaac.sh (Isaac Sim's Python 3.12 + its bundled ROS 2 Humble rclpy):

    sim/run_isaac.sh sim/isaac_portrait_sim.py --headless --record /tmp/run

ROS interface (identical to what the real hardware provides to the rest of the stack):
    in : /isaac/joint_command  sensor_msgs/JointState   (from arm_node, backend:=sim)
         /portrait/guidance    std_msgs/String          (voice prompts: the virtual subject reacts)
         /portrait/session_state, /portrait/speech       (only for the recording overlay)
    out: /isaac/joint_states   sensor_msgs/JointState
         /camera/image_raw     sensor_msgs/Image (rgb8)  flange webcam
         /camera/camera_info   sensor_msgs/CameraInfo

The flange camera pose is computed from the simulated joint angles with the same kinematics and
camera mount (portrait.yaml) that arm_node uses, so sim and controller agree by construction.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--headless', action='store_true')
ap.add_argument('--config', default=os.path.join(ROOT, 'src/portrait_bringup/config/portrait.yaml'))
ap.add_argument('--layout', default='', help='layout file from calibrate_layout.py (merged on top of --config)')
ap.add_argument('--subject', default='F_Business_02', help='Isaac People character name')
ap.add_argument('--subject-yaw-offset', type=float, default=18.0,
                help='initial head/body turn in degrees (+ = towards their left); voice guidance corrects it')
ap.add_argument('--subject-scale', type=float, default=1.0,
                help='size of the person (0.9 = 10%% shorter); the pedestal stays set for an average person')
ap.add_argument('--subject-offset', type=float, nargs=2, default=[0.0, 0.0], metavar=('DX', 'DY'),
                help='where the person stands relative to the mark, in robot base x/y metres')
ap.add_argument('--leave-on', default='next_angle',
                help='guidance key after which the subject steps off the mark for --leave-seconds ("" = never)')
ap.add_argument('--leave-seconds', type=float, default=5.0)
ap.add_argument('--width', type=int, default=1280)
ap.add_argument('--height', type=int, default=720)
ap.add_argument('--hfov', type=float, default=82.0)
ap.add_argument('--record', default='', help='directory for an mp4 of the run (observer + robot view)')
ap.add_argument('--probe', action='store_true', help='render a few frames, save PNGs and exit')
ap.add_argument('--max-seconds', type=float, default=0.0, help='exit after this many seconds (0 = run forever)')
ap.add_argument('--exit-after-session', action='store_true',
                help='quit a few seconds after the session finishes (scripted recordings)')
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

# DLSS/NGX (and the OptiX denoiser) are not available on the lab machine, so the default
# real-time path tracer is very noisy. The reference path tracer with enough samples per frame
# gives clean frames for face detection. Override with PORTRAIT_RENDERER / PORTRAIT_SPP.
app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720,
                     'renderer': os.environ.get('PORTRAIT_RENDERER', 'PathTracing'),
                     'samples_per_pixel_per_frame': int(os.environ.get('PORTRAIT_SPP', '8')),
                     'denoiser': False, 'anti_aliasing': 1})

import omni.kit.app  # noqa: E402

_mgr = omni.kit.app.get_app().get_extension_manager()
for _e in ('isaacsim.ros2.bridge', 'isaacsim.asset.importer.urdf'):
    _mgr.set_extension_enabled_immediate(_e, True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import rclpy  # noqa: E402
import yaml  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, 'src/portrait_arm'))
from portrait_arm.kinematics import Kinematics, layout_base, load_chain, make_tf, rpy_to_matrix  # noqa: E402

URDF = os.path.join(ROOT, 'src/portrait_description/urdf/nova5_portrait.urdf')
CACHE = os.path.expanduser('~/.cache/nova5photo/sim')
JOINTS = [f'joint{i}' for i in range(1, 7)]


# ------------------------------------------------------------------------------------ helpers
def log(*a):
    print('[portrait_sim]', *a, flush=True)


def set_xform(prim, t=(0, 0, 0), yaw_deg=0.0, R=None):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    M = np.eye(4)
    if R is None:
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    M[:3, :3] = R
    M[:3, 3] = t
    xf.AddTransformOp().Set(Gf.Matrix4d(*M.T.ravel().tolist()))   # USD is row-vector convention


def import_robot():
    import shutil
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    out_dir = os.path.join(CACHE, 'nova5_portrait')
    stamp = os.path.join(CACHE, 'nova5_portrait.source_mtime')
    usd = os.path.join(out_dir, 'nova5_portrait', 'nova5_portrait.usda')
    mtime = str(os.path.getmtime(URDF))
    if os.path.exists(stamp) and open(stamp).read() == mtime and os.path.exists(usd):
        return usd
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    cfg = URDFImporterConfig(
        urdf_path=URDF, usd_path=out_dir, merge_fixed_joints=False, fix_base=True,
        collision_from_visuals=False, robot_type='Manipulator',
        joint_drive_type='force', joint_target_type='position',
        override_joint_stiffness=1.0e5, override_joint_damping=1.0e3,
        ros_package_paths=[{'name': 'portrait_description',
                            'path': os.path.join(ROOT, 'src/portrait_description')}])
    usd = URDFImporter(cfg).import_urdf()
    open(stamp, 'w').write(mtime)
    log('imported robot USD:', usd)
    return usd


# ------------------------------------------------------------------------------------ config
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from layout_lib import merged_config  # noqa: E402
_CFG = merged_config(args.config, args.layout)
P = _CFG['arm_node']['ros__parameters']
SP = _CFG['session_node']['ros__parameters']
mount = make_tf(rpy_to_matrix(*np.radians(P['camera_mount_rpy_deg'])), P['camera_mount_xyz'])
kin = Kinematics(load_chain(URDF, 'base_link', 'tool_link'), mount)
home = np.radians(P['named_poses']['home'])
face_nominal = np.array(P['nominal_face_position'], float)

# ------------------------------------------------------------------------------------ stage
ctx = omni.usd.get_context()
stage = ctx.get_stage()          # SimulationApp's startup stage (new_stage() crashes OmniGraph here)
for _ in range(3):
    app.update()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.Scene.Define(stage, '/World/Physics')
world = UsdGeom.Xform.Define(stage, '/World')

# Lighting: soft dome + key light from the camera side, like a simple portrait setup.
dome = UsdLux.DomeLight.Define(stage, '/World/Dome')
dome.CreateIntensityAttr(600.0)
key = UsdLux.RectLight.Define(stage, '/World/KeyLight')
key.CreateIntensityAttr(12000.0)
key.CreateWidthAttr(1.0)
key.CreateHeightAttr(1.0)
fill = UsdLux.SphereLight.Define(stage, '/World/Fill')
fill.CreateIntensityAttr(20000.0)
fill.CreateRadiusAttr(0.3)

# Floor and a back wall a few metres behind the subject (gives the depth-of-field step something
# to blur), plus a shelf to make the background less uniform.
def box(path, size, pos, color):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return c


# Subject on the floor; robot on a pedestal so the camera can reach eye height.
subject_eye_world = 1.58      # refined below from the character's bounding box
box('/World/Floor', (8, 8, 0.02), (1.0, 0, -0.01), (0.55, 0.55, 0.52))
box('/World/BackWall', (0.1, 6, 3.0), (3.6, 0, 1.5), (0.70, 0.74, 0.80))
box('/World/Shelf', (0.4, 1.2, 1.8), (3.2, -1.3, 0.9), (0.45, 0.30, 0.20))
box('/World/Plant', (0.3, 0.3, 1.4), (2.6, 1.4, 0.7), (0.20, 0.45, 0.22))
box('/World/Cabinet', (0.5, 0.9, 1.0), (2.2, -2.0, 0.5), (0.30, 0.32, 0.38))
box('/World/MarkedSpot', (0.35, 0.35, 0.004), (face_nominal[0], face_nominal[1], 0.002), (0.9, 0.75, 0.1))

# Subject (Isaac People character)
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

subj = stage.DefinePrim('/World/Subject', 'Xform')
inner = stage.DefinePrim('/World/Subject/Character', 'Xform')
if args.subject != 'none':
    char_url = f'{get_assets_root_path()}/Isaac/People/Characters/{args.subject}/{args.subject}.usd'
    inner.GetReferences().AddReference(char_url)
    log('subject:', char_url)
    for _ in range(20):
        app.update()
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = bb.ComputeWorldBound(inner).ComputeAlignedRange()
    top = rng.GetMax()[2] if not rng.IsEmpty() else 1.72
    if top > 3.0:          # centimetre asset
        UsdGeom.Xformable(inner).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
        rng = bb.ComputeWorldBound(inner).ComputeAlignedRange()
        top = rng.GetMax()[2]
    subject_eye_world = float(top) - 0.115
    log(f'character height {top:.2f} m, eye height ~{subject_eye_world:.2f} m')
    # Characters face +Y by default in this library; rotate so the subject faces the robot (-X).
    CHAR_FACING_YAW = float(os.environ.get('PORTRAIT_CHAR_FACING_YAW', '-90'))
    UsdGeom.Xformable(inner).AddRotateZOp().Set(CHAR_FACING_YAW)

pedestal_h = subject_eye_world - face_nominal[2]
box('/World/Pedestal', (0.30, 0.30, pedestal_h), (0, 0, pedestal_h / 2), (0.25, 0.25, 0.27))
log(f'pedestal height {pedestal_h:.3f} m (robot base at that height)')

subject_state = {'yaw': args.subject_yaw_offset, 'away_until': 0.0, 'left_once': False}
SUBJECT_XY = np.array([face_nominal[0] + 0.02, face_nominal[1]]) + np.array(args.subject_offset)
log(f'subject scale {args.subject_scale:.2f} (eyes ~{subject_eye_world * args.subject_scale:.2f} m), '
    f'offset from mark {args.subject_offset}')


def place_subject():
    now = time.time()
    away = subject_state.get('leave_at', 1e18) <= now < subject_state['away_until']
    xy = SUBJECT_XY + (np.array([0.4, 1.6]) if away else 0)
    c, s_ = math.cos(math.radians(subject_state['yaw'])), math.sin(math.radians(subject_state['yaw']))
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) * args.subject_scale
    set_xform(subj, (xy[0], xy[1], 0.0), R=R)


place_subject()
# Key light: in front-left of the subject, above, aimed at the face.
kl = UsdGeom.Xformable(stage.GetPrimAtPath('/World/KeyLight'))
kl.ClearXformOpOrder()
kl.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.9, 2.3))
kl.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, -35.0, 0.0))
UsdGeom.Xformable(stage.GetPrimAtPath('/World/Fill')).AddTranslateOp().Set(Gf.Vec3d(0.2, -1.2, 1.8))

# Robot
robot_usd = import_robot()
robot = stage.DefinePrim('/World/Robot', 'Xform')
robot.GetReferences().AddReference(robot_usd)
# Kinematic robot: no physics variant; link transforms are set from our FK every frame.
robot.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('none')
set_xform(robot, (0.0, 0.0, pedestal_h), R=layout_base(P.get('layout_yaw_deg', 0.0))[:3, :3])
for _ in range(10):
    app.update()
link_prims = {}
for prim in Usd.PrimRange(robot):
    if prim.GetName() in ('Link1', 'Link2', 'Link3', 'Link4', 'Link5', 'Link6') \
            and prim.GetName() not in link_prims:          # shallowest = the link Xform, not its mesh
        link_prims[prim.GetName()] = prim
log('robot links:', {k: v.GetPath().pathString for k, v in link_prims.items()})
# The Isaac world is the layout frame (+x towards the subject); the robot base is turned by
# layout_yaw_deg so the same joint angles as on the real arm point at the subject.
T_world_base = make_tf(None, (0.0, 0.0, pedestal_h)) @ layout_base(P.get('layout_yaw_deg', 0.0))


# Cameras: USD cameras look down -Z with +Y up; optical frame is +Z forward, +Y down.
OPT_TO_USD = np.diag([1.0, -1.0, -1.0])


def make_camera(path, w, h, hfov):
    cam = UsdGeom.Camera.Define(stage, path)
    aperture = 20.955
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateVerticalApertureAttr(aperture * h / w)
    cam.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(hfov) / 2)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
    rp = rep.create.render_product(path, (w, h))
    ann = rep.AnnotatorRegistry.get_annotator('rgb')
    ann.attach([rp])
    return cam, ann


robot_cam, robot_ann = make_camera('/World/RobotCamera', args.width, args.height, args.hfov)
fx = (args.width / 2) / math.tan(math.radians(args.hfov) / 2)

obs_cam, obs_ann = (None, None)
if args.record or args.probe:
    obs_cam, obs_ann = make_camera('/World/ObserverCamera', 960, 720, 60.0)
    eye = np.array([0.55, -2.1, 1.75])
    tgt = np.array([0.55, 0.0, 1.25])
    z = tgt - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    set_xform(obs_cam.GetPrim(), eye, R=np.column_stack([x, y, z]) @ OPT_TO_USD)


def camera_world_pose(q):
    return T_world_base @ kin.fk(q)


def update_robot_camera(q):
    T = camera_world_pose(q)
    set_xform(robot_cam.GetPrim(), T[:3, 3], R=T[:3, :3] @ OPT_TO_USD)


# ------------------------------------------------------------------------------------ robot
from portrait_arm.kinematics import axis_angle  # noqa: E402

timeline = omni.timeline.get_timeline_interface()
timeline.play()
_active = kin.chain.active
_nested = all(link_prims.get(f'Link{i + 1}') is not None for i in range(6)) and \
    link_prims['Link2'].GetParent() == link_prims['Link1']
log('link hierarchy nested:', _nested)
q_sim = np.zeros(6)
JOINT_SPEED_LIMIT = math.radians(60.0)    # physical-ish limit on top of the arm node's own cap


def set_joints(q):
    """Place every link from the joint angles (URDF joint origins x joint rotations)."""
    T = np.eye(4)
    for i, j in enumerate(_active):
        local = j.origin @ make_tf(axis_angle(j.axis, q[i]))
        T = T @ local
        prim = link_prims[f'Link{i + 1}']
        set_xform(prim, local[:3, 3], R=local[:3, :3]) if _nested else \
            set_xform(prim, (T_world_base @ T)[:3, 3], R=(T_world_base @ T)[:3, :3])


def get_joints():
    return q_sim.copy()


q_cmd = home.copy()
q_sim = home.copy()
set_joints(q_sim)
update_robot_camera(q_sim)
for _ in range(3):
    app.update()

# ------------------------------------------------------------------------------------ ROS
try:   # let Ctrl-C / SIGINT reach our own finally-block so the video is closed properly
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
except (ImportError, TypeError):
    rclpy.init()
node = rclpy.create_node('isaac_portrait_sim')
js_pub = node.create_publisher(JointState, '/isaac/joint_states', 10)
# Reliable QoS: multi-megabyte images lose UDP fragments under best-effort and almost never arrive.
img_pub = node.create_publisher(Image, '/camera/image_raw', 2)
info_pub = node.create_publisher(CameraInfo, '/camera/camera_info', 2)
overlay = {'state': '', 'speech': ''}


def on_cmd(msg):
    global q_cmd
    try:
        q_cmd = np.array([msg.position[msg.name.index(j)] for j in JOINTS])
    except ValueError:
        pass


def on_guidance(msg):
    k = msg.data
    step = 7.0
    if k == 'turn_left':
        subject_state['yaw'] += step
    elif k == 'turn_right':
        subject_state['yaw'] -= step
    # Steps are in the person's frame: they face the robot (-x), their left is -y.
    elif k in ('step_back', 'step_forward', 'step_left', 'step_right'):
        d = {'step_back': (0.10, 0), 'step_forward': (-0.10, 0),
             'step_left': (0, -0.10), 'step_right': (0, 0.10)}[k]
        SUBJECT_XY[:] = SUBJECT_XY + np.array(d)
        log(f'subject stepped: {k}, now at {np.round(SUBJECT_XY, 2).tolist()}')
    if args.leave_on and k == args.leave_on and not subject_state['left_once']:
        subject_state['left_once'] = True
        subject_state['away_until'] = time.time() + 1.5 + args.leave_seconds
        subject_state['leave_at'] = time.time() + 1.5
        log('subject will step off the mark (stop-on-lost-face test)')
    log(f'guidance {k}: subject yaw now {subject_state["yaw"]:+.0f} deg')


def on_state(msg):
    try:
        d = json.loads(msg.data)
        overlay['state'] = f"{d.get('state', '')} {d.get('detail', '')}".strip()
        if d.get('state') == 'IDLE' and str(d.get('detail', '')).startswith(('finished', 'after abort')):
            overlay['finished_at'] = time.time()
    except ValueError:
        overlay['state'] = msg.data


def on_speech(msg):
    overlay['speech'] = msg.data


node.create_subscription(JointState, '/isaac/joint_command', on_cmd, 10)
node.create_subscription(String, '/portrait/guidance', on_guidance, 10)
node.create_subscription(String, '/portrait/session_state', on_state, 10)
node.create_subscription(String, '/portrait/speech', on_speech, 10)

info = CameraInfo()
info.width, info.height = args.width, args.height
info.distortion_model = 'plumb_bob'
info.d = [0.0] * 5
info.k = [float(v) for v in (fx, 0.0, args.width / 2, 0.0, fx, args.height / 2, 0.0, 0.0, 1.0)]
info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
info.p = [float(v) for v in (fx, 0.0, args.width / 2, 0.0, 0.0, fx, args.height / 2, 0.0, 0.0, 0.0, 1.0, 0.0)]

class FfmpegWriter:
    """H.264 mp4 through the system ffmpeg (Isaac Sim's OpenCV has no video encoder)."""

    def __init__(self, path, size, fps):
        import shutil
        import subprocess
        exe = shutil.which('ffmpeg')
        if exe is None:
            raise RuntimeError('ffmpeg not found; install it or drop --record')
        self.path = path
        self.proc = subprocess.Popen(
            [exe, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}',
             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24',
             '-pix_fmt', 'yuv420p', '-movflags', 'frag_keyframe+empty_moov', path],
            stdin=subprocess.PIPE, start_new_session=True)   # fragmented: playable even if killed

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=60)


writer = None
if args.record:
    os.makedirs(args.record, exist_ok=True)
    writer = FfmpegWriter(os.path.join(args.record, 'sim_run.mp4'), (960 + 960, 720), 5)


def rgb_of(ann):
    d = ann.get_data()
    if d is None or not hasattr(d, 'shape') or d.size == 0:
        return None
    return np.asarray(d)[:, :, :3]


def compose(obs, cam):
    obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR) if obs is not None else np.zeros((720, 960, 3), np.uint8)
    cam_bgr = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
    cam_small = cv2.resize(cam_bgr, (960, 540))
    right = np.zeros((720, 960, 3), np.uint8)
    right[:540] = cam_small
    cv2.putText(right, 'flange webcam', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(right, overlay['state'][:60], (10, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 255), 2)
    words, line, y = overlay['speech'].split(), '', 625
    for w in words:
        if len(line) + len(w) > 60:
            cv2.putText(right, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            line, y = '', y + 32
        line = (line + ' ' + w).strip()
    if line:
        cv2.putText(right, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(obs_bgr, 'Isaac Sim: Nova 5 portrait session', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return np.hstack([obs_bgr, right])


# ------------------------------------------------------------------------------------ loop
# Rendering (path traced, shared GPU) runs at only a few Hz, so joint integration and the
# /isaac/joint_states feedback run in their own 100 Hz thread; the render loop draws whatever
# joint state is current when a frame starts.
import threading  # noqa: E402

state_lock = threading.Lock()
running = True


def joint_loop():
    global q_sim
    t_last = time.time()
    while running and rclpy.ok():
        try:
            rclpy.spin_once(node, timeout_sec=0.0)
        except Exception:  # noqa: BLE001 - context shut down
            break
        t_now = time.time()
        dt = min(0.05, t_now - t_last)
        t_last = t_now
        with state_lock:
            step = np.clip(q_cmd - q_sim, -JOINT_SPEED_LIMIT * dt, JOINT_SPEED_LIMIT * dt)
            q_sim = q_sim + step
            q = q_sim.copy()
        js = JointState()
        js.header.stamp = node.get_clock().now().to_msg()
        js.name = JOINTS
        js.position = [float(v) for v in q]
        js_pub.publish(js)
        time.sleep(0.01)


threading.Thread(target=joint_loop, daemon=True).start()
log('running; waiting for /isaac/joint_command')
t_start = time.time()
frames = 0
VIDEO_FPS = 5.0
video_t = None
loop_n, loop_t = 0, time.time()
prev_stamp = None
try:
    while app.is_running():
        place_subject()
        with state_lock:
            q = q_sim.copy()
        stamp_q = node.get_clock().now().to_msg()   # the arm is drawn at this instant
        set_joints(q)
        update_robot_camera(q)
        app.update()
        # The rgb annotator returns the frame rendered on the previous update, so stamp the image
        # with the pose time of that previous update (keeps image and camera pose consistent).
        now = prev_stamp if prev_stamp is not None else stamp_q
        prev_stamp = stamp_q
        rgb = rgb_of(robot_ann)
        if rgb is not None:
            frames += 1
            m = Image()
            m.header.stamp = now
            m.header.frame_id = 'camera_optical_frame'
            m.height, m.width = rgb.shape[:2]
            m.encoding = 'rgb8'
            m.step = m.width * 3
            m.data = np.ascontiguousarray(rgb).tobytes()
            img_pub.publish(m)
            info.header = m.header
            info_pub.publish(info)
            if writer is not None or args.probe:
                obs = rgb_of(obs_ann)
                frame = compose(obs, rgb)
                if writer is not None:
                    # real-time video: repeat frames to cover the wall-clock time since the last one
                    t = time.time()
                    n = 1 if video_t is None else max(1, int(round((t - video_t) * VIDEO_FPS)))
                    video_t = t if video_t is None else video_t + n / VIDEO_FPS
                    for _ in range(min(n, 50)):
                        writer.write(frame)
                if args.probe and frames in (5, 15):
                    out = args.record or CACHE
                    cv2.imwrite(os.path.join(out, f'probe_{frames}.png'), frame)
                    cv2.imwrite(os.path.join(out, f'probe_cam_{frames}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    log('saved probe frame', frames)
        loop_n += 1
        if time.time() - loop_t > 30.0:
            log(f'render loop {loop_n / (time.time() - loop_t):.1f} Hz')
            loop_n, loop_t = 0, time.time()
        if args.probe and frames >= 15:
            break
        if args.max_seconds and time.time() - t_start > args.max_seconds:
            break
        if args.exit_after_session and time.time() - overlay.get('finished_at', 1e18) > 5.0:
            log('session finished; exiting')
            break
except KeyboardInterrupt:
    pass
finally:
    running = False
    time.sleep(0.05)
    if writer is not None:
        writer.release()
        log('recording saved to', os.path.join(args.record, 'sim_run.mp4'))
    try:
        node.destroy_node()
        rclpy.try_shutdown()
    except Exception:  # noqa: BLE001
        pass
    timeline.stop()
    app.close()
