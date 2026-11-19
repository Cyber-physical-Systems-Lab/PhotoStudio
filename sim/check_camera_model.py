"""Sanity check that the simulated flange camera matches our kinematics + intrinsics.

Places red marker spheres at known positions, renders from a few arm poses, finds the markers in
the image and compares with the pixel predicted from FK + pinhole model. Run via sim/run_isaac.sh.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'width': 1280, 'height': 720, 'renderer': 'PathTracing',
                     'samples_per_pixel_per_frame': 8, 'denoiser': False, 'anti_aliasing': 1})
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

sys.path.insert(0, os.path.join(HERE, '..', 'src/portrait_arm'))
from portrait_arm.kinematics import Kinematics, load_chain, make_tf, rpy_to_matrix  # noqa: E402

ROOT = os.path.dirname(HERE)
P = yaml.safe_load(open(os.path.join(ROOT, 'src/portrait_bringup/config/portrait.yaml')))['arm_node']['ros__parameters']
kin = Kinematics(load_chain(os.path.join(ROOT, 'src/portrait_description/urdf/nova5_portrait.urdf'), 'base_link', 'tool_link'),
                 make_tf(rpy_to_matrix(*np.radians(P['camera_mount_rpy_deg'])), P['camera_mount_xyz']))
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr(1000.0)
W, H, HFOV = 1280, 720, 70.0
base_z = 0.712
markers = [np.array([1.0, 0.0, 0.78]), np.array([0.95, 0.12, 0.70]), np.array([1.05, -0.10, 0.88])]
for i, m in enumerate(markers):
    s = UsdGeom.Sphere.Define(stage, f'/World/M{i}')
    s.CreateRadiusAttr(0.01)
    s.AddTranslateOp().Set(Gf.Vec3d(m[0], m[1], m[2] + base_z))
    s.CreateDisplayColorAttr([Gf.Vec3f(1, 0, 0)])
cam = UsdGeom.Camera.Define(stage, '/World/Cam')
ap = 20.955
cam.CreateHorizontalApertureAttr(ap)
cam.CreateVerticalApertureAttr(ap * H / W)
cam.CreateFocalLengthAttr(ap / (2 * math.tan(math.radians(HFOV) / 2)))
cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
rp = rep.create.render_product('/World/Cam', (W, H))
ann = rep.AnnotatorRegistry.get_annotator('rgb')
ann.attach([rp])
omni.timeline.get_timeline_interface().play()
fx = (W / 2) / math.tan(math.radians(HFOV) / 2)
K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
OPT_TO_USD = np.diag([1.0, -1.0, -1.0])
for name in P['named_poses']['names']:
    q = np.radians(P['named_poses'][name])
    T = make_tf(None, (0, 0, base_z)) @ kin.fk(q)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    M = np.eye(4)
    M[:3, :3] = T[:3, :3] @ OPT_TO_USD
    M[:3, 3] = T[:3, 3]
    xf.AddTransformOp().Set(Gf.Matrix4d(*M.T.ravel().tolist()))
    for _ in range(6):
        app.update()
    img = np.asarray(ann.get_data())[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = ((img[:, :, 0] > 150) & (img[:, :, 1] < 80) & (img[:, :, 2] < 80)).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    found = [cents[j] for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] > 5]
    for m in markers:
        pw = np.append(m + [0, 0, base_z], 1.0)
        pc = np.linalg.inv(T) @ pw
        uv = (K @ pc[:3])[:2] / pc[2]
        d = min((np.linalg.norm(uv - f) for f in found), default=float('nan'))
        print(f'[check] {name:10s} marker {m.tolist()} predicted {np.round(uv, 1).tolist()} nearest blob {d:.1f} px', flush=True)
    cv2.imwrite(f'/tmp/check_{name}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
app.close()
