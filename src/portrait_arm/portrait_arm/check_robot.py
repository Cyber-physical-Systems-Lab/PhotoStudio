"""Read-only check of the real Nova 5: connects, prints state, compares URDF FK with the
controller's flange pose. Sends NO motion and does not enable the robot.

    ros2 run portrait_arm check_robot --ip 192.168.5.1
"""
import argparse
import time

import numpy as np

from .dobot_v3 import MODE_NAMES, DobotV3Client
from .kinematics import Kinematics, load_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', default='192.168.5.1')
    ap.add_argument('--urdf', default='')
    args = ap.parse_args()
    if not args.urdf:
        from ament_index_python.packages import get_package_share_directory
        args.urdf = get_package_share_directory('portrait_description') + '/urdf/nova5_portrait.urdf'
    kin = Kinematics(load_chain(args.urdf, 'base_link', 'tool_link'))
    c = DobotV3Client(args.ip, log=lambda s: None)
    c.connect()
    try:
        time.sleep(0.3)
        fb = c.feedback()
        q = c.joints_rad()
        p_ctrl, rpy_ctrl = c.tool_vector()
        p_fk = kin.fk(q)[:3, 3]
        print(f'robot_mode      : {MODE_NAMES.get(int(fb["robot_mode"]), fb["robot_mode"])}')
        print(f'enable / error  : {int(fb["enable_status"])} / {int(fb["error_status"])}')
        print(f'speed factor    : {int(fb["velocity_ratio"])} %   user {int(fb["user"])} tool {int(fb["tool"])}')
        print(f'joints [deg]    : {np.round(np.degrees(q), 2).tolist()}')
        print(f'controller xyz  : {np.round(p_ctrl * 1000, 1).tolist()} mm  rpy {np.round(rpy_ctrl, 1).tolist()} deg')
        print(f'URDF FK xyz     : {np.round(p_fk * 1000, 1).tolist()} mm')
        print(f'difference      : {np.linalg.norm(p_ctrl - p_fk) * 1000:.1f} mm (must be < 10 mm to move)')
        print('error id        :', c.error_id())
    finally:
        c.close()


if __name__ == '__main__':
    main()
