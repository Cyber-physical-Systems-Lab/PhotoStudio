#!/usr/bin/env python3
"""Webcam intrinsic calibration with a checkerboard (OpenCV).

Live (needs a display):    scripts/calibrate_camera.py --device /dev/video0 --cols 9 --rows 6 --square 0.025
From saved images:         scripts/calibrate_camera.py --images 'calib/*.png' --cols 9 --rows 6 --square 0.025

cols/rows count INNER corners. Live mode: SPACE grabs a view (aim for 15-25 views covering the whole
image, tilted at different angles), Q finishes. Writes an OpenCV YAML that camera_node reads via the
`calibration_file` parameter in portrait.yaml.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='/dev/video0')
    ap.add_argument('--images', default='')
    ap.add_argument('--cols', type=int, default=9)
    ap.add_argument('--rows', type=int, default=6)
    ap.add_argument('--square', type=float, default=0.025, help='square size in metres')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--out', default=os.path.expanduser('~/Nova5Photo/calibration/webcam.yaml'))
    a = ap.parse_args()

    pattern = (a.cols, a.rows)
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    obj_pts, img_pts, size = [], [], None

    def add(img):
        nonlocal size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pattern, None)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            obj_pts.append(objp)
            img_pts.append(corners)
            size = gray.shape[::-1]
        return ok, corners

    if a.images:
        for f in sorted(glob.glob(a.images)):
            ok, _ = add(cv2.imread(f))
            print(f, 'ok' if ok else 'no board')
    else:
        cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        os.makedirs(os.path.join(os.path.dirname(a.out), 'views'), exist_ok=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            show = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
            if found:
                cv2.drawChessboardCorners(show, pattern, corners, found)
            cv2.putText(show, f'views: {len(obj_pts)}  SPACE=grab  Q=done', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('calibration', cv2.resize(show, (show.shape[1] // 2, show.shape[0] // 2)))
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' ') and found:
                add(frame)
                cv2.imwrite(os.path.join(os.path.dirname(a.out), 'views', f'view_{len(obj_pts):02d}.png'), frame)
            elif k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    if len(obj_pts) < 8:
        raise SystemExit(f'only {len(obj_pts)} usable views; need at least 8')
    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f'RMS reprojection error {rms:.3f} px (good: < 0.5)')
    print('K =\n', K, '\nD =', D.ravel())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fs = cv2.FileStorage(a.out, cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', size[0])
    fs.write('image_height', size[1])
    fs.write('camera_matrix', K)
    fs.write('distortion_coefficients', D)
    fs.write('rms', rms)
    fs.release()
    print('wrote', a.out)


if __name__ == '__main__':
    main()
