"""Turn a burst of raw shots into one webpage portrait.

One-off run on a raw folder (the watcher in portrait_post.watch does this automatically):

    python -m portrait_post.cli --raw-dir portraits/ada/2026-11-16_101500_raw \
        --out portraits/ada/2026-11-16_101500.jpg [--size 800x1000]

Also writes <raw-dir>/compare.jpg (best raw frame cropped vs. final) and post.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from .dof import person_mask, refine_matte, relative_depth, render_bokeh
from .select import FaceTools, score_frames
from . import style as styles
from .tone import expose_face, portrait_crop, white_balance


def session_style(raw_dir):
    try:
        with open(os.path.join(raw_dir, 'session.json')) as f:
            return json.load(f).get('style')
    except (OSError, ValueError):
        return None


def process(raw_dir, out, size=(1200, 1500), blur=None, face_model=None, log=print, style=None, grade=None):
    """style: 'natural' | 'professional' (default: from session.json, else natural)."""
    t0 = time.time()
    style = style or session_style(raw_dir) or styles.DEFAULT
    P = styles.get(style)
    if blur is not None:
        P['blur'] = blur
    if grade is not None:
        P['grade'] = grade
    log(f'style: {style}, grade: {P["grade"]}')
    paths = sorted(glob.glob(os.path.join(raw_dir, '*.png')) + glob.glob(os.path.join(raw_dir, '*.jpg')))
    paths = [p for p in paths if os.path.basename(p).startswith("shot_")]
    if not paths:
        raise SystemExit(f'no images in {raw_dir}')
    tools = FaceTools(face_model)
    scores = score_frames(paths, tools)
    best = scores[0]
    log(f'best frame: {os.path.basename(best.path)} (score {best.total:.2f}, sharp {best.sharpness:.0f}, '
        f'eyes {best.eyes_open:.2f}, frontal {best.frontal:.2f})')
    img = best.image

    mask = person_mask(img, log)
    matte = refine_matte(img, mask)
    defocus = relative_depth(img, matte, log)
    bokeh = render_bokeh(img, matte, defocus, P['blur'], face_bbox=best.bbox)
    toned = expose_face(white_balance(bokeh), best.bbox)
    base, box = portrait_crop(toned, best.bbox, best.landmarks, *size, face_frac=P['face_frac'],
                              eye_line=P['eye_line'], offset_x=P['offset_x'])
    # Style adjustments on the crop (matte and face box cropped and scaled the same way).
    x0, y0, cw, ch = box
    sx, sy = size[0] / cw, size[1] / ch
    m_crop = cv2.resize(matte[y0:y0 + ch, x0:x0 + cw].astype(np.float32), size, interpolation=cv2.INTER_AREA)
    fb = None if best.bbox is None else [(best.bbox[0] - x0) * sx, (best.bbox[1] - y0) * sy,
                                          best.bbox[2] * sx, best.bbox[3] * sy]
    final = styles.apply(base, m_crop, fb, P)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, final, [cv2.IMWRITE_JPEG_QUALITY, 92])

    before = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(raw_dir, 'compare.jpg'), np.hstack([before, final]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(os.path.join(raw_dir, 'matte.png'), (matte * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(raw_dir, 'depth.png'), (defocus * 255).astype(np.uint8))
    info = {
        'output': out, 'style': style, 'grade': P['grade'], 'best': os.path.basename(best.path), 'seconds': round(time.time() - t0, 1),
        'ranking': [{'file': os.path.basename(s.path), 'score': round(s.total, 3),
                     'sharpness': round(s.sharpness, 1), 'eyes_open': round(s.eyes_open, 2),
                     'frontal': round(s.frontal, 2), **s.notes} for s in scores],
    }
    with open(os.path.join(raw_dir, 'post.json'), 'w') as f:
        json.dump(info, f, indent=2)
    log(f'wrote {out} in {info["seconds"]} s')
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', default='1200x1500')
    ap.add_argument('--blur', type=float, default=None, help='override the style\'s max blur radius (fraction of size)')
    ap.add_argument('--style', default=None, help='natural | professional (default: from session.json)')
    ap.add_argument('--grade', default=None, help='vivid | classic | hncs | none (default: the style\'s grade)')
    ap.add_argument('--face-model', default=None)
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split('x'))
    process(a.raw_dir, a.out, (w, h), a.blur, a.face_model, style=a.style, grade=a.grade)


if __name__ == '__main__':
    main()
