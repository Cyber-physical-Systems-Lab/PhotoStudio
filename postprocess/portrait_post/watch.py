"""Post-processing service, independent of ROS.

The ROS 2 capture session only writes raw shots:
    <root>/<name>/<stamp>_raw/shot_*.png, session.json, capture_complete.json
This process watches <root>, and for every raw folder that has capture_complete.json but no
post.json it produces <root>/<name>/<stamp>.jpg (plus compare.jpg, matte.png, depth.png and post.json
in the raw folder). Failures are written to post_error.txt and not retried until it is deleted.

    python -m portrait_post.watch --root ~/Nova5Photo/portraits            # run forever
    python -m portrait_post.watch --root ~/Nova5Photo/portraits --once     # process backlog, exit
"""
from __future__ import annotations

import argparse
import glob
import os
import time
import traceback

from .cli import process

MARKER = 'capture_complete.json'


def pending(root: str):
    for marker in sorted(glob.glob(os.path.join(root, '*', '*_raw', MARKER))):
        raw = os.path.dirname(marker)
        if not os.path.exists(os.path.join(raw, 'post.json')) and \
                not os.path.exists(os.path.join(raw, 'post_error.txt')):
            yield raw


def output_path(raw_dir: str) -> str:
    stamp = os.path.basename(raw_dir)[:-len('_raw')]
    return os.path.join(os.path.dirname(raw_dir), f'{stamp}.jpg')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.path.expanduser('~/Nova5Photo/portraits'))
    ap.add_argument('--size', default='1200x1500')   # 4K shots: ~1100 px of head-and-shoulders detail
    ap.add_argument('--blur', type=float, default=None, help='override the style\'s background blur')
    ap.add_argument('--style', default=None, help='natural | professional (default: per session, set by the layout)')
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split('x'))
    root = os.path.expanduser(a.root)
    os.makedirs(root, exist_ok=True)
    print(f'[portrait_post] watching {root}', flush=True)
    while True:
        for raw in pending(root):
            out = output_path(raw)
            print(f'[portrait_post] processing {raw}', flush=True)
            try:
                process(raw, out, (w, h), a.blur, log=lambda m: print('[portrait_post]', m, flush=True), style=a.style)
            except Exception:  # noqa: BLE001 - keep the service alive, record the failure
                err = traceback.format_exc()
                with open(os.path.join(raw, 'post_error.txt'), 'w') as f:
                    f.write(err)
                print(f'[portrait_post] FAILED {raw}\n{err}', flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
