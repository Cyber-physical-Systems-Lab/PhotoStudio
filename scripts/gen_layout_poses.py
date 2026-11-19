#!/usr/bin/env python3
"""Offline version of calibrate_layout.py: layout from a known face position and start joints.

    python3 scripts/gen_layout_poses.py --face-base 0.0 -1.597 0.338 \
        --seed -82.69 -50.67 -109.12 63.83 -89.67 5.54 [--out layout/current.yaml]
"""
import argparse
import sys

from layout_lib import SESSION_STYLES, DEFAULT_STYLE, apply_style, DEFAULT_CONFIG, layout_params, load_config, solve_layout

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--config', default=DEFAULT_CONFIG)
ap.add_argument('--face-base', type=float, nargs=3, required=True, help='face centre, robot base frame [m]')
ap.add_argument('--seed', type=float, nargs=6, required=True, help='start joints [deg]')
ap.add_argument('--standoff', type=float, default=None, help='default: session standoff_m')
ap.add_argument('--out', default='', help='write the layout file here (default: print it)')
ap.add_argument('--style', choices=sorted(SESSION_STYLES), default=DEFAULT_STYLE)
a = ap.parse_args()
P, SP = load_config(a.config)
SP = apply_style(SP, a.style)
r, problems = solve_layout(a.face_base, a.seed, P, SP, standoff=a.standoff)
if r is None:
    sys.exit('NOT OK:\n  ' + '\n  '.join(problems))
text = layout_params(r, f'generated offline by gen_layout_poses.py, style {a.style}', style=a.style)
if a.out:
    open(a.out, 'w').write(text)
    print('wrote', a.out)
else:
    print(text)
