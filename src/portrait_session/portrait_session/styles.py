"""Session presets per portrait style (arm angles, camera height, framing, voice). No ROS imports.

The style belongs to a place (its background): scripts/calibrate_layout.py --style <name> writes these
values into layout/current.yaml, solving and safety-checking the arm poses for exactly these angles
and heights. The post-processing look of each style is in postprocess/portrait_post/style.py.
"""

SESSION_STYLES = {
    # Relaxed, modern, approachable: more angle variety, casual camera height, more body in frame.
    'natural': {
        'style': 'natural',
        'shot_azimuths_deg': [-8.0, -15.0, -22.0],
        'shot_counts': [2, 2, 2],
        'camera_above_eyes_m': 0.12,
        'eye_line_fraction': 0.40,
        'yaw_tolerance_deg': 12.0,
        'pitch_tolerance_deg': 14.0,
        'expressions': ['smile', 'expr_big_smile', 'expr_laugh'],
    },
    # Clean, confident, formal: consistent gentle three-quarter view, camera close to eye level.
    'professional': {
        'style': 'professional',
        'shot_azimuths_deg': [-8.0, -12.0, -16.0],
        'shot_counts': [2, 2, 2],
        'camera_above_eyes_m': 0.08,
        'eye_line_fraction': 0.36,
        'yaw_tolerance_deg': 8.0,
        'pitch_tolerance_deg': 10.0,
        'expressions': ['expr_confident_smile', 'smile', 'expr_serious'],
    },
}
DEFAULT_STYLE = 'natural'


def session_style(name: str) -> dict:
    if name not in SESSION_STYLES:
        raise ValueError(f'unknown style {name!r}; choose from {sorted(SESSION_STYLES)}')
    return dict(SESSION_STYLES[name])
