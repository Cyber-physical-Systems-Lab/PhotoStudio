"""Portrait styles: 'natural' (relaxed, modern, approachable) and 'professional' (clean, confident, formal).

Lightroom-style adjustments applied to the cropped portrait, on top of the base rendering (auto white
balance + face exposure). The values are illustrative starting points from the lab's style sheet
(2026-11-18), not calibrated against RAW data. The face is never reshaped in either style.

| Parameter            | Natural                       | Professional                         |
|----------------------|-------------------------------|--------------------------------------|
| Exposure             | +0.20 EV                      | +0.10 EV                             |
| Temperature          | -300 K                        | -500 K                               |
| Contrast             | 0 (sheet: -5, looked hazy)    | +8                                   |
| Highlights / Shadows | -35 / +25                     | -40 / +20                            |
| Whites / Blacks      | +5 / +5                       | +5 / -5                              |
| Texture / Clarity    | +6 / 0 (sheet: +3 / -3)       | +5 / +3   (texture/sharpening: person only) |
| Vibrance / Saturation| +5 / -3                       | 0 / -5                               |
| Background saturation| -10                           | -20                                  |
| Face exposure        | +0.15 EV                      | +0.20 EV                             |
| Background blur      | soft, recognisable            | stronger, clean                      |
| Crop                 | chest-up / half-body, slightly asymmetric | head and shoulders, centred |
| Skin retouching      | none                          | subtle (light smoothing, skin only)  |
| Vignette             | none                          | very subtle                          |
| Sharpening           | moderate                      | moderately high                      |
"""
from __future__ import annotations

import cv2
import numpy as np

STYLES = {
    'natural': dict(
        # contrast/clarity 0 (not -5/-3) and lighter blur: the first renders looked hazy and smeared.
        exposure_ev=0.20, temperature_k=-300, contrast=0, highlights=-35, shadows=25, whites=5, blacks=5,
        texture=2, clarity=0, vibrance=5, saturation=-3, background_saturation=-10, face_ev=0.15,
        blur=0.007, face_frac=0.21, eye_line=0.34, offset_x=0.04, retouch=0.0, vignette=0.0, sharpen=0.20,
        denoise=4,
        grade='vivid'),
    'professional': dict(
        exposure_ev=0.10, temperature_k=-500, contrast=8, highlights=-40, shadows=20, whites=5, blacks=-5,
        texture=2, clarity=3, vibrance=0, saturation=-5, background_saturation=-20, face_ev=0.20,
        blur=0.022, face_frac=0.28, eye_line=0.33, offset_x=0.0, retouch=0.25, vignette=0.06, sharpen=0.25,
        denoise=4,
        grade='none'),          # user 2026-11-19: keep professional ungraded; vivid for natural
}

# Colour grades (调色), applied after the style's adjustments. Inspired by well-known camera looks; the
# makers' actual colour science is proprietary, these only reproduce the character.
#   vivid   - like Leica Vivid: punchy S-curve, deeper blacks, richer colour, skin held back
#   classic - like Leica Classic/Authentic: filmic curve, soft highlight roll-off, muted colour, vignette
#   hncs    - like Hasselblad Natural Colour Solution: faithful neutral colour, gentle curve, natural skin
GRADES = {
    'vivid':   dict(contrast=0.14, toe=0.03, shoulder=0.00, chroma=1.18, skin_chroma=1.05, skin_pull=0.0,
                    neutralise=0.0, warmth=0.01, vignette=0.0),
    'classic': dict(contrast=0.16, toe=0.05, shoulder=0.35, chroma=0.86, skin_chroma=0.95, skin_pull=0.0,
                    neutralise=0.0, warmth=0.02, vignette=0.12),
    'hncs':    dict(contrast=0.05, toe=0.01, shoulder=0.25, chroma=1.00, skin_chroma=0.97, skin_pull=0.35,
                    neutralise=0.3, warmth=0.0, vignette=0.0),
    'none':    None,
}
DEFAULT = 'natural'


def get(style: str | None) -> dict:
    return dict(STYLES.get((style or DEFAULT).lower(), STYLES[DEFAULT]))


def _srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def face_mask(shape, face_bbox, matte=None):
    """Soft ellipse over the face (from the detector box), limited to the person."""
    H, W = shape[:2]
    m = np.zeros((H, W), np.float32)
    if face_bbox is None:
        return m
    x, y, w, h = face_bbox
    cv2.ellipse(m, (int(x + w / 2), int(y + h * 0.55)), (int(w * 0.62), int(h * 0.75)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, w * 0.12))
    if matte is not None:
        m *= matte
    return m


def grade(f: np.ndarray, g) -> np.ndarray:
    """Colour grade on float BGR 0..1 (see GRADES)."""
    if not g:
        return f
    lab = cv2.cvtColor(f.astype(np.float32), cv2.COLOR_BGR2LAB)
    L = lab[..., 0] / 100.0
    # Tone: S-curve around mid grey, a toe that deepens blacks, a shoulder that rolls highlights off.
    L = L + g['contrast'] * (L - 0.5) * (1 - np.abs(2 * L - 1))
    L = L - g['toe'] * (1 - _smooth(0.0, 0.3, L))
    hi = _smooth(0.7, 1.0, L)
    L = L - g['shoulder'] * 0.15 * hi * hi
    lab[..., 0] = np.clip(L, 0, 1) * 100.0
    a, b = lab[..., 1], lab[..., 2]
    if g['neutralise']:                                   # pull a remaining colour cast towards grey
        a -= g['neutralise'] * float(np.median(a))
        b -= g['neutralise'] * float(np.median(b))
    hue = np.degrees(np.arctan2(b, a))
    chroma = np.hypot(a, b)
    skin = _smooth(20, 35, hue) * (1 - _smooth(70, 85, hue)) * _smooth(5, 15, chroma)   # skin hues ~35-70 deg
    gain = g['chroma'] * (1 - skin) + g['skin_chroma'] * skin
    if g['skin_pull']:                                    # natural skin: hue towards ~55 deg (less magenta / orange)
        hue = hue + g['skin_pull'] * skin * (55.0 - hue)
    chroma = chroma * gain
    lab[..., 1] = chroma * np.cos(np.radians(hue))
    lab[..., 2] = chroma * np.sin(np.radians(hue)) + g['warmth'] * 100 * 0.1
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1)
    if g['vignette']:
        H, W = out.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
        out = out * (1 - g['vignette'] * _smooth(0.35, 1.0, r))[..., None]
    return out


def apply(img: np.ndarray, matte: np.ndarray, face_bbox, p: dict) -> np.ndarray:
    """img: BGR uint8 (already cropped); matte: person 0..1 same size; face_bbox in img pixels."""
    H, W = img.shape[:2]
    scale = H / 1500.0                                       # radii are tuned for a 1200x1500 output
    if p.get('denoise'):
        # Webcam frames carry compression blocks and sensor noise; remove them before any detail
        # enhancement, otherwise texture/sharpening amplify them into a 'crunchy' look.
        img = cv2.fastNlMeansDenoisingColored(img, None, p['denoise'], p['denoise'], 5, 15)
    f = img.astype(np.float32) / 255.0
    fm = face_mask(img.shape, face_bbox, matte)

    # Exposure (global + face) and temperature, in linear light.
    lin = _srgb_to_lin(f)
    lin *= 2.0 ** p['exposure_ev']
    lin *= (2.0 ** (p['face_ev'] * fm))[..., None]
    k = p['temperature_k'] / 1000.0                          # negative = cooler
    lin[..., 2] *= 1 + 0.08 * k                              # red (BGR order)
    lin[..., 0] *= 1 - 0.10 * k                              # blue
    f = _lin_to_srgb(lin).astype(np.float32)

    # Tone (on luminance; colour follows the luminance ratio).
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    L = lab[..., 0] / 100.0
    L0 = L.copy()
    L = L + (p['highlights'] / 100) * 0.30 * _smooth(0.5, 1.0, L) * (L - 0.45)
    L = L + (p['shadows'] / 100) * 0.35 * (1 - _smooth(0.0, 0.5, L)) * (0.55 - L)
    L = L + (p['whites'] / 100) * 0.10 * _smooth(0.75, 1.0, L)
    L = L + (p['blacks'] / 100) * 0.10 * (1 - _smooth(0.0, 0.25, L))
    L = 0.5 + (L - 0.5) * (1 + 0.5 * p['contrast'] / 100)
    # Clarity: mid-scale local contrast (midtones); texture: fine detail.
    mid = 1 - np.abs(L - 0.5) * 2
    L = L + (p['clarity'] / 100) * 1.5 * mid * (L - cv2.GaussianBlur(L, (0, 0), 25 * scale))
    # Texture only on the person: sharpening the blurred background brings out its smears.
    L = L + (p['texture'] / 100) * 2.0 * matte * (L - cv2.GaussianBlur(L, (0, 0), 1.6 * scale))
    L = np.clip(L, 0, 1)
    lab[..., 0] = L * 100.0
    # Saturation, vibrance (less on already saturated colours), background saturation.
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2) / 100.0
    sat = (1 + p['saturation'] / 100) * (1 + (p['vibrance'] / 100) * (1 - np.clip(chroma, 0, 1)))
    sat = sat * (1 + (p['background_saturation'] / 100) * (1 - matte))
    lab[..., 1] *= sat
    lab[..., 2] *= sat
    f = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1)

    f = grade(f, GRADES.get(p.get('grade') or 'none'))

    # Subtle skin smoothing (professional): edge-preserving, skin pixels of the face only, blended.
    if p['retouch'] > 0 and face_bbox is not None:
        u8 = (f * 255).astype(np.uint8)
        smooth = cv2.bilateralFilter(u8, 0, 18, max(2.0, 4 * scale)).astype(np.float32) / 255
        ycc = cv2.cvtColor(u8, cv2.COLOR_BGR2YCrCb)
        skin = ((ycc[..., 1] > 135) & (ycc[..., 1] < 180) & (ycc[..., 2] > 85) & (ycc[..., 2] < 135)).astype(np.float32)
        skin = cv2.GaussianBlur(skin, (0, 0), 2 * scale) * fm
        a = (p['retouch'] * skin)[..., None]
        f = f * (1 - a) + smooth * a

    # Very subtle vignette (professional).
    if p['vignette'] > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
        f *= (1 - p['vignette'] * _smooth(0.45, 1.0, r))[..., None]

    # Output sharpening (unsharp mask), on the person only.
    if p['sharpen'] > 0:
        blur = cv2.GaussianBlur(f, (0, 0), 1.0 * scale)
        f = f + (p['sharpen'] * matte)[..., None] * (f - blur)
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
