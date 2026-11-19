"""Light exposure / colour correction and the webpage crop."""
from __future__ import annotations

import cv2
import numpy as np


def white_balance(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    gain = means.mean() / np.maximum(means, 1e-3)
    gain = 1 + strength * (gain - 1)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def expose_face(img: np.ndarray, face_bbox, target: float = 0.55, max_gain: float = 1.6) -> np.ndarray:
    """Gamma so that the face's mean luminance approaches `target`, then gentle local contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32) / 255
    if face_bbox is not None:
        x, y, w, h = [int(v) for v in face_bbox]
        roi = L[max(0, y):y + h, max(0, x):x + w]
        m = float(roi.mean()) if roi.size else float(L.mean())
    else:
        m = float(L.mean())
    m = min(max(m, 0.05), 0.95)
    gamma = np.clip(np.log(target) / np.log(m), 1 / max_gain, max_gain)
    L = L ** gamma
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    L8 = clahe.apply((L * 255).astype(np.uint8))
    lab[..., 0] = (0.6 * L * 255 + 0.4 * L8).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.05, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def portrait_crop(img: np.ndarray, face_bbox, landmarks, out_w: int = 800, out_h: int = 1000,
                  face_frac: float = 0.28, eye_line: float = 1 / 3, offset_x: float = 0.0) -> np.ndarray:
    """Crop to out_w:out_h with the eyes on the upper third and the face ~face_frac of the height.

    offset_x > 0 places the face slightly off centre, leaving more room on the side the person looks
    towards (from the nose position relative to the eyes).
    """
    H, W = img.shape[:2]
    aspect = out_w / out_h
    if face_bbox is None:
        ch = H
        cw = int(ch * aspect)
        x0 = (W - cw) // 2
        y0 = 0
    else:
        fx, fy, fw, fh = face_bbox
        eye = landmarks[:2].mean(axis=0) if landmarks is not None else np.array([fx + fw / 2, fy + fh * 0.4])
        ch = min(H, int(fh / face_frac))
        cw = int(ch * aspect)
        if cw > W:
            cw = W
            ch = int(cw / aspect)
        cx = eye[0]
        if offset_x and landmarks is not None and len(landmarks) >= 3:
            look = np.sign(landmarks[2][0] - eye[0])          # nose right of the eyes: looking right
            cx = eye[0] + look * offset_x * cw
        x0 = int(np.clip(cx - cw / 2, 0, W - cw))
        y0 = int(np.clip(eye[1] - eye_line * ch, 0, H - ch))
    crop = img[y0:y0 + ch, x0:x0 + cw]
    interp = cv2.INTER_AREA if ch > out_h else cv2.INTER_LANCZOS4
    return cv2.resize(crop, (out_w, out_h), interpolation=interp), (x0, y0, cw, ch)
