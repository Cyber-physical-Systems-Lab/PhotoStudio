"""Simulated shallow depth of field.

1. Person matte: MediaPipe multiclass selfie segmenter (hair, skin, clothes) if available,
   otherwise torchvision DeepLabV3 person class. Edges refined with a guided filter so hair
   gets a soft edge.
2. Relative depth: Depth Anything V2 (small) on the GPU. Fallback: distance from the person.
3. Blur radius grows with depth difference from the subject; the background is blurred with
   disc (bokeh-like) kernels using normalised convolution so the subject does not bleed into it.
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

CACHE = os.path.expanduser('~/.cache/nova5photo')
MP_MULTICLASS_URL = ('https://storage.googleapis.com/mediapipe-models/image_segmenter/'
                     'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite')


def _device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return None


# --------------------------------------------------------------------------- segmentation
def person_mask(img: np.ndarray, log=print) -> np.ndarray:
    for fn in (_mask_mediapipe, _mask_deeplab):
        try:
            m = fn(img)
            if m is not None and m.mean() > 0.01:
                log(f'segmentation: {fn.__name__}')
                return m
        except Exception as e:  # noqa: BLE001
            log(f'segmentation {fn.__name__} unavailable: {e}')
    raise RuntimeError('no segmentation backend available')


def _mask_mediapipe(img):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'selfie_multiclass_256x256.tflite')
    if not os.path.exists(path):
        urllib.request.urlretrieve(MP_MULTICLASS_URL, path)
    opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=path),
                                        output_confidence_masks=True)
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        bg = res.confidence_masks[0].numpy_view().astype(np.float32)
    return np.clip(1.0 - bg, 0, 1)


def _mask_deeplab(img):
    import torch
    import torchvision
    dev = _device()
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT').to(dev).eval()
    h, w = img.shape[:2]
    s = 520 / max(h, w)
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (int(w * s), int(h * s))).astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    t = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(dev)
    with torch.no_grad():
        out = model(t)['out'][0].softmax(0)[15].cpu().numpy()   # VOC class 15 = person
    return cv2.resize(out, (w, h))


def refine_matte(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Guided-filter refinement: keeps fine hair structure and makes a soft edge.

    (2026-11-19: a colour-guided band refinement and blur-fusion foreground estimation were tried and
    made hair worse - dark halo, strands cut off - so this stays the simple grey guided filter. A real
    matting model, e.g. BiRefNet, is the next step.)
    """
    h, w = mask.shape
    guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    r = max(4, int(0.006 * max(h, w)))
    try:
        m = cv2.ximgproc.guidedFilter(guide, mask.astype(np.float32), r, 1e-3)
    except AttributeError:
        m = _guided_filter(guide, mask.astype(np.float32), r, 1e-3)
    # remove isolated blobs far from the main body
    hard = (m > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard)
    if n > 2:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        far = (lab != keep) & (lab != 0)
        m[far] = 0
    return np.clip(m, 0, 1)


def _guided_filter(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mI, mp_ = cv2.boxFilter(I, -1, k), cv2.boxFilter(p, -1, k)
    cov = cv2.boxFilter(I * p, -1, k) - mI * mp_
    var = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cov / (var + eps)
    b = mp_ - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


# --------------------------------------------------------------------------- depth
def relative_depth(img: np.ndarray, mask: np.ndarray, log=print) -> np.ndarray:
    """Returns a 'defocus' map: 0 at the subject's depth, growing with distance behind it."""
    try:
        disp = _depth_anything(img)
        log('depth: Depth Anything V2 small')
        subj = float(np.median(disp[mask > 0.7])) if (mask > 0.7).any() else float(disp.max())
        far = float(np.percentile(disp, 2))
        d = np.clip((subj - disp) / max(1e-6, subj - far), 0, 1)
        return d
    except Exception as e:  # noqa: BLE001
        log(f'depth model unavailable ({e}); using distance from the person instead')
        hard = (mask > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)
        return np.clip(dist / (0.25 * max(img.shape[:2])), 0, 1) ** 0.5


_DA = None


def _depth_anything(img):
    global _DA
    import torch
    from PIL import Image
    from transformers import pipeline
    if _DA is None:
        _DA = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf',
                       device=0 if _device() == 'cuda' else -1)
    with torch.no_grad():
        out = _DA(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    disp = out['predicted_depth']
    disp = disp.squeeze().cpu().numpy() if hasattr(disp, 'cpu') else np.asarray(disp)
    return cv2.resize(disp.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- blur
def _smooth01(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def disc_kernel(r: int) -> np.ndarray:
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1.0, -1, lineType=cv2.LINE_AA)
    return k / k.sum()


def render_bokeh(img: np.ndarray, matte: np.ndarray, defocus: np.ndarray, max_radius_frac: float = 0.018,
                 face_bbox=None, hair_band_frac: float = 0.06):
    """Depth-graded background blur.

    Around the head the blur fades in with distance from the hair (none right at the strands, full
    blur hair_band_frac of the image size away): the coarse person matte cannot follow single hairs,
    and a hard switch from sharp hair to fully blurred background exposed that as a cut-out edge.
    """
    h, w = matte.shape
    rmax = max(3, int(max_radius_frac * max(h, w)))
    levels = sorted(set([0] + [max(1, int(rmax * f)) for f in (0.15, 0.3, 0.5, 0.75, 1.0)]))
    imgf = img.astype(np.float32) / 255
    lin = imgf ** 2.2                            # blur in linear light: brighter highlights
    bgw = np.clip(1.0 - matte, 0, 1) ** 2         # background weight, excludes the subject
    stack = []
    for r in levels:
        if r == 0:
            stack.append(lin)
            continue
        k = disc_kernel(r)
        num = cv2.filter2D(lin * bgw[..., None], -1, k)
        den = cv2.filter2D(bgw, -1, k)[..., None]
        blurred = np.where(den > 1e-3, num / np.maximum(den, 1e-3), cv2.filter2D(lin, -1, k))
        stack.append(blurred)
    radius = defocus * rmax
    # behind the subject's edge the background must be blurred even if depth says "near"
    radius = np.maximum(radius, (1 - matte) * 0.35 * rmax)
    if face_bbox is not None:
        fx, fy, fw, fh = [float(v) for v in face_bbox]
        person = (matte > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(1 - person, cv2.DIST_L2, 5)          # px from the person outline
        band = max(4.0, hair_band_frac * max(h, w))
        ramp = _smooth01(dist / band)                                      # 0 at the hair -> 1 further out
        # only around the head: rows above the chin, columns near the face (soft limits, no seams)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        chin = fy + 1.05 * fh
        rows = 1 - _smooth01((yy - chin) / (0.6 * fh))
        cols = 1 - _smooth01((np.abs(xx - (fx + fw / 2)) - 1.3 * fw) / (0.6 * fw))
        head = rows * cols
        radius = radius * (1 - head + head * ramp)
    out = np.zeros_like(lin)
    lv = np.array(levels, np.float32)
    idx = np.clip(np.searchsorted(lv, radius, side='right') - 1, 0, len(levels) - 2)
    lo, hi = lv[idx], lv[idx + 1]
    t = np.clip((radius - lo) / np.maximum(hi - lo, 1e-6), 0, 1)[..., None]
    for i in range(len(levels) - 1):
        sel = (idx == i)[..., None]
        out += np.where(sel, (1 - t) * stack[i] + t * stack[i + 1], 0)
    a = matte[..., None]
    comp = a * lin + (1 - a) * out
    return np.clip(comp ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)
