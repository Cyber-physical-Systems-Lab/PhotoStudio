import cv2
import numpy as np

from portrait_post.dof import disc_kernel, refine_matte, render_bokeh
from portrait_post.tone import portrait_crop


def test_bokeh_keeps_subject_sharp_and_blurs_background():
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 255, (200, 300, 3))).astype(np.uint8)
    matte = np.zeros((200, 300), np.float32)
    matte[50:150, 100:200] = 1.0
    defocus = 1.0 - matte
    out = render_bokeh(img, matte, defocus, 0.03)
    inside = np.abs(out[70:130, 120:180].astype(int) - img[70:130, 120:180].astype(int)).mean()
    outside_var = out[:, :60].astype(float).var()
    assert inside < 2.0
    assert outside_var < img[:, :60].astype(float).var() * 0.3


def test_disc_kernel_normalised():
    assert abs(disc_kernel(5).sum() - 1.0) < 1e-5


def test_refine_matte_range():
    img = np.full((100, 100, 3), 128, np.uint8)
    m = np.zeros((100, 100), np.float32)
    m[30:70, 30:70] = 1
    r = refine_matte(img, m)
    assert r.min() >= 0 and r.max() <= 1


def test_crop_aspect_and_eye_line():
    img = np.zeros((1080, 1920, 3), np.uint8)
    bbox = np.array([900, 520, 160, 180], float)
    lm = np.array([[950, 600], [1050, 600], [1000, 660], [960, 700], [1040, 700]], float)
    out, (x0, y0, cw, ch) = portrait_crop(img, bbox, lm, 800, 1000)
    assert out.shape == (1000, 800, 3)
    assert abs(cw / ch - 0.8) < 0.01
    assert abs((600 - y0) / ch - 1 / 3) < 0.02
