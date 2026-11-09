import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread replacement that also works with non-ASCII Windows paths."""
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def fit_cornea(bgr, margin=0.10):
    """Fit the red annotation ellipse; return ellipse, crop box and validity."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = ((r > 90) & (r - g > 40) & (r - b > 40)).astype(np.uint8)
    pts = cv2.findNonZero(red)
    if pts is not None and len(pts) >= 100:
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(pts)
        if min(MA, ma) >= 100:
            t = np.deg2rad(ang)
            a, bb = MA / 2.0, ma / 2.0
            ex = float(np.sqrt(a * a * np.cos(t) ** 2 + bb * bb * np.sin(t) ** 2))
            ey = float(np.sqrt(a * a * np.sin(t) ** 2 + bb * bb * np.cos(t) ** 2))
            x0 = int(max(0, cx - ex * (1 + margin)))
            x1 = int(min(w, cx + ex * (1 + margin)))
            y0 = int(max(0, cy - ey * (1 + margin)))
            y1 = int(min(h, cy + ey * (1 + margin)))
            return {
                "ellipse": [float(cx), float(cy), float(MA), float(ma), float(ang)],
                "crop": [x0, y0, x1, y1],
                "valid": True,
            }
    return {
        "ellipse": [w / 2.0, h / 2.0, float(w * 0.9), float(h * 0.9), 0.0],
        "crop": [0, 0, w, h],
        "valid": False,
    }


def cornea_roi(shape, ellipse, dilate=0):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cx, cy, MA, ma, ang = ellipse
    cv2.ellipse(mask, ((cx, cy), (MA, ma), ang), 1, -1)
    if dilate:
        k = 2 * int(dilate) + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def erase_white_annotation(bgr, ellipse=None, margin=20, white_min=165, white_chroma=35):
    """Remove the near-white hand-drawn lesion outline by local inpainting."""
    mn = bgr.min(axis=2)
    mx = bgr.max(axis=2)
    white = ((mn > white_min) & ((mx - mn) < white_chroma)).astype(np.uint8)
    if ellipse is not None:
        white &= cornea_roi(bgr.shape[:2], ellipse, margin)
    ys, xs = np.nonzero(white)
    if len(ys) == 0:
        return bgr
    h, w = bgr.shape[:2]
    y0, y1 = max(0, int(ys.min()) - 8), min(h, int(ys.max()) + 9)
    x0, x1 = max(0, int(xs.min()) - 8), min(w, int(xs.max()) + 9)
    mask = cv2.dilate(white[y0:y1, x0:x1] * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    crop = bgr[y0:y1, x0:x1]
    out = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
    bgr = bgr.copy()
    bgr[y0:y1, x0:x1] = out
    return bgr
