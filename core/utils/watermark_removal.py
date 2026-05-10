"""
core/utils/watermark_removal.py
Watermark detection and removal — optimised for speed.

Techniques:
  - Inpainting: Telea (fast marching) & Navier-Stokes
  - Frequency notch filtering (FFT on luminance only)
  - Median fill
  - Gaussian fill
"""
import time
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Cap the longest edge to this many pixels before processing.
# Keeps inpainting fast even on large uploads.
MAX_SIDE = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _resize_if_large(img: np.ndarray, max_side: int = MAX_SIDE):
    """
    Downscale img so its longest side ≤ max_side.
    Returns (resized_img, scale_factor).  scale_factor=1.0 means no change.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _mask_coverage_pct(mask: np.ndarray) -> float:
    """Return percentage of pixels that are non-zero in the mask."""
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
    return round(float(np.count_nonzero(gray)) / gray.size * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Detection strategies  (all return uint8 single-channel mask, 0/255)
# ─────────────────────────────────────────────────────────────────────────────

def detect_watermark_bright(img: np.ndarray, threshold: int = 200,
                             dilate_iter: int = 3) -> np.ndarray:
    """Detect bright / white watermark regions."""
    gray = _to_gray(img)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def detect_watermark_dark(img: np.ndarray, threshold: int = 60,
                           dilate_iter: int = 3) -> np.ndarray:
    """Detect dark / black watermark regions."""
    gray = _to_gray(img)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def detect_watermark_edges(img: np.ndarray, dilate_iter: int = 4) -> np.ndarray:
    """Detect watermark via strong edge clusters (text / logo outlines)."""
    gray = _to_gray(img)
    edges = cv2.Canny(gray, 50, 150)
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    return edges


def detect_watermark_saturation(img: np.ndarray, sat_threshold: int = 30,
                                  dilate_iter: int = 3) -> np.ndarray:
    """Detect semi-transparent grey/white overlays via low HSV saturation."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    low_sat  = (s < sat_threshold).astype(np.uint8) * 255
    high_val = (v > 180).astype(np.uint8) * 255
    mask = cv2.bitwise_and(low_sat, high_val)
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def detect_watermark_frequency(img: np.ndarray) -> np.ndarray:
    """
    Detect periodic watermarks via FFT peak analysis.
    Operates on luminance only — much faster than per-channel.
    """
    gray = _to_gray(img).astype(np.float32)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    # Zero out DC
    magnitude[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5] = 0
    threshold = np.percentile(magnitude, 99)
    peak_mask = (magnitude > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    peak_mask = cv2.dilate(peak_mask, kernel, iterations=2)
    # Reconstruct without peaks → diff reveals watermark location
    fshift_clean = fshift * (1 - peak_mask)
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_clean)))
    diff = np.abs(gray - img_back)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, spatial_mask = cv2.threshold(diff_norm, 30, 255, cv2.THRESH_BINARY)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    spatial_mask = cv2.dilate(spatial_mask, kernel2, iterations=2)
    return spatial_mask


# ─────────────────────────────────────────────────────────────────────────────
# Removal methods
# ─────────────────────────────────────────────────────────────────────────────

def remove_inpaint_telea(img: np.ndarray, mask: np.ndarray,
                          radius: int = 5) -> np.ndarray:
    """Fast marching (Telea) inpainting."""
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)


def remove_inpaint_ns(img: np.ndarray, mask: np.ndarray,
                       radius: int = 5) -> np.ndarray:
    """Navier-Stokes inpainting."""
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)


def remove_frequency_notch(img: np.ndarray, notch_radius: int = 10) -> np.ndarray:
    """
    Remove periodic watermarks via frequency-domain notch filtering.
    Processes luminance channel only, then recompose — ~3× faster than
    per-channel FFT.
    """
    # Work in YCrCb so we only FFT the luma channel
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    fshift = np.fft.fftshift(np.fft.fft2(y))
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    magnitude[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5] = 0
    threshold = np.percentile(magnitude, 99.5)
    peak_mask = (magnitude > threshold).astype(np.uint8)
    kr = notch_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr))
    peak_mask = cv2.dilate(peak_mask, kernel, iterations=1)

    fshift_filtered = fshift * (1 - peak_mask)
    y_clean = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
    ycrcb[:, :, 0] = np.clip(y_clean, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def remove_median_fill(img: np.ndarray, mask: np.ndarray,
                        ksize: int = 15) -> np.ndarray:
    """Replace masked pixels with local median — fast, good for small marks."""
    # ksize must be odd
    if ksize % 2 == 0:
        ksize += 1
    result = img.copy()
    blurred = cv2.medianBlur(img, ksize)
    result[mask > 0] = blurred[mask > 0]
    return result


def remove_gaussian_fill(img: np.ndarray, mask: np.ndarray,
                          sigma: float = 15.0) -> np.ndarray:
    """Replace masked pixels with Gaussian-blurred background."""
    result = img.copy()
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    result[mask > 0] = blurred[mask > 0]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _build_mask(img: np.ndarray, detection_method: str,
                threshold: int, dilate_iter: int) -> np.ndarray:
    """Dispatch to the correct detection function."""
    dispatch = {
        'bright':     lambda: detect_watermark_bright(img, threshold, dilate_iter),
        'dark':       lambda: detect_watermark_dark(img, threshold, dilate_iter),
        'edges':      lambda: detect_watermark_edges(img, dilate_iter),
        'saturation': lambda: detect_watermark_saturation(img, dilate_iter=dilate_iter),
        'frequency':  lambda: detect_watermark_frequency(img),
    }
    return dispatch.get(detection_method, dispatch['bright'])()


def remove_watermark(img: np.ndarray,
                     detection_method: str = 'bright',
                     removal_method: str = 'telea',
                     threshold: int = 200,
                     dilate_iter: int = 3,
                     inpaint_radius: int = 5,
                     notch_radius: int = 10) -> dict:
    """
    Full watermark removal pipeline.

    Automatically downscales large images to MAX_SIDE before processing
    and upscales the result back to the original resolution.

    Returns:
        result  — cleaned image (same size as input)
        mask    — BGR visualisation of the detected mask (same size as input)
        processing_time_ms — float
        mask_coverage_pct  — float
    """
    t0 = time.time()
    orig_h, orig_w = img.shape[:2]

    # ── Downscale for speed ──────────────────────────────────────────────────
    work_img, scale = _resize_if_large(img)

    # ── Detect ──────────────────────────────────────────────────────────────
    mask = _build_mask(work_img, detection_method, threshold, dilate_iter)

    # ── Remove ──────────────────────────────────────────────────────────────
    removal_dispatch = {
        'telea':           lambda: remove_inpaint_telea(work_img, mask, inpaint_radius),
        'ns':              lambda: remove_inpaint_ns(work_img, mask, inpaint_radius),
        'frequency_notch': lambda: remove_frequency_notch(work_img, notch_radius),
        'median':          lambda: remove_median_fill(work_img, mask),
        'gaussian':        lambda: remove_gaussian_fill(work_img, mask),
    }
    result_small = removal_dispatch.get(removal_method,
                                        removal_dispatch['telea'])()

    # ── Upscale back to original size ────────────────────────────────────────
    if scale < 1.0:
        result = cv2.resize(result_small, (orig_w, orig_h),
                            interpolation=cv2.INTER_LANCZOS4)
        mask_full = cv2.resize(mask, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    else:
        result = result_small
        mask_full = mask

    mask_bgr = _ensure_bgr(mask_full)
    coverage = _mask_coverage_pct(mask_bgr)

    return {
        'result': result,
        'mask': mask_bgr,
        'processing_time_ms': round((time.time() - t0) * 1000, 2),
        'mask_coverage_pct': coverage,
    }


def compare_all_methods(img: np.ndarray,
                         detection_method: str = 'bright',
                         threshold: int = 200,
                         dilate_iter: int = 3) -> tuple:
    """
    Run all 5 removal methods with the same detection mask.
    Returns (results_dict, mask_bgr).
    """
    orig_h, orig_w = img.shape[:2]
    work_img, scale = _resize_if_large(img)

    mask = _build_mask(work_img, detection_method, threshold, dilate_iter)

    methods = {
        'telea':           lambda: remove_inpaint_telea(work_img, mask),
        'ns':              lambda: remove_inpaint_ns(work_img, mask),
        'median':          lambda: remove_median_fill(work_img, mask),
        'gaussian':        lambda: remove_gaussian_fill(work_img, mask),
        'frequency_notch': lambda: remove_frequency_notch(work_img),
    }

    results = {}
    for name, fn in methods.items():
        t0 = time.time()
        try:
            out_small = fn()
        except Exception:
            out_small = work_img.copy()
        elapsed = round((time.time() - t0) * 1000, 2)
        # Upscale if needed
        if scale < 1.0:
            out = cv2.resize(out_small, (orig_w, orig_h),
                             interpolation=cv2.INTER_LANCZOS4)
        else:
            out = out_small
        results[name] = {'image': out, 'time': elapsed}

    # Full-size mask
    if scale < 1.0:
        mask_full = cv2.resize(mask, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    else:
        mask_full = mask

    return results, _ensure_bgr(mask_full)
