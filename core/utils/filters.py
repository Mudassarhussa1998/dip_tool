"""
core/utils/filters.py
All spatial filtering, point processing, and enhancement functions.
"""
import time
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import mean_squared_error as mse_metric


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """Compute PSNR, MSE, SSIM between two images."""
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    if orig.shape != proc.shape:
        proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
        proc = proc.astype(np.float64)
    try:
        mse_val = float(np.mean((orig - proc) ** 2))
        if mse_val == 0:
            psnr_val = 100.0
        else:
            psnr_val = float(10 * np.log10((255.0 ** 2) / mse_val))
        if orig.ndim == 3:
            ssim_val = float(ssim_metric(orig.astype(np.uint8),
                                         proc.astype(np.uint8),
                                         channel_axis=2,
                                         data_range=255))
        else:
            ssim_val = float(ssim_metric(orig.astype(np.uint8),
                                         proc.astype(np.uint8),
                                         data_range=255))
    except Exception:
        mse_val, psnr_val, ssim_val = 0.0, 0.0, 0.0
    return {'psnr': round(psnr_val, 4),
            'mse': round(mse_val, 4),
            'ssim': round(ssim_val, 4)}


def safe_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — Intro / basic properties
# ─────────────────────────────────────────────────────────────────────────────

def get_image_properties(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'bit_depth': 8,
        'pixel_min': int(img.min()),
        'pixel_max': int(img.max()),
        'mean': round(float(img.mean()), 2),
        'std': round(float(img.std()), 2),
    }


def split_channels(img: np.ndarray):
    """Return R, G, B channel images as uint8 arrays."""
    if img.ndim == 2:
        return img, img, img
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(r)
    r_img = cv2.merge([zeros, zeros, r])
    g_img = cv2.merge([zeros, g, zeros])
    b_img = cv2.merge([b, zeros, zeros])
    return r_img, g_img, b_img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — Sampling & Quantization
# ─────────────────────────────────────────────────────────────────────────────

INTERP_MAP = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
}


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h, new_w = max(1, h // factor), max(1, w // factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def upsample(img: np.ndarray, target_size: tuple, method: str = 'bilinear') -> np.ndarray:
    interp = INTERP_MAP.get(method, cv2.INTER_LINEAR)
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=interp)


def quantize(img: np.ndarray, bits: int) -> np.ndarray:
    """Reduce image to `bits` bits per channel."""
    if bits >= 8:
        return img.copy()
    levels = 2 ** bits
    factor = 256 // levels
    quantized = (img.astype(np.uint16) // factor) * factor
    return safe_uint8(quantized)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — Enhancement & Point Processing
# ─────────────────────────────────────────────────────────────────────────────

def brightness(img: np.ndarray, value: float) -> np.ndarray:
    return safe_uint8(img.astype(np.float32) + value)


def contrast(img: np.ndarray, factor: float) -> np.ndarray:
    return safe_uint8(img.astype(np.float32) * factor)


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def log_transform(img: np.ndarray) -> np.ndarray:
    c = 255.0 / np.log1p(img.max() + 1e-6)
    result = c * np.log1p(img.astype(np.float32))
    return safe_uint8(result)


def negative_image(img: np.ndarray) -> np.ndarray:
    return 255 - img


def threshold_image(img: np.ndarray, thresh: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else binary


def piecewise_linear_stretch(img: np.ndarray,
                              r1: int, s1: int,
                              r2: int, s2: int) -> np.ndarray:
    """Piecewise linear contrast stretching with two breakpoints."""
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        if i < r1:
            lut[i] = (s1 / max(r1, 1)) * i
        elif i < r2:
            lut[i] = s1 + ((s2 - s1) / max(r2 - r1, 1)) * (i - r1)
        else:
            lut[i] = s2 + ((255 - s2) / max(255 - r2, 1)) * (i - r2)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def mean_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.blur(img, (ksize, ksize))


def gaussian_filter(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, ksize)


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def scale_image(img: np.ndarray, fx: float, fy: float) -> np.ndarray:
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


def flip_image(img: np.ndarray, direction: str) -> np.ndarray:
    if direction == 'horizontal':
        return cv2.flip(img, 1)
    elif direction == 'vertical':
        return cv2.flip(img, 0)
    return cv2.flip(img, -1)


def translate_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h))


def shear_image(img: np.ndarray, shx: float, shy: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, shx, 0], [shy, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Bit Plane Slicing & Watermarking
# ─────────────────────────────────────────────────────────────────────────────

def get_bit_planes(img: np.ndarray) -> list:
    """Return list of 8 bit-plane images (plane 0 = LSB, plane 7 = MSB)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    planes = []
    for bit in range(8):
        plane = ((gray >> bit) & 1) * 255
        planes.append(plane.astype(np.uint8))
    return planes


def reconstruct_from_planes(img: np.ndarray, selected_bits: list) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    result = np.zeros_like(gray, dtype=np.uint16)
    for bit in selected_bits:
        result += ((gray >> bit) & 1).astype(np.uint16) * (2 ** bit)
    return np.clip(result, 0, 255).astype(np.uint8)


def embed_text_watermark_lsb(img: np.ndarray, text: str) -> np.ndarray:
    """Embed text into LSB of grayscale image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    flat = gray.flatten().astype(np.uint8)
    bits = ''.join(format(ord(c), '08b') for c in text) + '00000000'  # null terminator
    if len(bits) > len(flat):
        bits = bits[:len(flat)]
    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)
    result = flat.reshape(gray.shape)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else result


def extract_text_watermark_lsb(img: np.ndarray, length: int = 50) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    flat = gray.flatten()
    bits = [str(p & 1) for p in flat[:length * 8]]
    chars = []
    for i in range(0, len(bits), 8):
        byte = ''.join(bits[i:i + 8])
        val = int(byte, 2)
        if val == 0:
            break
        chars.append(chr(val))
    return ''.join(chars)


def add_visible_watermark(img: np.ndarray, text: str, alpha: float = 0.4) -> np.ndarray:
    overlay = img.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 400)
    thickness = max(1, int(w / 200))
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2
    cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness + 2)
    cv2.putText(overlay, text, (x, y), font, font_scale, (0, 0, 0), thickness)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — Histogram Processing
# ─────────────────────────────────────────────────────────────────────────────

def compute_histogram(img: np.ndarray) -> dict:
    """Return histogram data for Chart.js."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
    cdf = np.cumsum(hist)
    cdf_norm = (cdf / cdf[-1] * 255).tolist()
    return {'histogram': hist, 'cdf': cdf_norm, 'labels': list(range(256))}


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return cv2.equalizeHist(img)


def histogram_matching(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match histogram of src to ref."""
    def _match_channel(s, r):
        s_hist, _ = np.histogram(s.flatten(), 256, [0, 256])
        r_hist, _ = np.histogram(r.flatten(), 256, [0, 256])
        s_cdf = s_hist.cumsum() / s_hist.sum()
        r_cdf = r_hist.cumsum() / r_hist.sum()
        lut = np.zeros(256, dtype=np.uint8)
        r_idx = 0
        for s_idx in range(256):
            while r_idx < 255 and r_cdf[r_idx] < s_cdf[s_idx]:
                r_idx += 1
            lut[s_idx] = r_idx
        return cv2.LUT(s, lut)

    if src.ndim == 3:
        channels = []
        for c in range(3):
            ref_c = ref[:, :, c] if ref.ndim == 3 else ref
            channels.append(_match_channel(src[:, :, c], ref_c))
        return cv2.merge(channels)
    return _match_channel(src, ref)


def local_histogram_eq(img: np.ndarray, tile_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_size, tile_size))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else eq


def clahe_equalization(img: np.ndarray, clip_limit: float = 2.0,
                       tile_size: int = 8) -> np.ndarray:
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                 tileGridSize=(tile_size, tile_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — Spatial Filtering & Smoothing
# ─────────────────────────────────────────────────────────────────────────────

BORDER_MAP = {
    'zero': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'wrap': cv2.BORDER_WRAP,
}


def apply_custom_kernel(img: np.ndarray, kernel: np.ndarray,
                        mode: str = 'convolution',
                        border: str = 'reflect') -> np.ndarray:
    border_type = BORDER_MAP.get(border, cv2.BORDER_REFLECT)
    k = kernel.astype(np.float32)
    if mode == 'convolution':
        k = np.flip(k)
    result = cv2.filter2D(img, -1, k, borderType=border_type)
    return safe_uint8(result)


def bilateral_filter(img: np.ndarray, d: int = 9,
                     sigma_color: float = 75,
                     sigma_space: float = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def compare_smoothing_filters(img: np.ndarray, ksize: int = 5,
                               sigma: float = 1.0,
                               border: str = 'reflect') -> dict:
    """Apply all 4 smoothing filters and return results + metrics."""
    border_type = BORDER_MAP.get(border, cv2.BORDER_REFLECT)
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    t0 = time.time()
    mean_r = cv2.blur(img, (ksize, ksize), borderType=border_type)
    t_mean = (time.time() - t0) * 1000

    t0 = time.time()
    gauss_r = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=border_type)
    t_gauss = (time.time() - t0) * 1000

    t0 = time.time()
    med_r = cv2.medianBlur(img, ksize)
    t_med = (time.time() - t0) * 1000

    t0 = time.time()
    bil_r = cv2.bilateralFilter(img, 9, 75, 75)
    t_bil = (time.time() - t0) * 1000

    return {
        'mean': {'image': mean_r, 'time': t_mean,
                 'metrics': compute_metrics(img, mean_r)},
        'gaussian': {'image': gauss_r, 'time': t_gauss,
                     'metrics': compute_metrics(img, gauss_r)},
        'median': {'image': med_r, 'time': t_med,
                   'metrics': compute_metrics(img, med_r)},
        'bilateral': {'image': bil_r, 'time': t_bil,
                      'metrics': compute_metrics(img, bil_r)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — Sharpening & Edge Detection
# ─────────────────────────────────────────────────────────────────────────────

def laplacian_sharpening(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray_f = gray.astype(np.float32)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    sharpened = gray_f - alpha * lap
    result = safe_uint8(sharpened)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else result


def unsharp_mask(img: np.ndarray, radius: int = 5, amount: float = 1.5) -> np.ndarray:
    radius = radius if radius % 2 == 1 else radius + 1
    blurred = cv2.GaussianBlur(img, (radius, radius), 0)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return safe_uint8(sharpened)


def high_boost_filter(img: np.ndarray, boost: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    result = img.astype(np.float32) + boost * mask
    return safe_uint8(result)


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def sobel_edges(img: np.ndarray) -> dict:
    gray = _to_gray(img)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return {
        'x': safe_uint8(np.abs(sx)),
        'y': safe_uint8(np.abs(sy)),
        'combined': safe_uint8(mag / mag.max() * 255 if mag.max() > 0 else mag),
    }


def prewitt_edges(img: np.ndarray) -> dict:
    gray = _to_gray(img)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    px = cv2.filter2D(gray.astype(np.float32), -1, kx)
    py = cv2.filter2D(gray.astype(np.float32), -1, ky)
    mag = np.sqrt(px ** 2 + py ** 2)
    return {
        'x': safe_uint8(np.abs(px)),
        'y': safe_uint8(np.abs(py)),
        'combined': safe_uint8(mag / mag.max() * 255 if mag.max() > 0 else mag),
    }


def roberts_edges(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    rx = cv2.filter2D(gray.astype(np.float32), -1, kx)
    ry = cv2.filter2D(gray.astype(np.float32), -1, ky)
    mag = np.sqrt(rx ** 2 + ry ** 2)
    return safe_uint8(mag / mag.max() * 255 if mag.max() > 0 else mag)


def log_edges(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    gray = _to_gray(img).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_32F)
    result = np.abs(lap)
    return safe_uint8(result / result.max() * 255 if result.max() > 0 else result)


def canny_edges(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    gray = _to_gray(img)
    return cv2.Canny(gray, low, high)


def all_edge_detectors(img: np.ndarray, canny_low: int = 50,
                       canny_high: int = 150) -> dict:
    """Run all edge detectors and return results + stats."""
    t0 = time.time()
    sob = sobel_edges(img)
    t_sob = (time.time() - t0) * 1000

    t0 = time.time()
    pre = prewitt_edges(img)
    t_pre = (time.time() - t0) * 1000

    t0 = time.time()
    rob = roberts_edges(img)
    t_rob = (time.time() - t0) * 1000

    t0 = time.time()
    log_r = log_edges(img)
    t_log = (time.time() - t0) * 1000

    t0 = time.time()
    can = canny_edges(img, canny_low, canny_high)
    t_can = (time.time() - t0) * 1000

    gray = _to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_8U)

    def edge_count(e):
        return int(np.count_nonzero(e > 30))

    return {
        'sobel_x': sob['x'],
        'sobel_y': sob['y'],
        'sobel': sob['combined'],
        'prewitt_x': pre['x'],
        'prewitt_y': pre['y'],
        'prewitt': pre['combined'],
        'roberts': rob,
        'laplacian': lap,
        'log': log_r,
        'canny': can,
        'times': {
            'sobel': round(t_sob, 2),
            'prewitt': round(t_pre, 2),
            'roberts': round(t_rob, 2),
            'log': round(t_log, 2),
            'canny': round(t_can, 2),
        },
        'edge_counts': {
            'sobel': edge_count(sob['combined']),
            'prewitt': edge_count(pre['combined']),
            'roberts': edge_count(rob),
            'laplacian': edge_count(lap),
            'log': edge_count(log_r),
            'canny': edge_count(can),
        },
    }
