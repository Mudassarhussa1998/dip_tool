"""
core/utils/restoration.py
Noise models, noise removal filters, and DnCNN simulation.
"""
import time
import numpy as np
import cv2
from scipy.signal import wiener
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _ensure_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def compute_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    if orig.shape != proc.shape:
        proc = cv2.resize(proc.astype(np.uint8),
                          (orig.shape[1], orig.shape[0])).astype(np.float64)
    mse_val = float(np.mean((orig - proc) ** 2))
    psnr_val = float(10 * np.log10(255.0 ** 2 / mse_val)) if mse_val > 0 else 100.0
    try:
        if orig.ndim == 3:
            ssim_val = float(ssim_sk(orig.astype(np.uint8),
                                     proc.astype(np.uint8),
                                     channel_axis=2, data_range=255))
        else:
            ssim_val = float(ssim_sk(orig.astype(np.uint8),
                                     proc.astype(np.uint8), data_range=255))
    except Exception:
        ssim_val = 0.0
    return {'psnr': round(psnr_val, 3),
            'mse': round(mse_val, 3),
            'ssim': round(ssim_val, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Noise Models
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(img: np.ndarray, mean: float = 0,
                       sigma: float = 25) -> np.ndarray:
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, density: float = 0.05) -> np.ndarray:
    noisy = img.copy()
    total = img.size
    n_salt = int(total * density / 2)
    n_pepper = int(total * density / 2)
    coords = [np.random.randint(0, i, n_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, n_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy


def add_speckle_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.randn(*img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + img.astype(np.float32) * noise * 0.1
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_poisson_noise(img: np.ndarray) -> np.ndarray:
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img.astype(np.float32) / 255.0 * vals) / vals * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_periodic_noise(img: np.ndarray, freq: float = 0.1,
                       direction: str = 'horizontal') -> np.ndarray:
    h, w = img.shape[:2]
    if direction == 'horizontal':
        noise = np.sin(2 * np.pi * freq * np.arange(w)) * 30
        noise_2d = np.tile(noise, (h, 1))
    else:
        noise = np.sin(2 * np.pi * freq * np.arange(h)) * 30
        noise_2d = np.tile(noise.reshape(-1, 1), (1, w))
    if img.ndim == 3:
        noise_2d = np.stack([noise_2d] * 3, axis=2)
    noisy = img.astype(np.float32) + noise_2d.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Noise Removal Filters
# ─────────────────────────────────────────────────────────────────────────────

def remove_mean(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.blur(img, (ksize, ksize))


def remove_gaussian(img: np.ndarray, ksize: int = 5,
                    sigma: float = 1.0) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def remove_median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, ksize)


def remove_bilateral(img: np.ndarray, d: int = 9,
                     sigma_color: float = 75,
                     sigma_space: float = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def remove_wiener(img: np.ndarray, mysize: int = 5) -> np.ndarray:
    """Apply Wiener filter channel-wise."""
    if img.ndim == 3:
        channels = []
        for c in range(3):
            ch = img[:, :, c].astype(np.float64)
            filtered = wiener(ch, (mysize, mysize))
            channels.append(np.clip(filtered, 0, 255).astype(np.uint8))
        return cv2.merge(channels)
    filtered = wiener(img.astype(np.float64), (mysize, mysize))
    return np.clip(filtered, 0, 255).astype(np.uint8)


def remove_nlm(img: np.ndarray) -> np.ndarray:
    """Non-local means denoising."""
    if img.ndim == 3:
        img_float = img.astype(np.float32) / 255.0
        sigma_est = np.mean(estimate_sigma(img_float, channel_axis=2))
        patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=2)
        denoised = denoise_nl_means(img_float, h=1.15 * sigma_est,
                                    fast_mode=True, **patch_kw)
        return np.clip(denoised * 255, 0, 255).astype(np.uint8)
    img_float = img.astype(np.float32) / 255.0
    sigma_est = np.mean(estimate_sigma(img_float))
    denoised = denoise_nl_means(img_float, h=1.15 * sigma_est, fast_mode=True,
                                patch_size=5, patch_distance=6)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)


def remove_dncnn_simulation(img: np.ndarray) -> np.ndarray:
    """
    DnCNN simulation using residual learning approach.
    Approximates DnCNN behavior: estimate noise residual via Gaussian blur,
    subtract from noisy image (residual learning principle).
    """
    img_f = img.astype(np.float32)
    # Multi-scale residual estimation
    blur1 = cv2.GaussianBlur(img_f, (5, 5), 1.0)
    blur2 = cv2.GaussianBlur(img_f, (9, 9), 2.0)
    blur3 = cv2.GaussianBlur(img_f, (13, 13), 3.0)
    # Weighted combination (simulates deep feature aggregation)
    noise_estimate = 0.5 * (img_f - blur1) + 0.3 * (img_f - blur2) + 0.2 * (img_f - blur3)
    # Residual subtraction
    denoised = img_f - noise_estimate
    # Final refinement pass
    denoised = cv2.bilateralFilter(
        np.clip(denoised, 0, 255).astype(np.uint8), 5, 50, 50
    ).astype(np.float32)
    return np.clip(denoised, 0, 255).astype(np.uint8)


def adaptive_mean_filter(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Adaptive mean filter based on local variance."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    img_f = img.astype(np.float32)
    local_mean = cv2.blur(img_f, (ksize, ksize))
    local_sq_mean = cv2.blur(img_f ** 2, (ksize, ksize))
    local_var = local_sq_mean - local_mean ** 2
    noise_var = np.mean(local_var)
    ratio = np.clip(noise_var / (local_var + 1e-6), 0, 1)
    result = img_f - ratio * (img_f - local_mean)
    return np.clip(result, 0, 255).astype(np.uint8)


def adaptive_median_filter(img: np.ndarray, max_size: int = 7) -> np.ndarray:
    """Adaptive median filter with variable window size."""
    gray = _to_gray(img)
    result = gray.copy()
    h, w = gray.shape
    for y in range(h):
        for x in range(w):
            for s in range(3, max_size + 1, 2):
                half = s // 2
                y1, y2 = max(0, y - half), min(h, y + half + 1)
                x1, x2 = max(0, x - half), min(w, x + half + 1)
                patch = gray[y1:y2, x1:x2].flatten()
                z_min, z_max = patch.min(), patch.max()
                z_med = np.median(patch)
                if z_min < z_med < z_max:
                    z_xy = gray[y, x]
                    if z_min < z_xy < z_max:
                        result[y, x] = z_xy
                    else:
                        result[y, x] = int(z_med)
                    break
                if s == max_size:
                    result[y, x] = int(z_med)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else result


def compare_all_restoration(noisy: np.ndarray,
                             original: np.ndarray) -> dict:
    """Apply all denoising methods and return results + metrics."""
    methods = {
        'mean': lambda: remove_mean(noisy),
        'gaussian': lambda: remove_gaussian(noisy),
        'median': lambda: remove_median(noisy),
        'bilateral': lambda: remove_bilateral(noisy),
        'wiener': lambda: remove_wiener(noisy),
        'nlm': lambda: remove_nlm(noisy),
        'dncnn': lambda: remove_dncnn_simulation(noisy),
    }
    results = {}
    for name, fn in methods.items():
        t0 = time.time()
        try:
            denoised = fn()
        except Exception as e:
            denoised = noisy.copy()
        elapsed = (time.time() - t0) * 1000
        metrics = compute_metrics(original, denoised)
        results[name] = {
            'image': denoised,
            'time': round(elapsed, 2),
            'metrics': metrics,
        }
    return results
