"""
core/utils/frequency.py
FFT / DFT frequency domain analysis and filtering.
"""
import numpy as np
import cv2
import time


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def compute_fft(img: np.ndarray) -> dict:
    """Compute FFT and return magnitude, phase, power spectra as uint8 images."""
    gray = _to_gray(img).astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)
    power = magnitude ** 2

    # Log scale for display
    mag_log = np.log1p(magnitude)
    mag_norm = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    phase_norm = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    pow_log = np.log1p(power)
    pow_norm = cv2.normalize(pow_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 1D cross-section through center
    center_row = mag_log[mag_log.shape[0] // 2, :]
    cross_section = center_row.tolist()

    return {
        'magnitude': cv2.cvtColor(mag_norm, cv2.COLOR_GRAY2BGR),
        'phase': cv2.cvtColor(phase_norm, cv2.COLOR_GRAY2BGR),
        'power': cv2.cvtColor(pow_norm, cv2.COLOR_GRAY2BGR),
        'cross_section': cross_section,
        'dft_shift': dft_shift,  # complex array for filtering
    }


def _make_filter_mask(shape: tuple, filter_type: str,
                      cutoff_low: float, cutoff_high: float,
                      order: int = 2) -> np.ndarray:
    """Create a frequency domain filter mask."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_d = np.sqrt(crow ** 2 + ccol ** 2)
    d_norm = d / (max_d + 1e-6)
    D0 = cutoff_low / 100.0
    D1 = cutoff_high / 100.0

    if filter_type == 'ideal_lp':
        mask = (d_norm <= D0).astype(np.float32)
    elif filter_type == 'ideal_hp':
        mask = (d_norm > D0).astype(np.float32)
    elif filter_type == 'butter_lp':
        mask = 1.0 / (1.0 + (d_norm / (D0 + 1e-9)) ** (2 * order))
    elif filter_type == 'butter_hp':
        mask = 1.0 / (1.0 + ((D0 + 1e-9) / (d_norm + 1e-9)) ** (2 * order))
    elif filter_type == 'gauss_lp':
        mask = np.exp(-(d_norm ** 2) / (2 * D0 ** 2 + 1e-9))
    elif filter_type == 'gauss_hp':
        mask = 1.0 - np.exp(-(d_norm ** 2) / (2 * D0 ** 2 + 1e-9))
    elif filter_type == 'bandpass':
        mask = ((d_norm >= D0) & (d_norm <= D1)).astype(np.float32)
    elif filter_type == 'notch':
        # Simple notch: reject a band around D0
        w = 0.05
        mask = 1.0 - np.exp(-((d_norm - D0) ** 2) / (2 * w ** 2))
    else:
        mask = np.ones((rows, cols), dtype=np.float32)

    return mask


def apply_frequency_filter(img: np.ndarray, filter_type: str,
                            cutoff_low: float = 30.0,
                            cutoff_high: float = 70.0,
                            order: int = 2) -> dict:
    """Apply frequency domain filter and return filtered spectrum + reconstructed image."""
    t0 = time.time()
    gray = _to_gray(img).astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    mask = _make_filter_mask(gray.shape, filter_type, cutoff_low, cutoff_high, order)

    filtered_dft = dft_shift * mask

    # Filtered spectrum for display
    mag_filtered = np.log1p(np.abs(filtered_dft))
    spec_norm = cv2.normalize(mag_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Reconstruct
    idft_shift = np.fft.ifftshift(filtered_dft)
    reconstructed = np.abs(np.fft.ifft2(idft_shift))
    reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Filter transfer function as image
    mask_norm = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 1D cross-section of filter
    center_row = mask[mask.shape[0] // 2, :]
    cross_section = center_row.tolist()

    elapsed = (time.time() - t0) * 1000

    return {
        'filtered_spectrum': cv2.cvtColor(spec_norm, cv2.COLOR_GRAY2BGR),
        'reconstructed': cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2BGR),
        'filter_mask': cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR),
        'cross_section': cross_section,
        'processing_time_ms': round(elapsed, 2),
    }
