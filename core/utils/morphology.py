"""
core/utils/morphology.py
Morphological operations on binary and grayscale images.
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize as sk_skeletonize


def _get_structuring_element(shape: str, size: int) -> np.ndarray:
    size = size if size % 2 == 1 else size + 1
    shape_map = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS,
    }
    morph_shape = shape_map.get(shape, cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (size, size))


def binarize(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def erosion(img: np.ndarray, se_shape: str = 'rect',
            se_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = _get_structuring_element(se_shape, se_size)
    return cv2.erode(img, kernel, iterations=iterations)


def dilation(img: np.ndarray, se_shape: str = 'rect',
             se_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = _get_structuring_element(se_shape, se_size)
    return cv2.dilate(img, kernel, iterations=iterations)


def opening(img: np.ndarray, se_shape: str = 'rect',
            se_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = _get_structuring_element(se_shape, se_size)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,
                            iterations=iterations)


def closing(img: np.ndarray, se_shape: str = 'rect',
            se_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = _get_structuring_element(se_shape, se_size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,
                            iterations=iterations)


def morphological_gradient(img: np.ndarray, se_shape: str = 'rect',
                            se_size: int = 3) -> np.ndarray:
    kernel = _get_structuring_element(se_shape, se_size)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def top_hat(img: np.ndarray, se_shape: str = 'rect',
            se_size: int = 9) -> np.ndarray:
    kernel = _get_structuring_element(se_shape, se_size)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def black_hat(img: np.ndarray, se_shape: str = 'rect',
              se_size: int = 9) -> np.ndarray:
    kernel = _get_structuring_element(se_shape, se_size)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def boundary_extraction(img: np.ndarray, se_shape: str = 'rect',
                        se_size: int = 3) -> np.ndarray:
    eroded = erosion(img, se_shape, se_size)
    return cv2.subtract(img, eroded)


def hole_filling(img: np.ndarray) -> np.ndarray:
    """Fill holes in binary image using flood fill."""
    binary = img.copy()
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    flood = thresh.copy()
    h, w = thresh.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(thresh, flood_inv)


def skeletonize(img: np.ndarray) -> np.ndarray:
    """Morphological skeletonization."""
    binary = img.copy()
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    bool_img = thresh > 0
    skeleton = sk_skeletonize(bool_img)
    return (skeleton * 255).astype(np.uint8)


def apply_all_morphology(img: np.ndarray, threshold: int = 127,
                         se_shape: str = 'rect', se_size: int = 3,
                         iterations: int = 1) -> dict:
    """Apply all 8 morphological operations and return results."""
    binary = binarize(img, threshold)
    return {
        'binary': binary,
        'erosion': erosion(binary, se_shape, se_size, iterations),
        'dilation': dilation(binary, se_shape, se_size, iterations),
        'opening': opening(binary, se_shape, se_size, iterations),
        'closing': closing(binary, se_shape, se_size, iterations),
        'gradient': morphological_gradient(binary, se_shape, se_size),
        'tophat': top_hat(binary, se_shape, se_size),
        'blackhat': black_hat(binary, se_shape, se_size),
        'boundary': boundary_extraction(binary, se_shape, se_size),
        'holes': hole_filling(binary),
        'skeleton': skeletonize(binary),
    }
