"""
core/utils/segmentation.py
Image segmentation: thresholding, region-based, edge-based, Hough transforms.
"""
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Colorize a label map for display."""
    n_labels = labels.max() + 1
    colors = np.random.randint(0, 255, (n_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    colored = colors[labels]
    return colored.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Point & Line Detection
# ─────────────────────────────────────────────────────────────────────────────

def point_detection(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img).astype(np.float32)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)
    result = cv2.filter2D(gray, -1, kernel)
    result = np.abs(result)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


def line_detection(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    gray = _to_gray(img).astype(np.float32)
    kernels = {
        'horizontal': np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32),
        'vertical':   np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32),
        'diagonal45': np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32),
        'diagonal_45': np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32),
    }
    kernel = kernels.get(direction, kernels['horizontal'])
    result = cv2.filter2D(gray, -1, kernel)
    result = np.abs(result)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result


def hough_lines(img: np.ndarray, threshold: int = 100) -> dict:
    gray = _to_gray(img)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Hough accumulator visualization
    h, w = gray.shape
    accumulator = np.zeros((180, max(h, w) * 2), dtype=np.uint32)
    for y in range(h):
        for x in range(w):
            if edges[y, x] > 0:
                for t in range(0, 180, 2):
                    theta = t * np.pi / 180
                    rho = int(x * np.cos(theta) + y * np.sin(theta))
                    rho_idx = rho + max(h, w)
                    if 0 <= rho_idx < accumulator.shape[1]:
                        accumulator[t, rho_idx] += 1
    acc_norm = cv2.normalize(accumulator.astype(np.float32), None,
                             0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {
        'overlay': overlay,
        'edges': cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
        'accumulator': cv2.cvtColor(acc_norm, cv2.COLOR_GRAY2BGR),
        'line_count': len(lines) if lines is not None else 0,
    }


def hough_circles(img: np.ndarray, min_radius: int = 10,
                  max_radius: int = 100) -> dict:
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=20, param1=50, param2=30,
                               minRadius=min_radius, maxRadius=max_radius)
    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = len(circles[0])
        for c in circles[0]:
            cv2.circle(overlay, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(overlay, (c[0], c[1]), 2, (0, 0, 255), 3)
    return {'overlay': overlay, 'circle_count': count}


# ─────────────────────────────────────────────────────────────────────────────
# Thresholding
# ─────────────────────────────────────────────────────────────────────────────

def global_threshold(img: np.ndarray, thresh: int = 127) -> np.ndarray:
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def otsu_threshold(img: np.ndarray) -> dict:
    gray = _to_gray(img)
    thresh_val, binary = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        'image': cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        'threshold': int(thresh_val),
    }


def adaptive_threshold(img: np.ndarray, block_size: int = 11,
                       C: int = 2) -> np.ndarray:
    gray = _to_gray(img)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, C)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def multi_level_threshold(img: np.ndarray, n_levels: int = 3) -> np.ndarray:
    gray = _to_gray(img)
    try:
        thresholds = threshold_multiotsu(gray, classes=n_levels)
        regions = np.digitize(gray, bins=thresholds)
        colored = _colorize_labels(regions)
        return colored
    except Exception:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Region-Based Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def region_growing(img: np.ndarray, seed_x: int, seed_y: int,
                   tolerance: int = 20) -> np.ndarray:
    gray = _to_gray(img)
    h, w = gray.shape
    seed_x = max(0, min(seed_x, w - 1))
    seed_y = max(0, min(seed_y, h - 1))
    seed_val = int(gray[seed_y, seed_x])
    visited = np.zeros((h, w), dtype=bool)
    region = np.zeros((h, w), dtype=np.uint8)
    stack = [(seed_y, seed_x)]
    while stack:
        y, x = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        if abs(int(gray[y, x]) - seed_val) <= tolerance:
            region[y, x] = 255
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    stack.append((ny, nx))
    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[region > 0] = [0, 255, 0]
    cv2.circle(overlay, (seed_x, seed_y), 5, (0, 0, 255), -1)
    return overlay


def watershed_segmentation(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    bgr = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(bgr, markers)
    result = bgr.copy()
    result[markers == -1] = [0, 0, 255]
    return result


def edge_based_segmentation(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(closed))
    colored = _colorize_labels(labels)
    return colored
