"""
core/utils/compression.py
Image compression: JPEG quality, DCT visualization, RLE, Huffman demo.
"""
import io
import time
import numpy as np
import cv2
from PIL import Image
from collections import Counter


def jpeg_compress(img: np.ndarray, quality: int = 75) -> dict:
    """Compress image with JPEG at given quality, return compressed image + stats."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    compressed_size = buf.tell()
    buf.seek(0)
    compressed_pil = Image.open(buf)
    compressed_np = cv2.cvtColor(np.array(compressed_pil), cv2.COLOR_RGB2BGR)

    # Original size estimate
    original_size = img.shape[0] * img.shape[1] * img.shape[2] if img.ndim == 3 else img.shape[0] * img.shape[1]
    compression_ratio = original_size / max(compressed_size, 1)
    size_reduction = (1 - compressed_size / original_size) * 100

    # PSNR
    orig_f = img.astype(np.float64)
    comp_f = compressed_np.astype(np.float64)
    mse = np.mean((orig_f - comp_f) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 100.0

    return {
        'image': compressed_np,
        'compressed_size_kb': round(compressed_size / 1024, 2),
        'original_size_kb': round(original_size / 1024, 2),
        'compression_ratio': round(compression_ratio, 2),
        'size_reduction_pct': round(size_reduction, 1),
        'psnr': round(psnr, 2),
    }


def psnr_vs_quality(img: np.ndarray) -> dict:
    """Compute PSNR for quality levels 10, 20, ..., 100."""
    qualities = list(range(10, 101, 10))
    psnr_values = []
    sizes = []
    for q in qualities:
        result = jpeg_compress(img, q)
        psnr_values.append(result['psnr'])
        sizes.append(result['compressed_size_kb'])
    return {'qualities': qualities, 'psnr': psnr_values, 'sizes': sizes}


def visualize_dct_blocks(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Show DCT coefficient visualization on 8×8 blocks."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    # Pad to multiple of block_size
    ph = ((h + block_size - 1) // block_size) * block_size
    pw = ((w + block_size - 1) // block_size) * block_size
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h, :w] = gray.astype(np.float32)

    dct_img = np.zeros_like(padded)
    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i + block_size, j:j + block_size]
            dct_block = cv2.dct(block - 128)
            # Log scale for visualization
            dct_img[i:i + block_size, j:j + block_size] = np.log1p(np.abs(dct_block))

    dct_norm = cv2.normalize(dct_img[:h, :w], None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    # Draw block grid
    result = cv2.cvtColor(dct_norm, cv2.COLOR_GRAY2BGR)
    for i in range(0, h, block_size):
        cv2.line(result, (0, i), (w, i), (0, 255, 0), 1)
    for j in range(0, w, block_size):
        cv2.line(result, (j, 0), (j, h), (0, 255, 0), 1)
    return result


def run_length_encoding(img: np.ndarray) -> dict:
    """RLE on binary image — return encoded pairs and compression stats."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
    flat = binary.flatten()
    rle = []
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1]:
            count += 1
        else:
            rle.append((int(flat[i - 1]), count))
            count = 1
    rle.append((int(flat[-1]), count))
    original_bits = len(flat)
    encoded_bits = len(rle) * 16  # approx: value(1) + count(15)
    ratio = original_bits / max(encoded_bits, 1)
    return {
        'pairs': rle[:50],  # first 50 pairs for display
        'total_pairs': len(rle),
        'original_bits': original_bits,
        'encoded_bits': encoded_bits,
        'compression_ratio': round(ratio, 3),
    }


def huffman_coding_demo(img: np.ndarray) -> dict:
    """Build Huffman frequency table and code lengths for grayscale image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    flat = gray.flatten().tolist()
    freq = Counter(flat)
    total = len(flat)

    # Build Huffman tree (simplified — compute code lengths)
    import heapq
    heap = [[f, [sym, '']] for sym, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    codes = {}
    if heap:
        for sym, code in heap[0][1:]:
            codes[sym] = code if code else '0'

    # Top 20 symbols by frequency
    top = sorted(freq.items(), key=lambda x: -x[1])[:20]
    table = []
    for sym, cnt in top:
        prob = cnt / total
        code = codes.get(sym, '')
        table.append({
            'symbol': sym,
            'frequency': cnt,
            'probability': round(prob, 5),
            'code': code,
            'code_length': len(code),
        })

    avg_len = sum(len(codes.get(s, '')) * f / total for s, f in freq.items())
    entropy = -sum((f / total) * np.log2(f / total + 1e-12) for f in freq.values())

    return {
        'table': table,
        'avg_code_length': round(avg_len, 3),
        'entropy': round(float(entropy), 3),
        'unique_symbols': len(freq),
    }


def compare_formats(img: np.ndarray) -> dict:
    """Compare JPEG, PNG, WebP file sizes."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    sizes = {}
    for fmt in ['JPEG', 'PNG', 'WEBP']:
        buf = io.BytesIO()
        try:
            pil_img.save(buf, format=fmt, quality=85)
            sizes[fmt.lower()] = round(buf.tell() / 1024, 2)
        except Exception:
            sizes[fmt.lower()] = 0
    return sizes
