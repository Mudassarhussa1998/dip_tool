"""
core/views.py  — Part 1: helpers + page views
"""
import base64
import io
import json
import time
import traceback

import cv2
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

from .utils import filters, frequency, morphology, segmentation, compression
from .utils import restoration as restoration_utils


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image(b64_string: str) -> np.ndarray:
    """Decode base64 image string to BGR numpy array."""
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_string)
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _encode_image(img: np.ndarray) -> str:
    """Encode BGR numpy array to base64 PNG string."""
    if img is None:
        return ''
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    success, buf = cv2.imencode('.png', img)
    if not success:
        return ''
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('utf-8')


def _json_error(msg: str, status: int = 400) -> JsonResponse:
    return JsonResponse({'error': msg}, status=status)


def _parse_body(request) -> dict:
    try:
        return json.loads(request.body)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Page views
# ─────────────────────────────────────────────────────────────────────────────

def home(request):
    modules = [
        {'num': '01', 'title': 'Introduction & Spectrum', 'url': '/intro/',
         'desc': 'Electromagnetic spectrum, image sensing pipeline, RGB channel separation.'},
        {'num': '02', 'title': 'Sampling & Quantization', 'url': '/sampling/',
         'desc': 'Downsampling, upsampling with interpolation methods, bit-depth reduction.'},
        {'num': '03', 'title': 'Enhancement & Point Ops', 'url': '/enhancement/',
         'desc': 'Brightness, contrast, gamma, log, negative, thresholding, geometric ops.'},
        {'num': '04', 'title': 'Bit Plane & Watermarking', 'url': '/bitplane/',
         'desc': 'Bit plane slicing, LSB watermarking, visible watermark overlay.'},
        {'num': '05', 'title': 'Histogram Processing', 'url': '/histogram/',
         'desc': 'Histogram equalization, matching, CLAHE, CDF visualization.'},
        {'num': '06', 'title': 'Spatial Filtering', 'url': '/spatial/',
         'desc': 'Correlation vs convolution, smoothing filter comparison, boundary handling.'},
        {'num': '07', 'title': 'Sharpening & Edge Detection', 'url': '/sharpening/',
         'desc': 'Laplacian, unsharp mask, Sobel, Prewitt, Roberts, LoG, Canny — all compared.'},
        {'num': '08/09', 'title': 'Frequency Domain', 'url': '/frequency/',
         'desc': 'FFT/DFT spectra, ideal/Butterworth/Gaussian LP/HP/BP/notch filters.'},
        {'num': '10', 'title': 'Image Restoration', 'url': '/restoration/',
         'desc': 'Noise models, Wiener, NLM, DnCNN simulation — full comparison table.'},
        {'num': '11', 'title': 'Segmentation: Edges', 'url': '/segmentation-edges/',
         'desc': 'Point/line detection, Hough line & circle transforms.'},
        {'num': '12', 'title': 'Segmentation: Region', 'url': '/segmentation-region/',
         'desc': 'Global/Otsu/adaptive thresholding, region growing, watershed.'},
        {'num': '13', 'title': 'Morphological Operations', 'url': '/morphology/',
         'desc': 'Erosion, dilation, opening, closing, gradient, top-hat, skeletonization.'},
        {'num': '14', 'title': 'Image Compression', 'url': '/compression/',
         'desc': 'JPEG quality, DCT blocks, RLE, Huffman coding, format comparison.'},
        {'num': '15', 'title': 'Colour Processing & ML', 'url': '/colour/',
         'desc': 'Colour spaces, pseudo-colouring, segmentation, ResNet-18 recognition + Grad-CAM.'},
    ]
    return render(request, 'core/home.html', {'modules': modules})

def intro(request):
    return render(request, 'core/intro.html', {
        'pipeline_steps': ['Scene', 'Lens', 'Sensor', 'ADC', 'Digital Image']
    })

def sampling(request):
    return render(request, 'core/sampling.html')

def enhancement(request):
    return render(request, 'core/enhancement.html')

def bitplane(request):
    return render(request, 'core/bitplane.html', {
        'bit_planes': [7, 6, 5, 4, 3, 2, 1, 0]
    })

def histogram_page(request):
    return render(request, 'core/histogram.html')

def spatial(request):
    return render(request, 'core/spatial.html', {
        'kernel_vals': [0, 0, 0, 0, 1, 0, 0, 0, 0]
    })

def sharpening(request):
    return render(request, 'core/sharpening.html')

def frequency_page(request):
    return render(request, 'core/frequency.html')

def restoration(request):
    return render(request, 'core/restoration.html', {
        'restoration_methods': ['mean', 'gaussian', 'median', 'bilateral', 'wiener', 'nlm', 'dncnn']
    })

def segmentation_edges(request):
    return render(request, 'core/segmentation_edges.html')

def segmentation_region(request):
    return render(request, 'core/segmentation_region.html')

def morphology_page(request):
    return render(request, 'core/morphology.html')

def compression_page(request):
    return render(request, 'core/compression.html')

def colour(request):
    return render(request, 'core/colour.html')

# ─────────────────────────────────────────────────────────────────────────────
# API: Module 1 — Intro
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_intro(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'properties')
        response = {}
        if operation == 'properties':
            props = filters.get_image_properties(img)
            response = {'properties': props, 'result_image': _encode_image(img)}
        elif operation == 'channels':
            r, g, b = filters.split_channels(img)
            response = {
                'r_channel': _encode_image(r),
                'g_channel': _encode_image(g),
                'b_channel': _encode_image(b),
                'result_image': _encode_image(img),
            }
        elif operation == 'grayscale':
            gray = filters.to_grayscale(img)
            response = {'result_image': _encode_image(gray)}
        response['processing_time_ms'] = round((time.time() - t0) * 1000, 2)
        return JsonResponse(response)
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 2 — Sampling
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_sampling(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        params = data.get('params', {})
        factor = int(params.get('factor', 4))
        method = params.get('method', 'bilinear')
        bits = int(params.get('bits', 8))
        downsampled = filters.downsample(img, factor)
        upsampled = filters.upsample(downsampled, img.shape[:2], method)
        quantized = filters.quantize(img, bits)
        metrics_up = filters.compute_metrics(img, upsampled)
        metrics_q = filters.compute_metrics(img, quantized)
        return JsonResponse({
            'result_image': _encode_image(upsampled),
            'downsampled': _encode_image(downsampled),
            'upsampled': _encode_image(upsampled),
            'quantized': _encode_image(quantized),
            'metrics_upsampled': metrics_up,
            'metrics_quantized': metrics_q,
            'processing_time_ms': round((time.time() - t0) * 1000, 2),
        })
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 3 — Enhancement
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_enhancement(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'brightness')
        params = data.get('params', {})
        op_map = {
            'brightness': lambda: filters.brightness(img, float(params.get('value', 50))),
            'contrast': lambda: filters.contrast(img, float(params.get('value', 1.5))),
            'gamma': lambda: filters.gamma_correction(img, float(params.get('value', 1.0))),
            'log': lambda: filters.log_transform(img),
            'negative': lambda: filters.negative_image(img),
            'threshold': lambda: filters.threshold_image(img, int(params.get('value', 127))),
            'piecewise': lambda: filters.piecewise_linear_stretch(
                img, int(params.get('r1', 64)), int(params.get('s1', 0)),
                int(params.get('r2', 192)), int(params.get('s2', 255))),
            'mean_filter': lambda: filters.mean_filter(img, int(params.get('ksize', 5))),
            'gaussian_filter': lambda: filters.gaussian_filter(
                img, int(params.get('ksize', 5)), float(params.get('sigma', 1.0))),
            'median_filter': lambda: filters.median_filter(img, int(params.get('ksize', 5))),
            'rotate': lambda: filters.rotate_image(img, float(params.get('angle', 45))),
            'scale': lambda: filters.scale_image(
                img, float(params.get('fx', 1.5)), float(params.get('fy', 1.5))),
            'flip': lambda: filters.flip_image(img, params.get('direction', 'horizontal')),
            'translate': lambda: filters.translate_image(
                img, int(params.get('dx', 20)), int(params.get('dy', 20))),
            'shear': lambda: filters.shear_image(
                img, float(params.get('shx', 0.2)), float(params.get('shy', 0.0))),
        }
        fn = op_map.get(operation)
        if fn is None:
            return _json_error(f'Unknown operation: {operation}')
        result = fn()
        metrics = filters.compute_metrics(img, result)
        return JsonResponse({
            'result_image': _encode_image(result),
            'metrics': metrics,
            'processing_time_ms': round((time.time() - t0) * 1000, 2),
        })
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 4 — Bit Plane
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_bitplane(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'planes')
        params = data.get('params', {})
        if operation == 'planes':
            planes = filters.get_bit_planes(img)
            return JsonResponse({
                'planes': [_encode_image(p) for p in planes],
                'result_image': _encode_image(planes[7]),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'reconstruct':
            selected = params.get('bits', [7, 6, 5])
            result = filters.reconstruct_from_planes(img, selected)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            metrics = filters.compute_metrics(img, result_bgr)
            return JsonResponse({
                'result_image': _encode_image(result_bgr),
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'watermark_lsb':
            text = params.get('text', 'DIP Tool')
            watermarked = filters.embed_text_watermark_lsb(img, text)
            extracted = filters.extract_text_watermark_lsb(watermarked, len(text))
            return JsonResponse({
                'result_image': _encode_image(watermarked),
                'extracted_text': extracted,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'watermark_visible':
            text = params.get('text', 'WATERMARK')
            alpha = float(params.get('alpha', 0.4))
            result = filters.add_visible_watermark(img, text, alpha)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))

# ─────────────────────────────────────────────────────────────────────────────
# API: Module 5 — Histogram
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_histogram(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'histogram')
        params = data.get('params', {})
        hist_data = filters.compute_histogram(img)
        if operation == 'histogram':
            return JsonResponse({
                'result_image': _encode_image(img),
                'histogram': hist_data,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'equalize':
            result = filters.histogram_equalization(img)
            new_hist = filters.compute_histogram(result)
            metrics = filters.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'original_histogram': hist_data,
                'result_histogram': new_hist,
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'match':
            ref_b64 = data.get('reference_image', '')
            if not ref_b64:
                return _json_error('No reference image')
            ref = _decode_image(ref_b64)
            result = filters.histogram_matching(img, ref)
            new_hist = filters.compute_histogram(result)
            ref_hist = filters.compute_histogram(ref)
            metrics = filters.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'original_histogram': hist_data,
                'result_histogram': new_hist,
                'reference_histogram': ref_hist,
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'local_eq':
            tile = int(params.get('tile_size', 8))
            result = filters.local_histogram_eq(img, tile)
            new_hist = filters.compute_histogram(result)
            metrics = filters.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'original_histogram': hist_data,
                'result_histogram': new_hist,
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'clahe':
            clip = float(params.get('clip_limit', 2.0))
            tile = int(params.get('tile_size', 8))
            result = filters.clahe_equalization(img, clip, tile)
            new_hist = filters.compute_histogram(result)
            metrics = filters.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'original_histogram': hist_data,
                'result_histogram': new_hist,
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 6 — Spatial Filtering
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_spatial(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'compare')
        params = data.get('params', {})
        if operation == 'custom_kernel':
            kernel_vals = params.get('kernel', [0, 0, 0, 0, 1, 0, 0, 0, 0])
            kernel = np.array(kernel_vals, dtype=np.float32).reshape(3, 3)
            mode = params.get('mode', 'convolution')
            border = params.get('border', 'reflect')
            result = filters.apply_custom_kernel(img, kernel, mode, border)
            corr = filters.apply_custom_kernel(img, kernel, 'correlation', border)
            conv = filters.apply_custom_kernel(img, kernel, 'convolution', border)
            diff = cv2.absdiff(corr, conv)
            metrics = filters.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'correlation': _encode_image(corr),
                'convolution': _encode_image(conv),
                'difference': _encode_image(diff),
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'compare':
            ksize = int(params.get('ksize', 5))
            sigma = float(params.get('sigma', 1.0))
            border = params.get('border', 'reflect')
            results = filters.compare_smoothing_filters(img, ksize, sigma, border)
            response = {'processing_time_ms': round((time.time() - t0) * 1000, 2)}
            psnr_chart = {}
            time_chart = {}
            for name, r in results.items():
                response[name] = _encode_image(r['image'])
                response[f'{name}_metrics'] = r['metrics']
                response[f'{name}_time'] = r['time']
                psnr_chart[name] = r['metrics']['psnr']
                time_chart[name] = r['time']
            response['psnr_chart'] = psnr_chart
            response['time_chart'] = time_chart
            response['result_image'] = response.get('gaussian', '')
            return JsonResponse(response)
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 7 — Sharpening & Edge Detection
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_sharpening(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'all_edges')
        params = data.get('params', {})
        if operation == 'laplacian_sharp':
            alpha = float(params.get('alpha', 1.0))
            result = filters.laplacian_sharpening(img, alpha)
            metrics = filters.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'unsharp':
            radius = int(params.get('radius', 5))
            amount = float(params.get('amount', 1.5))
            result = filters.unsharp_mask(img, radius, amount)
            metrics = filters.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'high_boost':
            boost = float(params.get('boost', 1.5))
            result = filters.high_boost_filter(img, boost)
            metrics = filters.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'all_edges':
            canny_low = int(params.get('canny_low', 50))
            canny_high = int(params.get('canny_high', 150))
            results = filters.all_edge_detectors(img, canny_low, canny_high)
            response = {
                'sobel_x': _encode_image(results['sobel_x']),
                'sobel_y': _encode_image(results['sobel_y']),
                'sobel': _encode_image(results['sobel']),
                'prewitt_x': _encode_image(results['prewitt_x']),
                'prewitt_y': _encode_image(results['prewitt_y']),
                'prewitt': _encode_image(results['prewitt']),
                'roberts': _encode_image(results['roberts']),
                'laplacian': _encode_image(results['laplacian']),
                'log': _encode_image(results['log']),
                'canny': _encode_image(results['canny']),
                'result_image': _encode_image(results['canny']),
                'times': results['times'],
                'edge_counts': results['edge_counts'],
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            }
            return JsonResponse(response)
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))

# ─────────────────────────────────────────────────────────────────────────────
# API: Module 8/9 — Frequency Domain
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_frequency(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'fft')
        params = data.get('params', {})
        if operation == 'fft':
            fft_data = frequency.compute_fft(img)
            return JsonResponse({
                'magnitude': _encode_image(fft_data['magnitude']),
                'phase': _encode_image(fft_data['phase']),
                'power': _encode_image(fft_data['power']),
                'cross_section': fft_data['cross_section'],
                'result_image': _encode_image(fft_data['magnitude']),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'filter':
            filter_type = params.get('filter_type', 'gauss_lp')
            cutoff_low = float(params.get('cutoff_low', 30))
            cutoff_high = float(params.get('cutoff_high', 70))
            order = int(params.get('order', 2))
            result = frequency.apply_frequency_filter(
                img, filter_type, cutoff_low, cutoff_high, order)
            return JsonResponse({
                'filtered_spectrum': _encode_image(result['filtered_spectrum']),
                'reconstructed': _encode_image(result['reconstructed']),
                'filter_mask': _encode_image(result['filter_mask']),
                'cross_section': result['cross_section'],
                'result_image': _encode_image(result['reconstructed']),
                'processing_time_ms': result['processing_time_ms'],
            })
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 10 — Restoration
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_restoration(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'compare')
        params = data.get('params', {})

        # Add noise
        noise_type = params.get('noise_type', 'gaussian')
        noise_map = {
            'gaussian': lambda: restoration_utils.add_gaussian_noise(
                img, float(params.get('mean', 0)), float(params.get('sigma', 25))),
            'salt_pepper': lambda: restoration_utils.add_salt_pepper_noise(
                img, float(params.get('density', 0.05))),
            'speckle': lambda: restoration_utils.add_speckle_noise(img),
            'poisson': lambda: restoration_utils.add_poisson_noise(img),
            'periodic': lambda: restoration_utils.add_periodic_noise(
                img, float(params.get('freq', 0.1)),
                params.get('direction', 'horizontal')),
        }
        noise_fn = noise_map.get(noise_type, noise_map['gaussian'])
        noisy = noise_fn()

        if operation == 'add_noise':
            metrics = restoration_utils.compute_metrics(img, noisy)
            return JsonResponse({
                'result_image': _encode_image(noisy),
                'noisy_image': _encode_image(noisy),
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'compare':
            results = restoration_utils.compare_all_restoration(noisy, img)
            response = {
                'noisy_image': _encode_image(noisy),
                'result_image': _encode_image(results['dncnn']['image']),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            }
            psnr_chart = {}
            time_chart = {}
            best_psnr = -1
            best_method = ''
            for name, r in results.items():
                response[name] = _encode_image(r['image'])
                response[f'{name}_metrics'] = r['metrics']
                response[f'{name}_time'] = r['time']
                psnr_chart[name] = r['metrics']['psnr']
                time_chart[name] = r['time']
                if r['metrics']['psnr'] > best_psnr:
                    best_psnr = r['metrics']['psnr']
                    best_method = name
            response['psnr_chart'] = psnr_chart
            response['time_chart'] = time_chart
            response['best_method'] = best_method
            return JsonResponse(response)
        elif operation == 'adaptive_mean':
            ksize = int(params.get('ksize', 7))
            result = restoration_utils.adaptive_mean_filter(noisy, ksize)
            metrics = restoration_utils.compute_metrics(img, result)
            return JsonResponse({
                'result_image': _encode_image(result),
                'noisy_image': _encode_image(noisy),
                'metrics': metrics,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 11 — Segmentation Edges
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_segmentation_edges(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'hough_lines')
        params = data.get('params', {})
        if operation == 'point_detection':
            result = segmentation.point_detection(img)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'line_detection':
            direction = params.get('direction', 'horizontal')
            result = segmentation.line_detection(img, direction)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'hough_lines':
            threshold = int(params.get('threshold', 100))
            result = segmentation.hough_lines(img, threshold)
            return JsonResponse({
                'result_image': _encode_image(result['overlay']),
                'edges': _encode_image(result['edges']),
                'accumulator': _encode_image(result['accumulator']),
                'line_count': result['line_count'],
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'hough_circles':
            min_r = int(params.get('min_radius', 10))
            max_r = int(params.get('max_radius', 100))
            result = segmentation.hough_circles(img, min_r, max_r)
            return JsonResponse({
                'result_image': _encode_image(result['overlay']),
                'circle_count': result['circle_count'],
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))

# ─────────────────────────────────────────────────────────────────────────────
# API: Module 12 — Segmentation Region
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_segmentation_region(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'otsu')
        params = data.get('params', {})
        if operation == 'global':
            thresh = int(params.get('threshold', 127))
            result = segmentation.global_threshold(img, thresh)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'otsu':
            result = segmentation.otsu_threshold(img)
            return JsonResponse({
                'result_image': _encode_image(result['image']),
                'threshold_value': result['threshold'],
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'adaptive':
            block = int(params.get('block_size', 11))
            C = int(params.get('C', 2))
            result = segmentation.adaptive_threshold(img, block, C)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'multi_level':
            levels = int(params.get('levels', 3))
            result = segmentation.multi_level_threshold(img, levels)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'region_growing':
            sx = int(params.get('seed_x', img.shape[1] // 2))
            sy = int(params.get('seed_y', img.shape[0] // 2))
            tol = int(params.get('tolerance', 20))
            result = segmentation.region_growing(img, sx, sy, tol)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'watershed':
            result = segmentation.watershed_segmentation(img)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'edge_based':
            result = segmentation.edge_based_segmentation(img)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 13 — Morphology
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_morphology(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        params = data.get('params', {})
        threshold = int(params.get('threshold', 127))
        se_shape = params.get('se_shape', 'rect')
        se_size = int(params.get('se_size', 3))
        iterations = int(params.get('iterations', 1))
        results = morphology.apply_all_morphology(
            img, threshold, se_shape, se_size, iterations)
        response = {
            'processing_time_ms': round((time.time() - t0) * 1000, 2),
            'result_image': _encode_image(results['erosion']),
        }
        for key, arr in results.items():
            response[key] = _encode_image(arr)
        return JsonResponse(response)
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 14 — Compression
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_compression(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'jpeg')
        params = data.get('params', {})
        if operation == 'jpeg':
            quality = int(params.get('quality', 75))
            result = compression.jpeg_compress(img, quality)
            return JsonResponse({
                'result_image': _encode_image(result['image']),
                'compressed_size_kb': result['compressed_size_kb'],
                'original_size_kb': result['original_size_kb'],
                'compression_ratio': result['compression_ratio'],
                'size_reduction_pct': result['size_reduction_pct'],
                'psnr': result['psnr'],
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'psnr_curve':
            curve = compression.psnr_vs_quality(img)
            return JsonResponse({
                'result_image': _encode_image(img),
                'curve': curve,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'dct':
            result = compression.visualize_dct_blocks(img)
            return JsonResponse({
                'result_image': _encode_image(result),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'rle':
            result = compression.run_length_encoding(img)
            return JsonResponse({
                'result_image': _encode_image(img),
                'rle': result,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'huffman':
            result = compression.huffman_coding_demo(img)
            return JsonResponse({
                'result_image': _encode_image(img),
                'huffman': result,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'format_compare':
            sizes = compression.compare_formats(img)
            return JsonResponse({
                'result_image': _encode_image(img),
                'sizes': sizes,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# API: Module 15 — Colour
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_colour(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'colorspaces')
        params = data.get('params', {})

        if operation == 'colorspaces':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # Channel splits
            b, g, r = cv2.split(img)
            h_ch, s_ch, v_ch = cv2.split(hsv)
            l_ch, a_ch, b_ch = cv2.split(lab)
            return JsonResponse({
                'result_image': _encode_image(img),
                'gray': _encode_image(gray),
                'hsv': _encode_image(hsv),
                'lab': _encode_image(lab),
                'ycbcr': _encode_image(ycbcr),
                'r_channel': _encode_image(r),
                'g_channel': _encode_image(g),
                'b_channel': _encode_image(b),
                'h_channel': _encode_image(h_ch),
                's_channel': _encode_image(s_ch),
                'v_channel': _encode_image(v_ch),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'pseudo_colour':
            colormap = int(params.get('colormap', cv2.COLORMAP_JET))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            colored = cv2.applyColorMap(gray, colormap)
            return JsonResponse({
                'result_image': _encode_image(colored),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'colour_histogram':
            b, g, r = cv2.split(img)
            def ch_hist(ch):
                h = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten().tolist()
                return h
            return JsonResponse({
                'result_image': _encode_image(img),
                'r_hist': ch_hist(r),
                'g_hist': ch_hist(g),
                'b_hist': ch_hist(b),
                'labels': list(range(256)),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'colour_segment':
            h_low = int(params.get('h_low', 0))
            h_high = int(params.get('h_high', 30))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,
                               np.array([h_low, 50, 50]),
                               np.array([h_high, 255, 255]))
            result = cv2.bitwise_and(img, img, mask=mask)
            return JsonResponse({
                'result_image': _encode_image(result),
                'mask': _encode_image(mask),
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            })
        elif operation == 'recognize':
            try:
                import torch
                import torchvision.transforms as T
                from torchvision.models import resnet18, ResNet18_Weights
                import urllib.request

                weights = ResNet18_Weights.DEFAULT
                model = resnet18(weights=weights)
                model.eval()

                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                preprocess = weights.transforms()
                input_tensor = preprocess(pil_img).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top5_prob, top5_idx = torch.topk(probs, 5)

                categories = weights.meta['categories']
                predictions = []
                for prob, idx in zip(top5_prob.tolist(), top5_idx.tolist()):
                    predictions.append({
                        'label': categories[idx],
                        'confidence': round(prob * 100, 2),
                    })

                # Simple Grad-CAM approximation using last conv layer activations
                activations = {}
                def hook_fn(module, input, output):
                    activations['feat'] = output.detach()
                handle = model.layer4.register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(input_tensor)
                handle.remove()
                feat = activations['feat'][0].mean(0).numpy()
                feat = cv2.resize(feat, (img.shape[1], img.shape[0]))
                feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
                heatmap = cv2.applyColorMap(
                    (feat * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

                return JsonResponse({
                    'result_image': _encode_image(overlay),
                    'predictions': predictions,
                    'gradcam': _encode_image(overlay),
                    'processing_time_ms': round((time.time() - t0) * 1000, 2),
                })
            except Exception as e:
                return _json_error(f'Recognition error: {str(e)}')
        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Watermark Removal — Page view
# ─────────────────────────────────────────────────────────────────────────────

def watermark_removal(request):
    removal_methods = [
        {'key': 'telea',           'label': 'Inpainting — Telea'},
        {'key': 'ns',              'label': 'Inpainting — Navier-Stokes'},
        {'key': 'median',          'label': 'Median Fill'},
        {'key': 'gaussian',        'label': 'Gaussian Fill'},
        {'key': 'frequency_notch', 'label': 'Frequency Notch'},
    ]
    return render(request, 'core/watermark_removal.html', {
        'removal_methods': removal_methods,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Watermark Removal — API endpoint
# ─────────────────────────────────────────────────────────────────────────────

from .utils import watermark_removal as wm_utils


@csrf_exempt
def api_watermark_removal(request):
    if request.method != 'POST':
        return _json_error('POST required')
    t0 = time.time()
    data = _parse_body(request)
    img_b64 = data.get('image', '')
    if not img_b64:
        return _json_error('No image provided')
    try:
        img = _decode_image(img_b64)
        operation = data.get('operation', 'remove')
        params = data.get('params', {})

        detection_method = params.get('detection_method', 'bright')
        removal_method   = params.get('removal_method', 'telea')
        threshold        = int(params.get('threshold', 200))
        dilate_iter      = int(params.get('dilate_iter', 3))
        inpaint_radius   = int(params.get('inpaint_radius', 5))
        notch_radius     = int(params.get('notch_radius', 10))

        if operation == 'detect':
            # Only detect and return the mask (no removal)
            result = wm_utils.remove_watermark(
                img,
                detection_method=detection_method,
                removal_method='telea',   # removal not used, but pipeline needs it
                threshold=threshold,
                dilate_iter=dilate_iter,
                inpaint_radius=inpaint_radius,
                notch_radius=notch_radius,
            )
            return JsonResponse({
                'mask_image': _encode_image(result['mask']),
                'mask_coverage_pct': result['mask_coverage_pct'],
                'processing_time_ms': result['processing_time_ms'],
            })

        elif operation == 'remove':
            result = wm_utils.remove_watermark(
                img,
                detection_method=detection_method,
                removal_method=removal_method,
                threshold=threshold,
                dilate_iter=dilate_iter,
                inpaint_radius=inpaint_radius,
                notch_radius=notch_radius,
            )
            return JsonResponse({
                'result_image': _encode_image(result['result']),
                'mask_image': _encode_image(result['mask']),
                'mask_coverage_pct': result['mask_coverage_pct'],
                'processing_time_ms': result['processing_time_ms'],
            })

        elif operation == 'compare':
            results, mask = wm_utils.compare_all_methods(
                img,
                detection_method=detection_method,
                threshold=threshold,
                dilate_iter=dilate_iter,
            )
            coverage = wm_utils._mask_coverage_pct(mask)
            response = {
                'mask_image': _encode_image(mask),
                'mask_coverage_pct': coverage,
                'processing_time_ms': round((time.time() - t0) * 1000, 2),
            }
            for name, r in results.items():
                response[name] = _encode_image(r['image'])
                response[f'{name}_time'] = r['time']
            return JsonResponse(response)

        return _json_error('Unknown operation')
    except Exception as e:
        return _json_error(str(e))
