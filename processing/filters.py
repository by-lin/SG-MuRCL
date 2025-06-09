import cv2
import math
import numpy as np
import skimage.morphology as sk_morphology
import logging

from PIL import Image
from skimage import img_as_ubyte
from skimage.color import rgb2hsv
from .utils import downsample_image

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG for more verbosity
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


logger = logging.getLogger(__name__)

# Constants for pen color thresholds
RED_PEN_THRESHOLDS = [
    (150, 80, 90), (110, 20, 30), (185, 65, 105), (195, 85, 125),
    (220, 115, 145), (125, 40, 70), (200, 120, 150), (100, 50, 65), (85, 25, 45)
]

GREEN_PEN_THRESHOLDS = [
    (150, 160, 140), (70, 110, 110), (45, 115, 100), (30, 75, 60),
    (195, 220, 210), (225, 230, 225), (170, 210, 200), (20, 30, 20),
    (50, 60, 40), (30, 50, 35), (65, 70, 60), (100, 110, 105),
    (165, 180, 180), (140, 140, 150), (185, 195, 195)
]

BLUE_PEN_THRESHOLDS = [
    (60, 120, 190), (120, 170, 200), (175, 210, 230), (145, 180, 210),
    (37, 95, 160), (30, 65, 130), (130, 155, 180), (40, 35, 85),
    (30, 20, 65), (90, 90, 140), (60, 60, 120), (110, 110, 175)
]


def otsu(slide, mask_downsample, mask_filepath: str = None):
    """Apply Otsu's thresholding method to generate a binary mask."""
    img, _ = downsample_image(slide, mask_downsample)
    img = cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask_filepath:
        cv2.imwrite(mask_filepath, thresh_otsu)
    return Image.fromarray(thresh_otsu), np.array([255, 255, 255])

def adaptive(slide, mask_downsample, mask_filepath: str = None):
    """Apply adaptive Gaussian thresholding to generate a binary mask."""
    img, _ = downsample_image(slide, mask_downsample)
    img = cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    thresh_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    if mask_filepath:
        cv2.imwrite(mask_filepath, thresh_adapt)
    return Image.fromarray(thresh_adapt), np.array([255, 255, 255])

def saturation(img, t=15):
    """Return True if average saturation is above threshold t."""
    hsv = rgb2hsv(img)
    sat_img = img_as_ubyte(hsv[:, :, 1])
    avg_sat = np.mean(sat_img)
    return avg_sat >= t

def mask_percent(np_img):
    """Calculate the percentage of background in the mask."""
    if np_img.ndim == 3 and np_img.shape[2] == 3:
        np_sum = np.sum(np_img, axis=2)
    else:
        np_sum = np_img
    return 100 - np.count_nonzero(np_sum) / np_sum.size * 100

def tissue_percent(np_img):
    """Calculate the percentage of tissue in the mask."""
    return 100 - mask_percent(np_img)

def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_pct = mask_percent(gr_ch_mask)
    if avoid_overmask and mask_pct >= overmask_thresh and green_thresh < 255:
        new_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        logger.info(f"Green channel overmask: {mask_pct:.2f}% >= {overmask_thresh}%, trying new_thresh={new_thresh}")
        return filter_green_channel(np_img, new_thresh, avoid_overmask, overmask_thresh, output_type)
    return _format_output(gr_ch_mask, output_type)

def filter_grays(rgb, tolerance=15, output_type="bool"):
    rgb = rgb.astype(int)
    result = ~((abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance) &
               (abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance) &
               (abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance))
    return _format_output(result, output_type)

def filter_red(rgb, r_thresh, g_thresh, b_thresh, output_type="bool"):
    result = ~((rgb[:, :, 0] > r_thresh) & (rgb[:, :, 1] < g_thresh) & (rgb[:, :, 2] < b_thresh))
    return _format_output(result, output_type)

def filter_green(rgb, r_thresh, g_thresh, b_thresh, output_type="bool"):
    result = ~((rgb[:, :, 0] < r_thresh) & (rgb[:, :, 1] > g_thresh) & (rgb[:, :, 2] > b_thresh))
    return _format_output(result, output_type)

def filter_blue(rgb, r_thresh, g_thresh, b_thresh, output_type="bool"):
    result = ~((rgb[:, :, 0] < r_thresh) & (rgb[:, :, 1] < g_thresh) & (rgb[:, :, 2] > b_thresh))
    return _format_output(result, output_type)

def _pen_filter(rgb, thresholds, filter_fn, output_type):
    result = np.ones(rgb.shape[:2], dtype=bool)
    for r, g, b in thresholds:
        result &= filter_fn(rgb, r, g, b, output_type="bool")
    return _format_output(result, output_type)

def filter_red_pen(rgb, output_type="bool"):
    return _pen_filter(rgb, RED_PEN_THRESHOLDS, filter_red, output_type)

def filter_green_pen(rgb, output_type="bool"):
    return _pen_filter(rgb, GREEN_PEN_THRESHOLDS, filter_green, output_type)

def filter_blue_pen(rgb, output_type="bool"):
    return _pen_filter(rgb, BLUE_PEN_THRESHOLDS, filter_blue, output_type)

def filter_remove_small_objects(np_img, min_size=500, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
    rem_sm = sk_morphology.remove_small_objects(np_img.astype(bool), min_size=min_size)
    mask_pct = mask_percent(rem_sm)
    if avoid_overmask and mask_pct >= overmask_thresh and min_size >= 1:
        new_min_size = min_size / 2
        logger.info(f"Small object removal overmask: {mask_pct:.2f}% >= {overmask_thresh}%, trying size={new_min_size}")
        return filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    return _format_output(rem_sm, output_type)

def _format_output(mask, output_type):
    if output_type == "bool":
        return mask
    elif output_type == "float":
        return mask.astype(float)
    else:
        return mask.astype("uint8") * 255

def mask_rgb(rgb, mask):
    """Apply a binary mask to an RGB image."""
    return rgb * np.dstack([mask, mask, mask])

def RGB_filter(slide, mask_downsample, mask_filepath: str = None):
    """Generate tissue mask using RGB-based filtering and pen artifact removal."""
    img, _ = downsample_image(slide, mask_downsample)
    rgb = img

    mask = (filter_grays(rgb) &
            filter_green_channel(rgb) &
            filter_red_pen(rgb) &
            filter_green_pen(rgb) &
            filter_blue_pen(rgb))

    mask_cleaned = filter_remove_small_objects(mask, min_size=500, output_type="bool")
    rgb_mask = mask_rgb(rgb, mask_cleaned)

    if mask_filepath:
        cv2.imwrite(mask_filepath, rgb_mask)

    binary_mask = np.uint8(255 * ~mask_cleaned)
    return Image.fromarray(binary_mask), np.array([255, 255, 255])
