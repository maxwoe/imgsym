import numpy as np
import cv2
from skimage.exposure import is_low_contrast


def fspecial_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def rescale(img, target_size):
    max_size = np.max(img.shape[:2])
    rf = target_size/max_size
    if rf < 1:
        width = int(img.shape[1] * rf)
        height = int(img.shape[0] * rf)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        rf = 1
    return img, rf


def get_colorspace(img, colorspace):
    """BGR image -> the requested colorspace channel(s). Single channel for
    gray/luminance/value/saturation/lab_a/lab_b; multi-channel for lab (L+a+b)
    and rgb (BGR as-is)."""
    if colorspace == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if colorspace == "luminance":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[..., 0]
    if colorspace == "value":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    if colorspace == "saturation":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    if colorspace == "lab_a":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 1]
    if colorspace == "lab_b":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 2]
    if colorspace == "lab":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if colorspace == "rgb":
        return img
    raise ValueError(f"unknown colorspace {colorspace!r}")


def calc_mag_ang(image, method="sobel", angle_in_degrees=False, angle_zero_to_pi=False, undefined=0):
    """
    computes image gradient magnitude and angle (by default from [-pi, pi])
    """
    if method == "sobel":  # 3x3
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    elif method == "prewitt":  # 3x3
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        dx = cv2.filter2D(image, cv2.CV_64F, kernelx)
        dy = cv2.filter2D(image, cv2.CV_64F, kernely)
    elif method == "central":  # 3x3
        dx = np.zeros_like(image, dtype=np.float64)
        dy = np.zeros_like(image, dtype=np.float64)
        dx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
        dy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
    elif method == "intermediate":  # forward 2x2
        dx = np.zeros_like(image, dtype=np.float64)
        dy = np.zeros_like(image, dtype=np.float64)
        dx[:, :-1] = image[:, 1:] - image[:, :-1]
        dy[:-1, :] = image[1:, :] - image[:-1, :]
    elif method == "roberts":  # 2x2
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        dx = cv2.filter2D(image, cv2.CV_64F, kernelx)
        dy = cv2.filter2D(image, cv2.CV_64F, kernely)
    elif method == "roberts2":  # 2x2
        dx = np.zeros_like(image, dtype=np.float64)
        dy = np.zeros_like(image, dtype=np.float64)
        dx[:-1, :-1] = (image[1:, 1:] - image[:-1, :-1]) / np.sqrt(2)
        dy[:-1, :-1] = (image[:-1, 1:] - image[1:, :-1]) / np.sqrt(2)
    else:
        raise ValueError("Invalid method specified")

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)

    if angle_zero_to_pi:
        angle = angle % (2 * np.pi)

    if angle_in_degrees:
        angle *= 180 / np.pi

    undefined_positions = (dx == 0) & (dy == 0)
    magnitude[undefined_positions] = undefined
    angle[undefined_positions] = undefined

    return magnitude, angle


def calc_mag_ang_zitnick(img, sigma=0.5, e=5.0e-2, alpha=0.25, lambda_val=2.0):
    """
    Compute normalized gradient of the input grayscale image.
    """
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    sigma_blur = np.sqrt((alpha * sigma) ** 2)
    kernel_blur = fspecial_gauss2D((1, int(8 * sigma_blur) + 1), sigma_blur)
    img_blurred = cv2.sepFilter2D(
        img, ddepth=-1, kernelX=kernel_blur, kernelY=kernel_blur)

    kernel = np.array([-1, 0, 1], dtype=np.float64).reshape(1, -1)
    dx = cv2.filter2D(img_blurred, ddepth=-1, kernel=kernel)
    dy = cv2.filter2D(img_blurred, ddepth=-1, kernel=kernel.T)

    magnitude = np.sqrt(dx ** 2 + dy ** 2).astype(np.float64)

    sigma_full = alpha * sigma * np.sqrt(lambda_val**2 - 1)
    kernel_avg = fspecial_gauss2D((1, int(8 * sigma_full) + 1), sigma_full)
    magnitude_avg = cv2.sepFilter2D(
        magnitude, ddepth=-1, kernelX=kernel_avg, kernelY=kernel_avg)

    magnitude = magnitude / np.maximum(magnitude_avg, e)
    magnitude[0, :] = 0
    magnitude[-1, :] = 0
    magnitude[:, 0] = 0
    magnitude[:, -1] = 0

    angle = np.arctan2(dy, dx)

    return magnitude, angle


def apply_CLAHE(image):
    def clahe(img):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return img

    if len(image.shape) == 3 and image.shape[2] == 3:
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_channel_clahe = clahe(l_channel)
        merged_channels = cv2.merge((l_channel_clahe, a_channel, b_channel))
        final_image = cv2.cvtColor(merged_channels, cv2.COLOR_Lab2BGR)
    elif len(image.shape) == 2 or image.shape[2] == 1:
        final_image = clahe(image)
    else:
        raise ValueError("Invalid number of channels.")

    return final_image


def apply_CLAHE_on_low_contrast(img):
    if is_low_contrast(img):
        img = apply_CLAHE(img)
    return img
