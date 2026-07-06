"""Shape-signature reflection symmetry detection (R- and LIP-signatures).

Python port of the MATLAB method of:
    T.P. Nguyen, H.P. Truong, T.T. Nguyen, Y.-G. Kim, "Reflection symmetry
    detection of shapes based on shape signatures," Pattern Recognition 128
    (2022) 108667. doi:10.1016/j.patcog.2022.108667

Operates on a binary shape/mask. A 1-D shape signature -- the R-signature
(integral of squared Radon slices, Eq. 2) or the LIP-signature (largest
intersection / projection) -- is built over projection angles; its merit
profile (Pearson correlation of forward vs. backward circular shifts) yields
candidate axes, which are then verified by testing each Radon projection
C_theta for reflectional symmetry (Eq. 4). 'r_lip' = R-signature + LIP-signature.
"""
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import grey_opening, median_filter
from scipy.signal import find_peaks
from skimage.transform import radon

log = logging.getLogger(__name__)


def detect_centroid(im):
    """
    Detect the centroid of a binary image using image moments.
    """
    M = cv2.moments(im.astype(np.uint8))
    if M["m00"] != 0:
        c_row = M["m01"] / M["m00"]
        c_col = M["m10"] / M["m00"]
    else:
        c_row, c_col = im.shape[0] / 2, im.shape[1] / 2  # Fallback to image center
    return c_row, c_col


def dist_correlation(h1, h2):
    """
    Compute the distance correlation between two histograms.
    """
    h1_m = np.mean(h1)
    h2_m = np.mean(h2)

    d_enu = np.dot((h1 - h1_m), (h2 - h2_m))
    d_den = np.sqrt(np.dot((h1 - h1_m), (h1 - h1_m)) * np.dot((h2 - h2_m), (h2 - h2_m)))

    if d_den != 0:
        d = d_enu / d_den
        d = 1 - d
    else:
        d = np.inf
        log.warning('Singularity problem: one of the histograms is zero or uniform')
    return d


def corr_distance(x, y, distance_metric):
    """
    Calculate the correlation or distance correlation between two arrays.
    """
    len_diff = len(x) - len(y)
    if len_diff > 0:
        y = np.pad(y, (0, len_diff), 'constant')
    elif len_diff < 0:
        x = np.pad(x, (0, -len_diff), 'constant')

    if distance_metric.lower() == 'corr':
        if np.std(x) == 0 or np.std(y) == 0:
            return 0  # Avoid division by zero in correlation
        c = np.corrcoef(x, y)[0, 1]
    elif distance_metric.lower() == 'dist_corr':
        c = dist_correlation(x, y)
    else:
        raise ValueError("Unknown distance_metric")
    return c


def im_LIP(im, thetas=None):
    """
    Compute the LIP signature of an image.
    """
    if thetas is None:
        thetas = np.arange(1, 181)
    im_radon = radon(im, theta=thetas, circle=False)

    # Normalization
    n = len(thetas)
    pw = np.zeros(n)
    lp = np.zeros(n)
    iL1 = np.zeros(n, dtype=int)
    iL2 = np.zeros(n, dtype=int)

    for k in range(n):
        ct = im_radon[:, k]
        it1 = 0
        it2 = len(ct) - 1
        while it1 < len(ct) and ct[it1] == 0:
            it1 += 1
        while it2 >= 0 and ct[it2] == 0:
            it2 -= 1
        it2 = max(it2, it1)  # Ensure it2 is not less than it1
        pw[k] = it2 - it1 + 3
        iL1[k] = it1
        iL2[k] = it2

    for k in range(n):
        ct = im_radon[iL1[k]:iL2[k]+1, k]
        if len(ct) == 0:
            lp[k] = 0
        else:
            m = np.max(ct)
            lp[k] = m / pw[k]

    # Smoothing
    lp = median_filter(lp, size=5)
    return lp


def radon_profile(profile):
    """
    Process the Radon profile by applying morphological opening and trimming low-intensity values.
    """
    delta = 1
    # Apply morphological opening with a disk structuring element
    structuring_element_size = 3  # Equivalent to [1,1,1] in MATLAB
    opened = grey_opening(profile, size=structuring_element_size)

    # Trim the profile based on the delta threshold
    ib = np.argmax(opened > delta)
    ie = len(opened) - np.argmax(np.flip(opened) > delta) - 1

    if opened[ib:ie+1].size > 0:
        ctheta = opened[ib:ie+1]
    else:
        ctheta = np.array([])
    return ctheta


def rtrans(im, thetas, m, norm=0):
    """
    Compute the generic R-transform of an image.
    """
    im_radon = radon(im, theta=thetas, circle=False)
    im_rtrans = np.zeros((len(m), len(thetas)))

    for i, exponent in enumerate(m):
        im_rtrans[i, :] = np.sum(im_radon**exponent, axis=0)

    # Normalization to have area under the curve == 1
    if norm != 0:
        im_rtrans = im_rtrans / np.sum(im_rtrans, axis=1, keepdims=True)

    return im_rtrans, im_radon


def plot_symmetry_axis(mask, symmetry_directions):
    center_y, center_x = detect_centroid(mask)

    out = cv2.cvtColor((255 * mask).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    col = (255, 0, 0)
    linewidth = 2

    rows, cols = mask.shape[:2]
    diagonal = np.sqrt(rows**2 + cols**2)
    mp = (int(center_x), int(center_y))  # x, y
    sl = diagonal/2

    for theta in symmetry_directions:
        # Adjust theta (add 90° to get the drawing angle)
        ag = np.deg2rad(theta + 90)
        # Calculate end points for the line.
        p1 = (int(mp[0] + sl * np.cos(ag)), int(mp[1] - sl * np.sin(ag)))
        p2 = (int(mp[0] - sl * np.cos(ag)), int(mp[1] + sl * np.sin(ag)))
        out = cv2.line(out, p1, p2, col, linewidth)
        out = cv2.circle(out, (mp[0], mp[1]), radius=5, color=col, thickness=-1)

    plt.imshow(out)
    plt.show()


def detect_reflection_sym(mask, threshold=0.1, mode="single", type_sig="LIP", dist_metric="corr", debug=False):
    """
    Detect reflection symmetry axes in a binary image.
    """
    angleR = []
    measureR = []
    thetas = np.arange(0, 180)
    n = len(thetas)
    reflection_merit = np.zeros(n)

    if type_sig.upper() == 'LIP':
        lip = im_LIP(mask, thetas)
        l = np.concatenate([lip, lip, lip])
    elif type_sig.upper() == 'R':
        rsig, _ = rtrans(mask, thetas, m=[2])
        rsig = rsig.flatten()
        l = np.concatenate([rsig, rsig, rsig])
    else:
        raise ValueError("Unknown type_sig")

    # Compute reflection merit for each theta
    for i in range(n):
        id = i + n
        l1 = l[id:id + n]
        l2 = l[id - n + 1:id + 1][::-1]
        reflection_merit[i] = corr_distance(l1, l2, dist_metric)

    # Handle circularity by appending first 10 elements
    reflection_merit = np.concatenate([reflection_merit, reflection_merit[:10]])

    # Find peaks
    peaks, _ = find_peaks(reflection_merit)
    measures = reflection_merit[peaks]
    angles = peaks

    # Filter peaks based on threshold
    valid = measures > threshold
    angles = angles[valid]
    measures = measures[valid]

    m_x = 0
    angle_x = 0
    for angle, measure in zip(angles, measures):
        if angle > 180:
            angle -= 180
        if angle <= 180:
            profile = radon(mask, theta=[angle], circle=False).flatten()
            c_theta = radon_profile(profile)
            if c_theta.size == 0:
                continue  # Skip if profile is empty after trimming
            c_theta = c_theta * (c_theta > 5)
            m = corr_distance(c_theta, c_theta[::-1], dist_metric)
            log.debug(f"{angle}, {m}")
            if mode.lower() == 'single':
                if m > m_x:
                    m_x = m
                    angle_x = angle
            elif mode.lower() == 'multi':
                if m > threshold:
                    angleR.append(angle)
                    measureR.append(m)
            else:
                raise ValueError("Unknown mode")

    if mode.lower() == 'single' and m_x > threshold:
        angleR.append(angle_x)
        measureR.append(m_x)

    sorted_pairs = sorted(zip(angleR, measureR), key=lambda x: x[1], reverse=True)
    sorted_angles, sorted_measures = zip(*sorted_pairs) if sorted_pairs else ([], [])

    if debug:
        plot_symmetry_axis(mask, angleR)

    log.debug(f"{angleR}, {measureR}")

    return sorted_angles, sorted_measures
