"""Wavelet-based reflection symmetry detection (textural + color histograms).

Port/adaptation of:
    M. Elawady, C. Ducottet, O. Alata, C. Barat, P. Colantoni,
    "Wavelet-Based Reflection Symmetry Detection via Textural and Color
    Histograms," ICCV Workshops (ICCVW), 2017, pp. 1725-1733.
    doi:10.1109/ICCVW.2017.202

Log-Gabor (phase-congruency) edge features are sampled on a regular grid; each
carries an orientation histogram (textural) and an HSV histogram (color, 8:2:2
hue:sat:val sampling -- paper Eqs. 6-7). Symmetric feature pairs vote in a
(rho, theta) Hough space with weight w = m * t * q (mirror Eq. 9 x textural
Eq. 10 x color Eq. 11); blurred peaks give the axes. The trailing symmetry-
center / diameter / bounding-box estimation is an added object-localization
step, not part of Elawady et al.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
from itertools import combinations
from skimage.feature import peak_local_max
from skimage.exposure import is_low_contrast
from ..utils.image import rescale, get_colorspace

try:
    from phasepack import phasecong
except ImportError:
    phasecong = None

log = logging.getLogger(__name__)


def calc_symmetry_lines(image, maxNOutputs=6, debug=False):

    if phasecong is None:
        raise ImportError(
            "The 'phasepack' package is required for wavelet-based symmetry detection. "
            "Install it with: pip install phasepack"
        )

    colorspace = get_colorspace(image, "luminance")
    # colorspace = cv2.bilateralFilter(colorspace, 9, 75, 75)
    colorspace, rf = rescale(colorspace, 400)
    image, rf = rescale(image, 400)

    grouping_tolerance = [3/180*np.pi, 3]

    h, w = image.shape[:2]
    diagonal = np.sqrt(h**2 + w**2)
    center = np.array(image.shape[:2]) / 2

    magnitude, _, orientation, _, _, _, _ = phasecong(colorspace)
    magnitude = cv2.normalize(magnitude, None, 0, 1,
                              cv2.NORM_MINMAX, cv2.CV_32F)

    patch_size = int(diagonal / 50)
    feature_points = extract_feature_points(
        magnitude, patch_size, 0.2, patch_size//2)
    if debug:
        log.debug(f"Feature points: {len(feature_points)}")
        plt.imshow(magnitude), plt.show()
        plt.imshow(orientation, cmap='jet'), plt.show()
        visualize_feature_points(
            image, patch_size, feature_points, patch_size//2)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram_whole_image = calc_color_histogram(image_hsv)

    # Compute features
    coordinates = np.zeros((len(feature_points), 2))
    color_hists = np.zeros((len(feature_points), 32))
    orientation_hists = np.zeros((len(feature_points), 32))

    for i, (x, y) in enumerate(feature_points):
        hHSV = calc_hHSV(image_hsv, x, y, patch_size)
        ho = calc_ho(orientation, x, y, patch_size)
        coordinates[i, 0] = x
        coordinates[i, 1] = y
        color_hists[i, :] = hHSV
        orientation_hists[i, :] = ho

    # all possible combinations
    indices = np.array(list(combinations(range(len(coordinates)), 2)))
    p = coordinates[indices[:, 0]].astype(int)
    q = coordinates[indices[:, 1]].astype(int)

    ang = calc_angle_x_axis(p, q)
    distance_weighting = calc_distance_weighting(p, q)

    orientation_sim = calc_orientation_similarity(
        orientation_hists[indices[:, 0]], np.flip(orientation_hists[indices[:, 1]]))
    color_sim = calc_color_similarity(
        color_hists[indices[:, 0]], color_hists[indices[:, 1]])
    color_sim_weighted = calc_color_similarity_weighted(
        color_hists[indices[:, 0]], color_hists[indices[:, 1]], histogram_whole_image)

    tau_p = np.array([np.cos(orientation[p[:, 1], p[:, 0]]),
                      np.sin(orientation[p[:, 1], p[:, 0]])]).T
    tau_q = np.array([np.cos(orientation[q[:, 1], q[:, 0]]),
                      np.sin(orientation[q[:, 1], q[:, 0]])]).T
    T_pq = ((q-p) / np.linalg.norm(q-p, axis=1, keepdims=True))
    T_perp_pq = np.stack([-T_pq[:, 1], T_pq[:, 0]], axis=-1)
    theta = np.arctan2(T_perp_pq[:, 1], T_perp_pq[:, 0])
    cos_theta = np.cos(2 * theta)
    sin_theta = np.sin(2 * theta)
    reverse_matrix = np.stack([cos_theta, sin_theta, sin_theta, -cos_theta],
                              axis=-1).reshape(-1, 2, 2)
    mirror_sim = np.abs(
        np.einsum('ij,ijk,ik->i', tau_q, reverse_matrix, tau_p))  # * magnitude[p[:, 1], p[:, 0]] * magnitude[q[:, 1], q[:, 0]]

    # prepare Hough transform
    sym = calc_midpoint(p, q)  # center point between p and q
    sym_x, sym_y = sym[:, 0], sym[:, 1]
    sym_x_c, sym_y_c = sym_x - w/2, sym_y - h/2  # origin in center of image
    ang_h = np.mod(ang, np.pi)
    r = sym_x_c * np.cos(ang_h) + sym_y_c * np.sin(ang_h)

    r_upper_bound = round(np.sqrt((h/2)**2 + (w/2)**2))
    ang_hs = np.floor(180/np.pi*ang_h).astype(int)  # theta hough space
    r_hs = np.round(r + r_upper_bound).astype(int)  # rho hough space

    ##############################
    # Symmetry axis detection
    ##############################

    # build Hough vote image
    H_sym_axis = np.zeros((np.max(ang_hs) + 1, np.max(r_hs) + 1))
    weights = mirror_sim * color_sim_weighted * orientation_sim * distance_weighting
    np.add.at(H_sym_axis, (ang_hs, r_hs), weights)
    if debug:
        plt.imshow(H_sym_axis, cmap="jet")
        plt.show()

    k = 5
    hist_sym_axis = cv2.GaussianBlur(H_sym_axis, (2*k+1, 2*k+1), k/2)

    # find peaks
    peaks_sym_axis = peak_local_max(hist_sym_axis, min_distance=10,
                                    num_peaks=maxNOutputs, exclude_border=False, threshold_rel=0.25)
    if debug:
        figure, ax = plt.subplots(1)
        ax.imshow(hist_sym_axis, cmap="jet")
        for idx, (y, x) in enumerate(peaks_sym_axis):
            ax.plot(x, y, 'rx', markersize=7)
            ax.annotate(f'{idx+1}', (x+3, y), color="r")
        plt.show()

    # peaks_sym_axis = merge_peaks(peaks_sym_axis)

    peak_count = len(peaks_sym_axis)
    max_ang = np.zeros(peak_count)
    max_diameter = np.zeros(peak_count)
    strength = np.zeros(peak_count)
    ind = [None] * peak_count

    angles, midpoints, seg_lengths, strengths = [], [], [], []

    for i, (theta, rho) in enumerate(peaks_sym_axis):
        max_ang[i] = theta * np.pi / 180
        max_diameter[i] = rho - r_upper_bound
        strength[i] = hist_sym_axis[theta, rho]
        ind[i] = assign_to_axis(r, ang, max_diameter[i],
                                max_ang[i], grouping_tolerance)
        try:
            _, _, angle, midpoint, length = calc_line(
                center, max_diameter[i], max_ang[i], sym_x, sym_y, ind[i])
            if length > diagonal * 0.1:
                angles.append(angle)
                midpoints.append(midpoint)
                seg_lengths.append(length)
                strengths.append(strength[i])
        except Exception as e:
            # log.debug(i+1, ind[i], sym_x[ind[i]], sym_y[ind[i]])
            # log.debug(f"Could not calculate line for peak #{i+1}.")
            pass

    angles = np.array(angles)
    midpoints = np.array(midpoints)/rf
    seg_lengths = np.array(seg_lengths)/rf
    strengths = np.array(strengths)
    if strengths.size > 0:
        strengths = strengths/np.max(strengths)

    return angles, midpoints, seg_lengths, strengths


def calc_angle_x_axis(i, j):
    x, y = i[:, 0] - j[:, 0], i[:, 1] - j[:, 1]
    angle = np.arctan2(y, x)
    return angle


def calc_midpoint(i, j):
    x, y = (i[:, 0] + j[:, 0]) / 2, (i[:, 1] + j[:, 1]) / 2
    return np.array([x, y]).T


def calc_distance(i, j):
    distance = np.linalg.norm(i - j, axis=1)
    return distance


def calc_distance_weighting(i, j):
    dist_sq = (j[:, 0] - i[:, 0])**2 + (j[:, 1] -
                                        i[:, 1])**2  # squared euclidean distance
    # with larger sigma weights will decay more slowly, i.e. points that are farther apart will still have relatively high weights
    sigma = np.sqrt(np.max(dist_sq))/6
    weight = 1 / (sigma * np.sqrt(2 * np.pi)) * \
        np.exp(-dist_sq / (2 * sigma**2))
    return weight


def calc_ho(orientation, x, y, patch_size, N=32):
    patch_orientation = orientation[max(0, y - patch_size // 2):min(orientation.shape[0], y + patch_size // 2 + 1),
                                    max(0, x - patch_size // 2): min(orientation.shape[1], x + patch_size // 2 + 1)]

    bins = np.linspace(0, np.rad2deg(np.pi), N + 1)
    ho, _ = np.histogram(patch_orientation, bins=bins)
    # Circularly shift ho with respect to the orientation of the maximal magnitude among the patch Ji
    max_orientation = np.argmax(ho)
    ho = np.roll(ho, -max_orientation)
    ho = ho / np.sum(ho)
    return ho


def calc_hHSV(image_hsv, x, y, patch_size):
    patch_hsv = image_hsv[max(0, y - patch_size // 2): min(image_hsv.shape[0], y + patch_size // 2 + 1),
                          max(0, x - patch_size // 2): min(image_hsv.shape[1], x + patch_size // 2 + 1)]
    return calc_color_histogram(patch_hsv)


def calc_color_histogram(image_hsv, C1=8, C2=2, C3=2):
    H, S, V = cv2.split(image_hsv)
    H = H / 180.0
    S = S / 255.0
    V = V / 255.0
    bins = [np.linspace(0, 1, C1 + 1), np.linspace(0, 1,
                                                   C2 + 1), np.linspace(0, 1, C3 + 1)]
    hist, _ = np.histogramdd([H.flatten(), S.flatten(), V.flatten()],
                             bins=bins)
    hHSV = hist.flatten().astype(np.int32)
    hHSV = hHSV / np.sum(hHSV)
    return hHSV


def calc_color_similarity(hist_i, hist_j, method=1):
    match method:
        case 0:
            similarity = np.sum(np.minimum(hist_i, hist_j))
        case 1:
            # dot product vectorized
            similarity = np.sum(hist_i * hist_j, axis=-1)
        case 2:
            dot_product = np.sum(hist_i * hist_j, axis=1)
            norms_hist_i = np.linalg.norm(
                hist_i, axis=1)  # L2 norm for each row
            norms_hist_j = np.linalg.norm(hist_j, axis=1)
            similarity = dot_product / (norms_hist_i * norms_hist_j)
    return similarity


# penalize pixels that frequently appear
def calc_color_similarity_weighted(hist_i, hist_j, histogram_whole_image, method=0):
    epsilon = 1e-9
    match method:
        case 0:
            similarity = np.sum(np.minimum(hist_i, hist_j) /
                                (histogram_whole_image + epsilon))
        case 1:
            norm_hist_i = hist_i / (histogram_whole_image + epsilon)
            norm_hist_j = hist_j / (histogram_whole_image + epsilon)
            similarity = np.sum(norm_hist_i * norm_hist_j,
                                axis=-1)  # dot product vectorized
    return similarity


def calc_orientation_similarity(hist_i, hist_j, method=1):
    match method:
        case 0:
            similarity = np.sum(np.minimum(hist_i, hist_j))
        case 1:
            # dot product vectorized
            similarity = np.sum(hist_i * hist_j, axis=-1)
        case 2:
            dot_product = np.sum(hist_i * hist_j, axis=-1)
            norms_hog_1 = np.linalg.norm(hist_i, axis=1)
            norms_hog_2 = np.linalg.norm(hist_j, axis=1)
            similarity = dot_product / (norms_hog_1 * norms_hog_2)
    return similarity


def extract_feature_points(image, patch_size, kf=0.2, edge_margin=0):
    stride = patch_size
    height, width = image.shape[:2]
    fmax = np.max(image)
    feature_points = []

    # Update loop ranges to consider edge_margin
    for y in range(edge_margin, height - patch_size - edge_margin + 1, stride):
        for x in range(edge_margin, width - patch_size - edge_margin + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            max_amplitude = np.max(patch)

            if max_amplitude <= kf * fmax:
                continue

            y_argmax, x_argmax = divmod(np.argmax(patch), patch_size)
            feature_point = (x + x_argmax, y + y_argmax)
            feature_points.append(feature_point)

    return feature_points


def visualize_feature_points(image, patch_size, feature_points, edge_margin=0):
    tmp = image.copy()
    height, width = image.shape[:2]

    stride = patch_size
    # Draw the grid on the image
    for y in range(edge_margin, height - patch_size - edge_margin + 1, stride):
        for x in range(edge_margin, width - patch_size - edge_margin + 1, stride):
            cv2.rectangle(tmp, (x, y), (x + patch_size,
                          y + patch_size), (255, 0, 0), 1)

    for point in feature_points:
        cv2.circle(tmp, point, radius=1, color=(0, 255, 0), thickness=2)

    plt.imshow(tmp[..., ::-1])
    plt.show()


def calc_line(center, max_r, max_ang, sym_x, sym_y, ind):
    tol = 5
    ang_tol = np.pi / 4

    if not ind or len(sym_x[ind]) == 0 or len(sym_y[ind]) == 0:
        raise Exception

    if (np.abs(max_ang) < ang_tol) or (np.abs(max_ang - np.pi) < ang_tol):
        Y = np.linspace(sym_y[ind].min() - tol, sym_y[ind].max() + tol, 101)
        X = ((-Y + center[0]) * np.sin(max_ang) + max_r) / \
            np.cos(max_ang) + center[1]
    else:
        X = np.linspace(sym_x[ind].min() - tol, sym_x[ind].max() + tol, 101)
        Y = ((-X + center[1]) * np.cos(max_ang) + max_r) / \
            np.sin(max_ang + 1e-8) + center[0]

    p_ind = np.where((X >= (sym_x[ind].min() - tol)) & (X <= (sym_x[ind].max() + tol)) &
                     (Y >= (sym_y[ind].min() - tol)) & (Y <= (sym_y[ind].max() + tol)))

    if p_ind[0].size > 0:
        start = (X[p_ind][0], Y[p_ind][0])
        end = (X[p_ind][-1], Y[p_ind][-1])
        x, y = start[0] - end[0], start[1] - end[1]
        angle = -np.arctan2(y, x) % np.pi
        # angle = (-np.arctan2(y, x) + np.pi/2) % np.pi
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        segment_length = np.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    else:
        raise Exception

    return start, end, angle, midpoint, segment_length


def assign_to_axis(r, ang, max_r, max_ang, grouping_thresh):
    temp = np.unwrap(np.column_stack(
        (ang, np.full(len(ang), max_ang))), axis=1)
    d_ang1 = np.abs(temp[:, 0] - temp[:, 1])
    d_r1 = np.abs(r - max_r)

    temp = np.unwrap(np.column_stack(
        (ang + np.pi, np.full(len(ang), max_ang))), axis=1)
    d_ang2 = np.abs(temp[:, 0] - temp[:, 1])
    d_r2 = np.abs(-r - max_r)

    d_ang = np.minimum(d_ang1, d_ang2)
    iind = np.argmin(np.column_stack((d_ang1, d_ang2)), axis=1)
    d_r = np.zeros_like(r)
    d_r[iind == 0] = d_r1[iind == 0]
    d_r[iind == 1] = d_r2[iind == 1]

    ind = np.where((d_ang < grouping_thresh[0]) & (d_r < grouping_thresh[1]))

    return ind


def merge_peaks(peaks, angular_thresh=5, radial_thresh=10):
    """ Merges peaks near 0 and 180 degrees with similar r values. """

    def find_twin(peak):
        """Helper function to find the twin peak."""
        for p in peaks:
            if abs(p[1] - peak[1]) < radial_thresh:
                if peak[0] <= angular_thresh and abs(p[0] - 180) < angular_thresh:
                    return p
                elif peak[0] >= (180 - angular_thresh) and p[0] < angular_thresh:
                    return p
        return None

    final_peaks = []
    processed = set()  # to keep track of peaks we've already considered

    for peak in peaks:
        if tuple(peak) in processed:
            continue

        theta, rho = peak[0], peak[1]
        twin = find_twin(peak)

        # If the peak is near 0 and has a twin near 180
        if theta <= angular_thresh and twin is not None:
            avg_deviation = (theta + (180 - twin[0])) / 2
            merged_angle = math.floor(0 + avg_deviation)
            final_peaks.append([merged_angle, (rho + twin[1]) // 2])
            processed.add(tuple(twin))

        # If the peak is near 180 and has a twin near 0, we won't process it here,
        # because it'll be processed when we encounter its twin (near 0).
        elif theta >= (180 - angular_thresh) and twin is not None:
            continue

        # For all other peaks including those near 180° without a twin or near 0° without a twin
        else:
            final_peaks.append([theta, rho])

        processed.add(tuple(peak))

    return np.array(final_peaks, dtype=np.int64)
