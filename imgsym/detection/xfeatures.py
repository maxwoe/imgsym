"""Feature-constellation reflection symmetry detection (SIFT / mirror descriptors).

Port/adaptation of:
    G. Loy, J.-O. Eklundh, "Detecting Symmetry and Symmetric Constellations of
    Features," ECCV 2006, LNCS 3952, pp. 508-521.

Mirror-matched feature pairs each carry a symmetry magnitude M = Phi * S * D
(angular Eq. 1, scale Eq. 2, distance Eq. 3) and vote in a (rho, theta) Hough
space; blurred peaks give the axes. Extends the original single-SIFT pipeline
by pooling matches from several descriptor matchers -- SIFT, ORB and the
mirror-invariant MI-SIFT / MIFT / MBR-SIFT.
"""
import copyreg
import logging
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from ..matching.descriptors import (MBRSIFTMatcher,
                                                      MIFTMatcher,
                                                      MISIFTMatcher,
                                                      ORBMatcher, SIFTMatcher)
from ..utils.image import rescale
from ..utils.visualization import MplColorHelper

log = logging.getLogger(__name__)


def calc_symmetry_lines(image, maxNOutputs=6, debug=False):

    image, rf = rescale(image, 600)

    scale_tolerance = 0.2  # percentage by which scale can vary within a matched pair

    h, w = image.shape[:2]
    diagonal = np.sqrt(h**2 + w**2)
    center = np.array(image.shape[:2]) / 2

    kp1, kp2, matches = [], [], []
    matchers = [
        SIFTMatcher(debug=False),
        ORBMatcher(debug=False),
        MISIFTMatcher(debug=False),
        MIFTMatcher(debug=False),
        MBRSIFTMatcher(debug=False)
    ]
    for matcher in matchers:
        log.debug(f"Running {matcher.__class__.__name__}…")
        kp1_, kp2_, matches_ = matcher.compute(image)
        log.debug(f" {len(matches_)} matches\n")

        # remember where we're about to insert them
        offset1 = len(kp1)
        offset2 = len(kp2)

        # extend the keypoint lists
        kp1.extend(kp1_)
        kp2.extend(kp2_)

        # shift indices on each match and append
        for m in matches_:
            new_m = cv2.DMatch(
                m.queryIdx + offset1,
                m.trainIdx + offset2,
                m.imgIdx,
                m.distance
            )
            matches.append(new_m)

    p = np.zeros((len(matches), 4))
    q = np.zeros((len(matches), 4))

    for i, match in enumerate(matches):
        point = deepcopy(kp1[match.queryIdx])
        point_m = deepcopy(kp2[match.trainIdx])

        # correct angle and pt cooridinates for mirror points
        point_m.angle = 180 - point_m.angle
        point_m.pt = (image.shape[1] - point_m.pt[0], point_m.pt[1])

        def build_arr(arr, kp):
            arr[i, 0] = kp.pt[0]
            arr[i, 1] = kp.pt[1]
            arr[i, 2] = np.deg2rad(kp.angle)
            arr[i, 3] = kp.size

        build_arr(p, point)
        build_arr(q, point_m)

    # compute constraints
    ang = calc_angle_x_axis(p, q)
    angular_symmetry = calc_angular_symmetry(p, q, ang)
    scale_diff = calc_scale_diff(p, q)

    # filter by constraints
    mask = (angular_symmetry > 0) & (scale_diff < scale_tolerance)
    ang = ang[mask]
    angular_symmetry = angular_symmetry[mask]
    p = p[mask]
    q = q[mask]

    # calculate weighting
    scale_weighting = calc_scale_weighting(p, q)
    distance_weighting = calc_distance_weighting(p, q)
    symmetry_magnitude = angular_symmetry * distance_weighting * scale_weighting

    # prepare Hough transform
    sym_x, sym_y = calc_midpoint(p, q)  # center point between p and q
    sym_x_c, sym_y_c = sym_x - w/2, sym_y - h/2  # origin in center of image
    ang_h = np.mod(ang, np.pi)
    r = sym_x_c * np.cos(ang_h) + sym_y_c * np.sin(ang_h)

    r_upper_bound = round(np.sqrt((h/2)**2 + (w/2)**2))
    ang_hs = np.floor(180/np.pi*ang_h).astype(int)  # theta Hough space
    r_hs = np.round(r + r_upper_bound).astype(int)  # rho Hough space

    # build Hough vote image
    H = np.zeros((np.max(ang_hs) + 1, np.max(r_hs) + 1))
    np.add.at(H, (ang_hs, r_hs), symmetry_magnitude)
    if debug:
        plt.imshow(H, cmap='jet')
        plt.show()

    # blur Hough space
    sigma_theta_bins = 3  # bins
    sigma_rho_bins = max(1, diagonal/2 * 0.01)  # bins
    H_blurred = gaussian_filter(H, sigma=(sigma_theta_bins, sigma_rho_bins), mode=('wrap', 'nearest'), truncate=3.0)

    if debug:
        figure, ax = plt.subplots(1)
        ax.imshow(H_blurred, cmap='jet')
        plt.show()

    # find peaks
    # fast way to find peaks in Hough space:
    # peaks = peak_local_max(H_blurred, min_distance=5,num_peaks=maxNOutputs, exclude_border=False, threshold_rel=0.25)
    # manually find peaks (with benefit of using wrap to avoid angular border issues)
    min_distance = 5
    threshold_rel = 0.3
    shape_size = (2*min_distance+1) ^ 2
    footprint = np.ones((shape_size, shape_size), bool)
    mf = maximum_filter(H_blurred, footprint=footprint, mode=('wrap', 'nearest'))
    mask = (H_blurred == mf) & (H_blurred > H_blurred.max() * threshold_rel)
    r_idx, t_idx = np.nonzero(mask)
    # keep only the N strongest
    strength = H_blurred[r_idx, t_idx]
    order = np.argsort(strength)[::-1][:maxNOutputs]
    peaks = np.column_stack((r_idx[order], t_idx[order]))

    if debug:
        figure, ax = plt.subplots(1)
        ax.imshow(H_blurred, cmap='jet')
        for idx, (y, x) in enumerate(peaks):
            ax.plot(x, y, 'rx', markersize=7)
            ax.annotate(f'{idx+1}', (x+3, y), color="r")
        plt.show()

    peak_count = len(peaks)
    max_ang = np.zeros(peak_count)
    max_r = np.zeros(peak_count)
    strength = np.zeros(peak_count)
    ind = [None] * peak_count
    left_ind = [None] * peak_count
    right_ind = [None] * peak_count
    pts = [None] * peak_count
    pts_m = [None] * peak_count

    angles, midpoints, seg_lengths, strengths = [], [], [], []

    # angular and radial tolerances for sym particles associated with the same symmetry axis
    grouping_tolerance = [3/180*np.pi, 2*sigma_rho_bins]

    for i, (theta, rho) in enumerate(peaks):
        max_ang[i] = theta * np.pi / 180
        max_r[i] = rho - r_upper_bound
        strength[i] = H_blurred[theta, rho]
        ind[i] = assign_to_axis(r, ang, max_r[i], max_ang[i], grouping_tolerance)
        left_ind[i], right_ind[i], pts[i], pts_m[i] = group_left_right(
            p, q, max_ang[i], ind[i])
        try:
            _, _, angle, midpoint, length = calc_line(
                center, max_r[i], max_ang[i], sym_x, sym_y, ind[i])
            if length > diagonal * 0.1:
                angles.append(angle)
                midpoints.append(midpoint)
                seg_lengths.append(length)
                strengths.append(strength[i])
        except Exception as e:
            # log.debug(i, ind[i], sym_x[ind[i]], sym_y[ind[i]])
            pass

    if debug:
        display_output(image, max_r, max_ang, ind, sym_x, sym_y,
                       left_ind, right_ind, pts, pts_m, strength)

    angles = np.array(angles)
    midpoints = np.array(midpoints)/rf
    seg_lengths = np.array(seg_lengths)/rf
    strengths = np.array(strengths)
    if strengths.size > 0:
        strengths = strengths/np.max(strengths)

    return angles, midpoints, seg_lengths, strengths


def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)


def filter_matches(matches, tresh=300):
    filtered = []
    for match in matches:
        if match.distance < tresh:
            filtered.append(match)
    return filtered


def calc_scale_diff(i, j):
    return (np.abs(i[:, 3] - j[:, 3]) / np.maximum(i[:, 3], j[:, 3]))


def calc_angular_symmetry(i, j, theta):
    # [-1, 1]
    Phi_ij = - np.cos(i[:, 2] + j[:, 2] - 2 * theta)
    return Phi_ij


def calc_scale_weighting(i, j, sigma=2):
    S_ij = np.exp(-abs(i[:, 3] - j[:, 3]) / sigma * (i[:, 3] + j[:, 3])) ** 2
    return S_ij


def calc_distance_weighting(i, j):
    dist_sq = (j[:, 0] - i[:, 0])**2 + (j[:, 1] - i[:, 1])**2
    sigma = np.sqrt(np.max(dist_sq))/6
    D_ij = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-dist_sq / (2 * sigma**2))
    return D_ij


def calc_midpoint(i, j):
    midpoint = (i[:, 0] + j[:, 0]) / 2, (i[:, 1] + j[:, 1]) / 2
    return midpoint


def calc_angle_x_axis(i, j):
    x, y = i[:, 0] - j[:, 0], i[:, 1] - j[:, 1]
    angle = np.arctan2(y, x)
    return angle


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


def group_left_right(p, q, max_ang, ind):
    p = p[ind][:, :2]
    q = q[ind][:, :2]

    norm_vec = np.array([np.cos(max_ang), np.sin(max_ang)])
    left_ind = np.where(np.dot(p - q, norm_vec) > 0)[0]
    right_ind = np.where(np.dot(p - q, norm_vec) < 0)[0]

    return left_ind, right_ind, p, q


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


def display_output(img, max_r, max_ang, ind, sym_x, sym_y, left_ind, right_ind, pts, pts_m, sym_strength):
    plt.figure()
    plt.imshow(img[:, :, ::-1])

    cmap = MplColorHelper("rainbow", 0, len(max_r))
    center = np.array(img.shape[:2]) / 2

    linewidth_max = np.mean(img.shape) * 0.01
    markersize = np.mean(img.shape) * 0.01

    for i in range(len(max_r)):
        col = cmap.get_rgb_float(i)
        linewidth = int(max(linewidth_max - (i*linewidth_max/4), 2))

        try:
            start, end, _, midpoint, _ = calc_line(
                center, max_r[i], max_ang[i], sym_x, sym_y, ind[i])
        except Exception as e:
            continue

        x_values = [start[0], end[0]]
        y_values = [start[1], end[1]]
        plt.plot(x_values, y_values, linewidth=linewidth, color=col)
        plt.plot(midpoint[0], midpoint[1],
                 'o', color=col, markersize=markersize)
        plt.annotate(f"{i}", (start[0], start[1]))

        # plt.plot(sym_x[ind[i]], sym_y[ind[i]], linewidth=linewidth, color=col)

        plt.plot(pts[i][right_ind[i], 0], pts[i][right_ind[i], 1],
                 '.', color=col, markersize=markersize)
        plt.plot(pts_m[i][left_ind[i], 0], pts_m[i]
                 [left_ind[i], 1], '.', color=col, markersize=markersize)
        plt.plot(pts[i][left_ind[i], 0], pts[i][left_ind[i], 1],
                 '.', color=col, markersize=markersize)
        plt.plot(pts_m[i][right_ind[i], 0], pts_m[i]
                 [right_ind[i], 1], '.', color=col, markersize=markersize)

    plt.show()
