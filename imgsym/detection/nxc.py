"""Registration-based reflection symmetry detection (normalized cross-correlation).

Port/adaptation of:
    M. Cicconet, D. G. C. Hildebrand, H. Elliott, "Finding Mirror Symmetry via
    Registration and Optimal Symmetric Pairwise Assignment of Curves," ICCV
    Workshops (ICCVW), 2017, pp. 1749-1758. doi:10.1109/ICCVW.2017.206

Mirror Symmetry via Registration (MSR): reflect the image about a candidate line,
register the reflection back onto the original, and recover the axis from the
reflection+registration mapping. Registration is the paper's RANSAC consensus over
an ensemble of patch-to-image normalized-cross-correlation (normxcorr2) matches --
swept over rotation angles, keeping correlation peaks above a threshold, then taking
the K best maxima. The name 'nxc' mirrors the paper's Fig. 3a label for this NCC
registration back-end.
"""
import numpy as np
import cv2
import math

from functools import partial
from scipy.ndimage import correlate
from scipy.signal import convolve2d, find_peaks
from skimage.transform import rotate, warp
from matplotlib import pyplot as plt

from ..utils.image import rescale, fspecial_gauss2D, get_colorspace

from multiprocessing.pool import ThreadPool as Pool


def calc_symmetry_lines(I, boxSize=50, nBoxSamples=100, maxNOutputs=6, angleSet=np.arange(0, 360, 6), multiprocessing=True):
    """Computes symmetry axis in 2D images via registration.

    Args:
        I (_type_): _description_
        boxSize (_type_, optional): Dimensions of patch for 'Normalized Cross Correlation' registration. Defaults to 50.
        nBoxSamples (_type_, optional): Number of patch samples for RANSAC. Defaults to 100.
        maxNOutputs (int, optional): Maximum number of output symmetry lines. Defaults to 4.
        angleSet (_type_, optional): Range of rotation angles (in degrees in counter-clockwise direction) used by registration algorithm.
            An angle of 0 degrees means that the image is not rotated (i.e. in a row/column coordinate system, the 0 degree line is a vertical line along the y-axis). Defaults to np.arange(0, 360, 6).
        multiprocessing (bool, optional): Whether to use multiprocessing for computation. Defaults to True.

    Returns:
        angles: Angles of symmetry lines
        midPoints: Midpoints of symmetry lines
        segLengths: Lenghts of symmetry lines
        strengths: Strenghts of symmetry lines (sorted in descending order, so the first line is always the strongest guess)
    """
    I = get_colorspace(I, "luminance")
    # No CLAHE / bilateral: the original (Cicconet et al., normxcorr2 registration)
    # applies no contrast enhancement -- keep the port faithful to it.

    newImSize = 200
    I, rf = rescale(I, newImSize)
    G = imgradient(I)[0]
    GI = normalize(G)
    I = normalize(I)

    cellp, cellv, srmags = eigsymNXC(
        GI, 0, boxSize, nBoxSamples, maxNOutputs, angleSet, multiprocessing=multiprocessing)
    srmags = srmags/np.max(srmags)
    strengths = srmags[0:len(cellp)]

    nOutputs = len(cellp)
    angles = np.zeros((nOutputs))
    midPoints = np.zeros((nOutputs, 2))
    segLengths = np.zeros((nOutputs))

    for iOutput in range(nOutputs):
        p = cellp[iOutput]
        v = cellv[iOutput]
        ag = np.arctan2(v[0], v[1])
        if ag < 0:
            ag = ag + np.pi
        xy = []
        h, w = I.shape
        for j in range(1, round(math.sqrt(2 * (newImSize ** 2)))):
            x = round(p[1] + j * math.cos(ag))
            y = round(p[0] + j * math.sin(ag))
            if x >= 1 and x <= h and y >= 1 and y <= w:
                xy.append((x, y))
            x = round(p[1] - j * math.cos(ag))
            y = round(p[0] - j * math.sin(ag))
            if x >= 1 and x <= h and y >= 1 and y <= w:
                xy.append((x, y))
        if xy:
            xy = np.round(np.mean(xy, axis=0))
        else:
            xy = np.round(np.array(np.shape(I)) / 2.0)

        midpointI, seglenI, _ = endpoints(I, 1, ag, xy)
        midpointI = midpointI/rf
        seglenI = seglenI/rf

        # row/col (y, x) coordinate system to a col/row (x, y) system
        # swap midpoint coords: y, x -> x, y;
        # translate angle: translate y-axis basis to x-axis and normalize to the range of [0, pi)
        angles[iOutput] = (ag - np.pi/2) % np.pi  # ag
        midPoints[iOutput] = midpointI[::-1]  # midpointI
        segLengths[iOutput] = seglenI

    # print(f"angles, midPoints, segLengths, strenghts: {angles}, {midPoints}, {segLengths}, {strengths}")
    return angles, midPoints, segLengths, strengths


def eigsymNXC(I, refAngle=0, boxSize=50, nBoxSamples=100, maxNOutputs=1, angleSet=np.arange(0, 360, 6), multiprocessing=True):
    # I should be double, in range [0,1]

    # reflect
    h, w = I.shape
    p = np.array([[w/2], [h/2]])
    N = np.array([[math.cos(refAngle)], [math.sin(refAngle)]])
    d = p*N
    S = np.block([[np.eye(2)-2*(N*N.T), 2*d*N], [0, 0, 1]])

    tform = S.T
    J = np.roll(warp(I, tform.T, mode='wrap'), -1, axis=1)

    # register
    tforms, srmags = computeNormxcorrTransforms(
        J, I, boxSize, nBoxSamples, maxNOutputs, angleSet, multiprocessing=multiprocessing)

    nTForms = len(tforms)
    p = {}
    v = {}

    for itform in np.arange(0, nTForms):
        tform = tforms[itform]

        R = tform.T  # compute sym line

        t = R[0:2, 2]
        T = np.dot(S[0:2, 0:2], R[0:2, 0:2].T)
        D, V = np.linalg.eigh(T)
        ieig = []

        check = np.abs(D + 1.0)
        ieig = np.argmin(check)

        v[itform] = V[:, ieig]  # eigenvector of eigenvalue -1
        v[itform] = np.array([- v[itform][1], v[itform][0]])  # perp

        p[itform] = (((R[0:2, 0:2]@(2*d*N)).T+t.T)/2)[0]  # point in line

    # print(f"p, v, srmags: {p}, {v}, {srmags}")
    return p, v, srmags


def computeAngle(iangle, rangles, boxSize, nBoxSamples, I, J):
    # print(f"iangle: {iangle}, {rangles[iangle]}")

    numrows, numcols = I.shape
    A = np.zeros((2*numrows, 2*numcols))
    flipI = rotate(J, rangles[iangle],
                   preserve_range=True, mode='constant', order=0)

    rng = np.random.default_rng(42)
    for index in np.arange(0, nBoxSamples):
        w = boxSize
        h = w
        x0 = math.floor(np.dot(flipI.shape[1] - w, rng.uniform()))
        y0 = math.floor(np.dot(flipI.shape[0] - h, rng.uniform()))
        xcFlipI = x0 + w / 2
        ycFlipI = y0 + h / 2

        subFlipI = flipI[y0:y0+h, x0:x0+w]
        if np.var(np.ravel(subFlipI)) > 0.001:
            ROI, c, mc = locateSubset(subFlipI, I)
            if mc > 0.25:  # maximum correlation above treshold
                xcI = ROI[0] + ROI[2] / 2
                ycI = ROI[1] + ROI[3] / 2

                v = np.array([xcI, ycI]) - \
                    np.array([xcFlipI, ycFlipI])  # translation

                row = min(round(v[1]+numrows-1), 2*numrows-1)
                col = min(round(v[0]+numcols-1), 2*numcols-1)
                A[row, col] = A[row, col] + 1

    A = correlate(A, fspecial_gauss2D((12, 12), 3), mode='constant', origin=-1)
    maxA = np.max(np.ravel(A))

    indexes = np.where(A.T == maxA)
    r = indexes[1]
    c = indexes[0]
    v = np.array((0, 0))
    v[0] = c[0] - numcols  # + 1
    v[1] = r[0] - numrows  # + 1

    # print(f"iangle, maxA, v: {iangle}, {maxA}, {v}")
    return iangle, maxA, v


def computeNormxcorrTransforms(J=None, I=None, boxSize=None, nBoxSamples=None, maxNOutputs=None, angleSet=None, multiprocessing=True):
    numrows, numcols = I.shape
    rangles = angleSet
    rmags = np.zeros((1, len(rangles)))
    vs = np.zeros((len(rangles), 2))

    if multiprocessing:
        with Pool() as pool:
            for r in pool.map(partial(computeAngle, rangles=rangles, boxSize=boxSize, nBoxSamples=nBoxSamples, I=I, J=J), np.arange(0, len(rangles))):
                rmags[0, r[0]] = r[1]
                vs[r[0], :] = r[2]
    else:
        for iangle in range(0, len(rangles)):
            r = computeAngle(iangle, rangles, boxSize, nBoxSamples, I, J)
            rmags[0, r[0]] = r[1]
            vs[r[0], :] = r[2]

    iangles = np.argsort(-rmags, None, kind='quicksort')
    srmags = np.take_along_axis(rmags.flatten(), iangles, axis=0)
    nTForms = min(len(rangles), maxNOutputs)
    tforms = {}

    for i in np.arange(0, nTForms):
        iangle = iangles[i]
        v = vs[iangle, :]
        arad = -rangles[iangle]/360*2*np.pi

        # rotation with respect to center
        T1 = np.block(
            [[np.eye(2), np.array([[- numcols / 2], [- numrows / 2]])], [0, 0, 1]])
        T2 = np.block([[np.array([[np.cos(arad), -math.sin(arad)],
                      [math.sin(arad), math.cos(arad)]]), np.array([[0], [0]])], [0, 0, 1]])
        T3 = np.block(
            [[np.eye(2), np.array([[numcols / 2], [numrows / 2]])], [0, 0, 1]])
        # translation
        T4 = np.block([[np.eye(2), v[np.newaxis].T], [0, 0, 1]])

        # transform
        tform = (np.dot(np.dot(np.dot(T4, T3), T2), T1)).T

        tforms[i] = tform

    # print(f"tforms: {tforms}")
    # print(f"srmags: {srmags}")
    return tforms, srmags


def locateSubset(subI=None, I=None):
    I = np.pad(I, (subI.shape[0]-1, subI.shape[1]-1))
    # Zero-mean normalized cross-correlation == MATLAB normxcorr2 (the original).
    # TM_CCORR_NORMED (no mean subtraction) was a port deviation; the mc>0.25 gate
    # in computeAngle is the original's threshold and only matches under CCOEFF.
    # nan_to_num mirrors normxcorr2 returning 0 for zero-variance (border) windows.
    c = cv2.matchTemplate(I.astype(np.float32), subI.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    mc = np.max(c)  # max correlation
    indexes = np.where(c.T == mc)
    ypeak = indexes[1]
    xpeak = indexes[0]
    yoffSet = ypeak[0] - subI.shape[0]
    xoffSet = xpeak[0] - subI.shape[1]
    ROI = np.array([xoffSet + 2, yoffSet + 2, subI.shape[1], subI.shape[0]])

    return ROI, c, mc


def cmorlet(sigma, freq, angle, halfkernel):
    # ref: http://arxiv.org/pdf/1203.1513.pdf, page 2
    support = 2.5*sigma

    xmin = -support
    xmax = -xmin
    ymin = xmin
    ymax = xmax
    xdomain = np.arange(xmin, xmax+1)
    ydomain = np.arange(ymin, ymax+1)
    x, y = np.meshgrid(xdomain, ydomain)

    xi = freq*np.asarray([math.sin(angle), math.cos(angle)])

    envelope = np.exp(
        np.dot(-0.5, (np.multiply(x, x) + np.multiply(y, y))) / sigma ** 2)
    carrier = np.exp(np.dot(1j, (np.dot(xi[0], x) + np.dot(xi[1], y))))

    C2 = np.sum(np.sum(np.multiply(envelope, carrier))) / \
        np.sum(np.sum(envelope))
    arg = np.multiply((carrier - C2), envelope)

    normfact = np.sum(np.sum(np.multiply(arg, arg.conj())))
    C1 = np.sqrt(1 / normfact)
    psi = np.multiply(np.dot(C1, (carrier - C2)), envelope)
    if halfkernel:
        condition = ((np.dot(xi[0], x) + np.dot(xi[1], y)) <= 0)
        mr = np.multiply(np.real(psi), condition)
        mi = np.multiply(np.imag(psi), condition)
    else:
        mr = np.real(psi)
        mi = np.imag(psi)

    return mr, mi


def endpoints(I, sigma, angleI, midpointI, debug=False):
    nr, nc = I.shape[:2]

    imcent = np.round(np.array((nr, nc)) / 2.0)
    d = imcent-midpointI
    # transform
    M = np.float32([[1, 0, d[1]], [0, 1, d[0]]])
    I = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]))
    I = rotate(I, -180*angleI/np.pi, preserve_range=True,
               mode='constant', order=0)

    freq = 1/sigma
    index = 0
    anglerange = [-np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3]
    Convs = np.zeros((nr, nc, len(anglerange)), dtype=complex)
    for langle in np.pi/2 + np.asarray(anglerange):
        mr, mi = cmorlet(sigma, freq, langle, 0)
        J = conv2(I, mr + np.dot(1j, mi), 'same')
        Convs[:, :, index] = J
        index = index+1

    hnc = math.floor(nc / 2)
    SS = np.zeros((nr, hnc))
    for i in np.arange(0, index):
        L = Convs[:, 0:hnc, i]
        R = Convs[:, -hnc:, len(anglerange)-1-i]
        R = np.fliplr(R)
        S = np.abs(L*np.conj(R))
        SS = np.maximum(SS, S)

    proximity = np.sum(SS)  # proximity between half images

    sort_idx = np.argsort(-SS, 1, kind='quicksort')
    SortS = np.take_along_axis(SS, sort_idx, axis=1)
    SortS = SortS[:, 0:10]
    s = np.sum(SortS, 1)
    l = 5
    k1 = np.block([np.ones((1, l)), -np.ones((1, l))]).flatten()
    s1 = conv(s, k1)

    lcs1, _ = find_peaks(np.maximum(s1, 0), height=0.05*np.max(np.abs(s1)),
                         distance=1, prominence=None, width=None, wlen=None, rel_height=0.5)
    if debug:
        plt.plot(np.maximum(s1, 0))
        plt.plot(lcs1, np.maximum(s1, 0)[lcs1], "x")
    lcs2, _ = find_peaks(np.maximum(-np.flipud(s1), 0), height=0.05*np.max(
        np.abs(s1)), distance=1, prominence=None, width=None, wlen=None, rel_height=0.5)
    if debug:
        plt.plot(np.maximum(-np.flipud(s1), 0))
        plt.plot(lcs2, np.maximum(-np.flipud(s1), 0)[lcs2], "x")
    i0 = lcs1[0]
    i1 = len(s) - lcs2[0] - 1

    if debug:
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1).plot(s, color="blue")
        plt.subplot(1, 2, 1).plot(s1, color="red")
        plt.subplot(1, 2, 1).plot(i0, s1[lcs1[0]], "x")
        plt.subplot(1, 2, 1).plot(i1, s1[len(s) - lcs2[0]-1], "x")
        plt.subplot(1, 2, 2).imshow(np.concatenate(
            (normalize(SS), SortS, I), axis=1), cmap="gray")

    ep0 = np.array([i0, round(nc / 2)])
    ep1 = np.array([i1, round(nc / 2)])
    R = np.array([[math.cos(angleI), - math.sin(angleI)],
                 [math.sin(angleI), math.cos(angleI)]])
    ep0 = np.round(np.dot(R, (ep0 - imcent).T) + imcent.T - d.T)
    ep1 = np.round(np.dot(R, (ep1 - imcent).T) + imcent.T - d.T)
    midpointI = np.round(np.dot(0.5, (ep0 + ep1))).T
    seglenI = np.linalg.norm(ep1 - ep0)

    # print(f"midpointI, seglenI, proximity: {midpointI}, {seglenI}, {proximity}")
    return midpointI, seglenI, proximity


def conv(x, y):
    npad = len(y) - 1
    x_padded = np.pad(x, (npad//2, npad - npad//2), mode='constant')
    return np.convolve(x_padded, y, 'valid')


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def normalize(I):
    J = I - np.min(I)
    J = J / np.max(J)
    return J


def imgradient(img):
    """
    imgradient MATLAB function equivalent
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # Find x and y gradients
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # Find magnitude and angle
    magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    return magnitude, angle
