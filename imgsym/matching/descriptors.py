import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper functions for MI-SIFT, MIFT, and MBR-SIFT
# -----------------------------------------------------------------------------

def mi_siftify(descriptors, do_mirror=True, do_invert=True, eps=1e-7):
    """
    Compute a fused SIFT descriptor from any subset of:
      - original
      - mirror-only (LR flip + orientation bin reversal)
      - invert-only (180 deg spatial rotation)
      - mirror+invert
    Returns: (N,128) float32 array of fused descriptors
    """
    N = descriptors.shape[0]
    d = descriptors.reshape(N, 4, 4, 8)

    f      = d
    f_flat = f.reshape(N, 128)

    d_spatial = f[:, :, ::-1, :]
    first_bin  = d_spatial[..., 0:1]
    other_bins = d_spatial[..., 1:]
    rev_other  = other_bins[..., ::-1]
    f_mirror   = np.concatenate([first_bin, rev_other], axis=-1).reshape(N, 128)

    f_invert = f[:, ::-1, ::-1, :].reshape(N, 128)

    f_mi = f_mirror.reshape(N, 4, 4, 8)
    f_mi = f_mi[:, ::-1, ::-1, :].reshape(N, 128)

    variants = [f_flat]
    if do_mirror: variants.append(f_mirror)
    if do_invert: variants.append(f_invert)
    if do_mirror and do_invert: variants.append(f_mi)

    V = np.stack(variants, axis=0).astype(np.float32)
    A =              V.sum(axis=0)
    B = np.sqrt(     (V**2).sum(axis=0)        )
    C =   ((V**3).sum(axis=0))**(1/3)
    D =   ((V**4).sum(axis=0))**(1/4)

    A4 = A.reshape(N,4,4,8)
    B4 = B.reshape(N,4,4,8)
    C4 = C.reshape(N,4,4,8)
    D4 = D.reshape(N,4,4,8)

    Atl = A4[:, 0:2,   0:2, :].reshape(N, -1)
    Btr = B4[:, 0:2,   2:4, :].reshape(N, -1)
    Cbl = C4[:, 2:4,   0:2, :].reshape(N, -1)
    Dbr = D4[:, 2:4,   2:4, :].reshape(N, -1)

    fmi = np.concatenate([Atl, Btr, Cbl, Dbr], axis=1)
    norms = np.linalg.norm(fmi, axis=1, keepdims=True)
    return (fmi / (norms + eps)).astype(np.float32)


def miftify(descriptors, tau=0.7):
    """
    Mirror-invariant SIFT: fuse each descriptor with its left-right mirrored copy,
    plus handle ambiguous cases via duplication.
    Returns: (M,128) fused descriptors, orig_idx array of length M
    """
    N = descriptors.shape[0]
    d = descriptors.reshape(N,4,4,8)

    L = d.sum(axis=(1,2))
    nd = L.argmax(axis=1)
    shifts = np.arange(1,4)
    idx_r = (nd[:,None] - shifts[None,:]) % 8
    idx_l = (nd[:,None] + shifts[None,:]) % 8
    mr = L[np.arange(N)[:,None], idx_r].sum(axis=1)
    ml = L[np.arange(N)[:,None], idx_l].sum(axis=1)

    flip_mask = mr > ml
    ambiguous = np.minimum(ml, mr) > tau * np.maximum(ml, mr)
    M = ambiguous.sum()

    flat   = d.reshape(N,128)
    d_spatial = d[:,:,::-1,:]
    first_bin = d_spatial[...,0:1]
    other     = d_spatial[...,1:]
    rev_other = other[...,::-1]
    d_flip    = np.concatenate([first_bin, rev_other], axis=-1).reshape(N,128)

    total = N + M
    out = np.empty((total,128), dtype=np.float32)
    orig_idx = np.empty(total, dtype=np.int32)

    out[:N]      = np.where(flip_mask[:,None], d_flip, flat)
    orig_idx[:N] = np.arange(N, dtype=np.int32)

    amb_i = np.nonzero(ambiguous)[0]
    out[N:]      = np.where(flip_mask[amb_i,None], flat[amb_i], d_flip[amb_i])
    orig_idx[N:] = amb_i

    return out, orig_idx


def mbr_siftify(descriptors, method='256bit', a=2.3, b=0.0):
    """
    Compute forward+mirror codebook for MBR-SIFT.
    Returns: codes_all (2N x C uint8), orig_idx_all (2N,)
    """
    descriptors = descriptors.astype(np.float32)
    N, D = descriptors.shape
    d = descriptors.reshape(N,4,4,8)
    r_sift = d.transpose(0,3,1,2).reshape(N,8*16)

    idx = np.arange(128)
    next_idx = (idx//16)*16 + (idx+1)%16
    AD = r_sift[:,next_idx] - r_sift[:,idx]

    if method=='128bit':
        codes = (AD >= 0).astype(np.uint8)
        C = 128
    else:
        mu = r_sift.mean(axis=1,keepdims=True)
        sigma = r_sift.std(axis=1,keepdims=True)
        T = a*sigma + b
        codes = np.zeros((N,256),dtype=np.uint8)
        gtT   = AD>=T
        ltNeg = AD<=-T
        gt0   = (AD>=0)&~gtT
        lt0   = (AD<0)&~ltNeg
        codes[:,0::2] = (gt0|gtT).astype(np.uint8)
        codes[:,1::2] = (lt0|gtT).astype(np.uint8)
        C = 256

    cells_per_dir = C//8
    block = codes.reshape(N,8,cells_per_dir)
    mirrored = 1 - block[:,:,::-1]
    dir_order = [0,7,6,5,4,3,2,1]
    codes_m = mirrored[:,dir_order,:].reshape(N,C)

    codes_all = np.vstack([codes, codes_m])
    orig_idx_all = np.concatenate([np.arange(N), np.arange(N)], axis=0).astype(np.int32)
    return codes_all, orig_idx_all


def imm_match_two_step_mbr(
    codes1_128, codes1_256,
    codes2_128, codes2_256,
    idx1, idx2,
    ratio_coarse=0.5,
    ratio_fine=0.84
):
    """Two-step MBR-SIFT matching: coarse 128-bit, fine 256-bit."""
    bf128 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw128 = bf128.knnMatch(codes1_128, codes2_128, k=5)
    matches = []
    for knn in raw128:
        if len(knn) < 2:
            continue
        q = knn[0].queryIdx
        d0, d1 = knn[0].distance, knn[1].distance
        n = 2 if d0 < ratio_coarse * d1 else 5
        trains = [m.trainIdx for m in knn[:n] if idx1[q] != idx2[m.trainIdx]]
        if len(trains) < 2:
            continue
        Q = codes1_256[q]
        T = codes2_256[trains]
        X = np.bitwise_xor(Q, T).reshape(len(trains),64,4)
        zero_cnt = (X.sum(axis=2)==0).sum(axis=1).astype(np.float32)
        D = np.arccos(zero_cnt/64.0)
        order = np.argsort(D)
        t1, t2 = trains[order[0]], trains[order[1]]
        d1f, d2f = float(D[order[0]]), float(D[order[1]])
        if d1f < ratio_fine * d2f:
            matches.append(cv2.DMatch(_queryIdx=q, _trainIdx=t1, _imgIdx=0, _distance=d1f))
    return matches

# -----------------------------------------------------------------------------
# Abstract base class
# -----------------------------------------------------------------------------

class DescriptorMatcher(ABC):
    def __init__(self, debug=False):
        self.debug = debug

    @abstractmethod
    def compute(self, image):
        """
        Detect keypoints, compute descriptors, match with flipped image.
        Returns: kp1, kp2, matches
        """
        pass

    def draw(self, image, kp1, kp2, matches, top_n=100):
        if not MATPLOTLIB_AVAILABLE:
            log.warning("matplotlib not available for visualization")
            return
        matches_sorted = sorted(matches, key=lambda m: m.distance)[:top_n]
        dbg = cv2.drawMatches(
            image, kp1,
            np.fliplr(image), kp2,
            matches_sorted, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(12,12))
        plt.imshow(dbg[..., ::-1])
        plt.axis('off')
        plt.show()

# -----------------------------------------------------------------------------
# Concrete matchers
# -----------------------------------------------------------------------------

class SIFTMatcher(DescriptorMatcher):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.sift = cv2.SIFT_create(enable_precise_upscale=False)
        self.bf   = cv2.BFMatcher()

    def compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.sift.detectAndCompute(gray, None)
        kp2, des2 = self.sift.detectAndCompute(np.fliplr(gray), None)
        if des1 is None or des2 is None:
            return kp1, kp2, []
        raw = self.bf.knnMatch(des1, des2, k=3)
        matches = [m for grp in raw for m in grp]
        if self.debug:
            log.debug(f"SIFT matches: {len(matches)}")
            self.draw(image, kp1, kp2, matches)
        return kp1, kp2, matches

class ORBMatcher(DescriptorMatcher):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.orb = cv2.ORB_create()
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.orb.detectAndCompute(gray, None)
        kp2, des2 = self.orb.detectAndCompute(np.fliplr(gray), None)
        if des1 is None or des2 is None:
            return kp1, kp2, []
        raw = self.bf.knnMatch(des1, des2, k=3)
        matches = [m for grp in raw for m in grp]
        if self.debug:
            log.debug(f"ORB matches: {len(matches)}")
            self.draw(image, kp1, kp2, matches)
        return kp1, kp2, matches


class MISIFTMatcher(DescriptorMatcher):
    def __init__(self, dist_ratio=0.75, debug=False):
        super().__init__(debug)
        self.dist_ratio = dist_ratio
        self.sift = cv2.SIFT_create()

    def compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp1, d1 = self.sift.detectAndCompute(gray, None)
        kp2, d2 = self.sift.detectAndCompute(np.fliplr(gray), None)

        m1 = mi_siftify(d1, do_mirror=True, do_invert=False)
        m2 = mi_siftify(d2, do_mirror=True, do_invert=False)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw = bf.knnMatch(m1, m2, k=2)
        matches = [m for m, n in raw if m.distance < self.dist_ratio * n.distance]

        if self.debug:
            log.debug(f"MI-SIFT matches: {len(matches)}")
            self.draw(image, kp1, kp2, matches)

        return kp1, kp2, matches


class MIFTMatcher(DescriptorMatcher):
    def __init__(self, dist_ratio=0.75, debug=False):
        super().__init__(debug)
        self.dist_ratio = dist_ratio
        self.sift = cv2.SIFT_create()

    def compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp1, d1 = self.sift.detectAndCompute(gray, None)
        kp2, d2 = self.sift.detectAndCompute(np.fliplr(gray), None)

        m1, orig1 = miftify(d1)
        m2, orig2 = miftify(d2)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw = bf.knnMatch(m1, m2, k=2)

        matches = []
        for m, n in raw:
            if (m.distance < self.dist_ratio * n.distance and
                orig1[m.queryIdx] != orig2[m.trainIdx]):
                matches.append(cv2.DMatch(
                    _queryIdx=orig1[m.queryIdx],
                    _trainIdx=orig2[m.trainIdx],
                    _imgIdx=m.imgIdx,
                    _distance=m.distance
                ))

        if self.debug:
            log.debug(f"MIFT matches: {len(matches)}")
            self.draw(image, kp1, kp2, matches)

        return kp1, kp2, matches


class MBRSIFTMatcher(DescriptorMatcher):
    def __init__(self, dist_ratio=0.75, debug=False):
        super().__init__(debug)
        self.dist_ratio = dist_ratio
        self.sift = cv2.SIFT_create()

    def compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp1, d1 = self.sift.detectAndCompute(gray, None)
        kp2, d2 = self.sift.detectAndCompute(np.fliplr(gray), None)

        c1_128, idx1 = mbr_siftify(d1, method='128bit')
        c2_128, idx2 = mbr_siftify(d2, method='128bit')
        c1_256, _    = mbr_siftify(d1, method='256bit')
        c2_256, _    = mbr_siftify(d2, method='256bit')

        raw = imm_match_two_step_mbr(
            c1_128, c1_256,
            c2_128, c2_256,
            idx1, idx2,
            ratio_coarse=0.5,
            ratio_fine=0.84
        )

        matches = []
        for m in raw:
            matches.append(cv2.DMatch(
                _queryIdx=idx1[m.queryIdx],
                _trainIdx=idx2[m.trainIdx],
                _imgIdx=m.imgIdx,
                _distance=m.distance
            ))

        if self.debug:
            log.debug(f"MBR-SIFT matches: {len(matches)}")
            self.draw(image, kp1, kp2, matches)

        return kp1, kp2, matches
