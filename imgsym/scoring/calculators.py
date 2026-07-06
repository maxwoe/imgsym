import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.fft import dctn
from skimage.metrics import structural_similarity as ssim
import math

# Try to import numba for optimization
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Base class for all symmetry calculators


class SymmetryCalculator:
    def calculate_score(self, bgr_image):
        """Calculate symmetry score from BGR image, returns float"""
        return self._calculate(bgr_image)

    def _calculate(self, bgr_image):
        """Override this in subclasses"""
        raise NotImplementedError

    def _to_grayscale(self, bgr_image):
        """Helper method for algorithms that need grayscale conversion"""
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


class PixelCorrelationCalculator(SymmetryCalculator):
    """
    Normalized cross-correlation between the grayscale image and its
    left-right mirror.

    References:
    - Mayer, S., & Landwehr, J. R. (2018). Quantifying visual aesthetics based
      on processing fluency theory: Four algorithmic measures for antecedents
      of aesthetic preferences. Psychology of Aesthetics, Creativity, and the
      Arts, 12(4), 399-431. https://doi.org/10.1037/aca0000187

    Their vertical-symmetry measure is the Pearson correlation of the left half
    against the mirrored right half. This calculator uses the closely related
    uncentered (cosine) form over the whole image,
    sum(I * flip(I)) / sqrt(sum(I^2) * sum(flip(I)^2)). The mean-centered,
    per-axis variant is available as
    SlidingWindowCalculator(sliding_method="pearson").
    """

    def _calculate(self, bgr_image):
        gray = self._to_grayscale(bgr_image)
        image = gray.astype(np.float32)
        flipped_image = np.fliplr(image)

        numerator = np.sum(image * flipped_image)
        denominator = np.sqrt(np.sum(image * image) *
                              np.sum(flipped_image * flipped_image))
        symmetry_score = numerator / denominator

        return symmetry_score


class SlidingWindowCalculator(SymmetryCalculator):
    """
    Axis-search measure: at a candidate column x, compare the left columns
    against the mirrored right columns with a configurable similarity metric
    (Pearson correlation, normalized cross-correlation, L2, SSIM, MSE, RMSE or
    mutual information), weighted by the overlap width, taking the peak over x
    as the symmetry axis.

    No single source. Scanning candidate axes and scoring each by the
    similarity of the two reflected halves (taking the peak as the axis) is a
    classical, generic technique, and the pluggable metric set makes this an
    in-house comparison harness rather than one paper's algorithm. Closest
    published relatives:
    - Gnutti, A., Guerrini, F., & Leonardi, R. (2017). Image symmetries: The
      right balance between evenness and perception. IWSSIP 2017. (And the 1D
      windowed normalized inner product in their 2021 IEEE TIP paper, Sec. IV,
      https://doi.org/10.1109/TIP.2021.3085202.) Same scan-window-for-peak
      structure, with a normalized inner product as the per-axis score.
    - Mayer, S., & Landwehr, J. R. (2018). Quantifying visual aesthetics based
      on processing fluency theory. Psychology of Aesthetics, Creativity, and
      the Arts, 12(4), 399-431. https://doi.org/10.1037/aca0000187 The
      "pearson" metric matches their Pearson correlation of the reflected
      halves. Note their measure itself already max-searches a small axis
      neighborhood (center +-1..5% of image width in 1% steps, 11 offsets);
      this harness generalizes that to any candidate column.
    """

    def __init__(self, sliding_method="pearson", x_position=None, debug=False):
        self.sliding_method = sliding_method
        self.x_position = x_position
        self.debug = debug

    def _calculate(self, bgr_image):
        gray = self._to_grayscale(bgr_image)
        h, w = gray.shape

        if self.x_position is None:
            x_position = w // 2
        else:
            x_position = self.x_position

        symmetry = self._calc_sliding_symmetry(gray, x_position)
        return symmetry[x_position] / h

    def _calc_sliding_symmetry(self, gray, x_position):
        h, w = gray.shape
        symmetry = np.zeros(w)

        x_positions = [x_position] if x_position is not None else np.arange(1, w)

        for x in x_positions:
            min_range = min(x, w - x)
            cols1 = gray[:, x - min_range:x]
            cols2 = np.fliplr(gray[:, x:x+min_range])

            if self.sliding_method == "pearson":
                r = np.corrcoef(cols1.flatten(), cols2.flatten())[0, 1]
            elif self.sliding_method == "nxc":
                r = cv2.matchTemplate(cols1.astype(np.float32), cols2.astype(
                    np.float32), method=cv2.TM_CCOEFF_NORMED)[0, 0]
            elif self.sliding_method == "l2":
                l2_distance = np.sqrt(np.sum((cols1 - cols2) ** 2))
                normalized_l2_distance = l2_distance / (min_range * h)
                r = 1 - normalized_l2_distance
            elif self.sliding_method == "ssim":
                r, _ = ssim(cols1, cols2, full=True)
            elif self.sliding_method == "mse":
                mse_value = np.mean((cols1 - cols2) ** 2)
                r = 1 - mse_value / np.max((cols1 ** 2, cols2 ** 2))
            elif self.sliding_method == "rmse":
                rmse_value = np.sqrt(np.mean((cols1 - cols2) ** 2))
                r = 1 - rmse_value / np.max((cols1 ** 2, cols2 ** 2))
            elif self.sliding_method == "mi":
                hist_2d, _, _ = np.histogram2d(cols1.flatten(), cols2.flatten(), bins=180)
                pxy = hist_2d / float(np.sum(hist_2d))
                px = np.sum(pxy, axis=1)
                py = np.sum(pxy, axis=0)
                px_py = px[:, None] * py[None, :]
                nzi = pxy > 0
                r = np.sum(pxy[nzi] * np.log(pxy[nzi] / px_py[nzi]))
            else:
                raise ValueError("Invalid method specified")

            symmetry[x] = r * (min_range / (w // 2))

        symmetry[np.isnan(symmetry)] = 0

        if self.debug:
            peak = np.argmax(symmetry)
            plt.figure(figsize=(10, 5))
            plt.plot(symmetry)
            plt.axvline(peak, color='r', linestyle='--', lw=0.5)
            plt.title('Symmetry Score')
            plt.xlabel('X Position')
            plt.ylabel('Score')
            plt.show()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(gray, cmap='gray')
            ax.axvline(peak, color='y', linestyle='-', lw=1)
            plt.show()

        return symmetry


class DCTCalculator(SymmetryCalculator):
    """
    DCT-based measure: ratio of the energy held in the even-indexed DCT
    coefficients (excluding the DC term) to the total AC energy. The energy of
    a reflection-symmetric signal concentrates in the even coefficients.

    References:
    - Gunlu, G., & Bilge, H. S. (2009). Symmetry analysis for 2D images by
      using DCT coefficients. IEEE, 2009. (Defines the even/total energy ratio
      as the symmetry measure implemented here.)
    - Kiryati, N., & Gofman, Y. (1998). Detecting symmetry in grey level
      images: The global optimization approach. International Journal of
      Computer Vision, 29(1), 29-45. https://doi.org/10.1023/A:1008034529558
      (Earlier spatial-domain symmetric/antisymmetric energy decomposition --
      ||f_s||^2 / ||f||^2 with f_s(x) = (f(x)+f(-x))/2, Gaussian-windowed --
      whose even/odd energy-ratio idea this DCT-domain measure realizes.)
    """

    def _calculate(self, bgr_image):
        gray = self._to_grayscale(bgr_image)

        dct_image = dctn(gray)

        N1, N2 = dct_image.shape
        even_indices_energy = np.sum(
            np.abs(dct_image[:, 0:N2:2]) ** 2) - np.abs(dct_image[0, 0]) ** 2

        total_energy = np.sum(np.abs(dct_image)**2) - np.abs(dct_image[0, 0])**2

        symmetry_measure = even_indices_energy / total_energy

        return symmetry_measure


class ErosCalculator(SymmetryCalculator):
    """
    EROS (Extraction of Robust Orientation using Symmetry): per-row even/odd
    intensity-profile decomposition about a candidate axis, with a local- and
    global-contrast correction. For each row profile and axis x, with
    even = sum|I(x+i) + I(x-i)| and odd = sum|I(x+i) - I(x-i)|, the score is
    s(x) = ((even - odd) * l) / ((even + odd) + g), where l is the row's local
    contrast and g a global-contrast floor; row scores are summed.

    References:
    - Smith, S. M., & Jenkinson, M. (1999). Accurate robust symmetry
      estimation. In MICCAI 1999, LNCS 1679, pp. 308-317. Springer.
      https://doi.org/10.1007/10704282_34

    Directly implements their Eqs. (1) and (2). The source specifies the
    corrections only qualitatively (l = "local contrast ... within the current
    perpendicular line", g = "a fraction of the global contrast"); this
    implementation instantiates l as the per-row intensity variance and
    g as 0.1x the global variance -- our design choice, not source values.
    """

    def __init__(self, x_position=None, num_rows_factor=1, debug=False):
        self.x_position = x_position
        self.num_rows_factor = num_rows_factor
        self.debug = debug

    def _calculate(self, bgr_image):
        gray = self._to_grayscale(bgr_image)
        h, w = gray.shape

        if self.x_position is None:
            x_position = w // 2
        else:
            x_position = self.x_position

        symmetry_profiles, _ = self._calc_eros_profiles(gray, x_position)
        score = np.sum(symmetry_profiles, axis=0)[x_position] / h
        return score

    def _calc_eros_profiles(self, gray, x_position):
        gray = gray.astype(np.float32)
        height, width = gray.shape
        symmetry_profiles = np.zeros((height, width))

        global_contrast = np.var(gray)
        local_contrasts = np.var(gray, axis=1)

        if self.num_rows_factor == 1:
            rows_to_process = np.arange(height)
        else:
            step = max(1, int(height / (height * self.num_rows_factor)))
            rows_to_process = np.arange(0, height, step)

        if x_position is not None:
            x_positions = [x_position]
        else:
            x_positions = np.arange(1, width - 1)

        for y in rows_to_process:
            intensity_profile = gray[y, :]
            l = local_contrasts[y]
            g = global_contrast * 0.1

            for x in x_positions:
                indices = np.arange(1, min(x, width - x - 1) + 1)
                even_sum = np.sum(
                    np.abs(intensity_profile[x + indices] + intensity_profile[x - indices]))
                odd_sum = np.sum(
                    np.abs(intensity_profile[x + indices] - intensity_profile[x - indices]))
                s = ((even_sum - odd_sum) * l) / ((even_sum + odd_sum) + g)
                s *= len(indices) / (width//2 -
                                     1) if width % 2 == 0 else len(indices) / (width//2)

                symmetry_profiles[y, x] = s

        if self.debug:
            cumulative_symmetry_profile = np.sum(symmetry_profiles, axis=0)
            peak_cumulative = np.argmax(cumulative_symmetry_profile)
            plt.figure(figsize=(10, 5))
            plt.plot(cumulative_symmetry_profile)
            plt.axvline(peak_cumulative, color='r', linestyle='--', lw=0.5)
            plt.title('Cumulative Symmetry Profile')
            plt.xlabel('X Position')
            plt.ylabel('Cumulative Symmetry Score')
            plt.show()

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(gray, cmap='gray')

            peak_positions = []
            for y in rows_to_process:
                symmetry_profile = symmetry_profiles[y]
                peak_index = np.argmax(symmetry_profile)
                peak_positions.append(peak_index)
                ax.axhline(y, color='b', linestyle='--', lw=0.5)
                ax.plot(peak_index, y, 'go')

            median_peak_position = int(np.median(peak_positions))

            ax.axvline(peak_cumulative, color='y', linestyle='-', lw=1)
            ax.axvline(median_peak_position, color='r', linestyle='--', lw=1)
            plt.show()

        return symmetry_profiles, rows_to_process


class GradientCalculator(SymmetryCalculator):
    """
    Gradient appearance + orientation measure. Sobel magnitude is normalized
    and thresholded; gradient angles of mirrored pixel pairs are combined as
    the direction-symmetry term DS = |angle + fliplr(angle)|, which equals 180
    for a perfect specular (mirror) pair. A direction factor F_D peaks at
    DS = 180 and falls off linearly within an angular window down to a base
    floor, weighting the magnitudes; per-row scores accumulate the weighted
    appearance correlation against the mirror.

    References:
    - Gnutti, A., Guerrini, F., & Leonardi, R. (2021). Combining appearance and
      gradient information for image symmetry detection. IEEE Transactions on
      Image Processing, 30, 5708-5723. https://doi.org/10.1109/TIP.2021.3085202

    The DS / F_D weighting is the "joint specularity of gradient orientation
    and high gradient magnitude" of Gnutti et al. (gradient directions at
    mirrored locations should be specular w.r.t. the axis). This is a
    simplified, global per-row variant rather than their candidate-axis
    selection-and-validation pipeline.
    """

    def __init__(self, base=0.1, alpha_window=90, debug=False):
        self.base = base
        self.alpha_window = alpha_window
        self.debug = debug

    def _calc_mag_ang(self, image, method="sobel", angle_in_degrees=False, angle_zero_to_pi=False, undefined=0):
        if method == "sobel":
            dx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
            dy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        elif method == "prewitt":
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            dx = cv2.filter2D(image, cv2.CV_64F, kernelx)
            dy = cv2.filter2D(image, cv2.CV_64F, kernely)
        elif method == "central":
            dx = np.zeros_like(image, dtype=np.float64)
            dy = np.zeros_like(image, dtype=np.float64)
            dx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
            dy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
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

    def _calculate(self, bgr_image):
        gray = self._to_grayscale(bgr_image)
        mag, ang = self._calc_mag_ang(gray, method="sobel", angle_in_degrees=True, angle_zero_to_pi=True, undefined=0)

        mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        mag[mag < 0.1] = 0

        ang *= 180 / np.pi
        ang = ang % 180

        G_dir = np.abs(ang + np.fliplr(ang))
        mask = (G_dir >= 180 - self.alpha_window) & (G_dir <= 180 + self.alpha_window)

        F_D = np.zeros_like(ang, dtype=np.float32)
        F_D[mask] = (1 - np.abs(G_dir[mask] - 180) / self.alpha_window) * (1 - self.base)
        F_D[~mask] = 0
        F_D += self.base
        F_GD = mag*F_D

        mask = np.nansum(mag, axis=1) > 0
        mag_x_dir = mag[mask, :] * F_D[mask, :]

        score = np.nansum(mag_x_dir * np.fliplr(mag_x_dir), axis=1) / \
            np.nansum(mag_x_dir ** 2, axis=1)
        score = np.nansum(score) / np.sqrt(bgr_image.shape[0])

        if self.debug:
            mag_uint8 = cv2.normalize(
                mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            F_GD = cv2.normalize(F_GD, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            plt.figure(figsize=(22, 12))
            plt.subplot(1, 6, 1).imshow(bgr_image[..., ::-1])
            plt.subplot(1, 6, 2).imshow(cv2.bitwise_not(mag_uint8), cmap='gray')
            plt.subplot(1, 6, 3).imshow(ang, cmap="jet")
            plt.subplot(1, 6, 4).imshow(G_dir, cmap="jet")
            plt.subplot(1, 6, 5).imshow(F_D, cmap="jet")
            plt.subplot(1, 6, 6).imshow(cv2.bitwise_not(F_GD), cmap="gray")
            plt.show()

        return score


class MultiScaleGradientCalculator(SymmetryCalculator):
    """
    Multi-scale gradient + mirror-aware orientation agreement.
    Returns a score in [0,1], higher = more symmetric.

    Combines Sobel gradients at two scales (3x3, 5x5) with texture-adaptive
    weights, then compares the gradient field of the image against its mirror
    (x-gradient sign flips, y-gradient is preserved) via a cosine orientation
    agreement plus a normalized magnitude difference, aggregated through an
    edge-weighted, bilateral-smoothed loss map.

    No direct source paper: this is an in-house extension that builds on the
    gradient-specularity idea of GradientCalculator (Gnutti, Guerrini &
    Leonardi, 2021, https://doi.org/10.1109/TIP.2021.3085202) by adding
    multi-scale fusion and an energy-normalized loss-to-score mapping. Treat
    as original work for attribution purposes.
    """

    def _calculate(self, bgr_image, beta: float = 0.8, use_energy_norm: bool = True, tau: float = 0.1):
        eps = 1e-6

        gray = self._to_grayscale(bgr_image).astype(np.float32) / 255.0
        flipped = np.fliplr(gray)

        # Multi-scale gradients (3x3 and 5x5)
        gx1 = cv2.Sobel(gray,   cv2.CV_32F, 1, 0, ksize=3)
        gy1 = cv2.Sobel(gray,   cv2.CV_32F, 0, 1, ksize=3)
        gx2 = cv2.Sobel(gray,   cv2.CV_32F, 1, 0, ksize=5)
        gy2 = cv2.Sobel(gray,   cv2.CV_32F, 0, 1, ksize=5)

        gx1_f = cv2.Sobel(flipped, cv2.CV_32F, 1, 0, ksize=3)
        gy1_f = cv2.Sobel(flipped, cv2.CV_32F, 0, 1, ksize=3)
        gx2_f = cv2.Sobel(flipped, cv2.CV_32F, 1, 0, ksize=5)
        gy2_f = cv2.Sobel(flipped, cv2.CV_32F, 0, 1, ksize=5)

        # Texture-adaptive weighting between scales
        texture = cv2.Laplacian(gray, cv2.CV_32F)
        texture_strength = float(np.mean(np.abs(texture)))
        texture_factor = min(texture_strength, 1.0)
        w1 = 0.8 - 0.2 * texture_factor
        w2 = 0.2 + 0.2 * texture_factor

        # Combine vector fields across scales
        gx = w1 * gx1 + w2 * gx2
        gy = w1 * gy1 + w2 * gy2
        gx_f = w1 * gx1_f + w2 * gx2_f
        gy_f = w1 * gy1_f + w2 * gy2_f

        mag = np.sqrt(gx*gx + gy*gy)
        mag_f = np.sqrt(gx_f*gx_f + gy_f*gy_f)

        # Orientation agreement across a mirror
        # For a perfect mirror, x-gradient flips sign; y-gradient keeps sign.
        dot = gx * (-gx_f) + gy * gy_f
        denom = (mag * mag_f) + eps
        cos_sim = np.clip(dot / denom, -1.0, 1.0)
        agree = 0.5 * (cos_sim + 1.0)

        # Magnitude symmetry (normalized difference)
        mag_diff = np.abs(mag - mag_f) / (mag + mag_f + eps)

        # Combine into a loss map
        edge_weight = (mag + mag_f) * 0.5
        edge_weight = edge_weight / (np.mean(edge_weight) + eps)
        edge_weight = np.clip(edge_weight, 0.0, 1.0)

        loss_map = edge_weight * (beta * mag_diff + (1.0 - beta) * (1.0 - agree))
        loss_map = loss_map.astype(np.float32)

        # Content-adaptive smoothing
        loss_smooth = cv2.bilateralFilter(loss_map, 5, 0.1, 1.0)

        # Aggregate loss
        m = float(np.mean(loss_smooth))

        # Map loss to score
        if use_energy_norm:
            e = float(np.mean(mag) + np.mean(mag_f)) + eps
            score = 1.0 - m / (m + e)
        else:
            score = float(np.exp(-m / max(tau, eps)))

        return float(score)


class HOGCalculator(SymmetryCalculator):
    """
    HOG-based measure: cosine similarity between the HOG descriptor of the
    image and the HOG descriptor of its mirror.

    References:
    - Renero-C, F.-J., Romero-H, R.-A., & Peregrina-B, H. (2017). Extracting
      the symmetry of the human face from digital photographs. Bio-Algorithms
      and Med-Systems, 13(2), 103-109. https://doi.org/10.1515/bams-2017-0002
      Source of the HOG symmetry measure: their Eq. (6),
      S_PH = (1/N) * sum_i M_i . N_i^R, the inner product of a HOG descriptor
      with the reflected descriptor of the mirrored side.
    - Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for
      human detection. CVPR 2005, pp. 886-893.
      https://doi.org/10.1109/CVPR.2005.177 (the HOG descriptor itself.)

    Differs from the source by computing a single cosine-normalized descriptor
    over the whole image, where the source averages per-region inner products
    (landmark patches for S_PH; whole-face vertical strips for their
    best-performing S_SH/S_HE variants).

    Defaults (cell_size=16, nbins=18, signed_gradient=False) were chosen by a
    parameter ablation for symmetry discrimination: skill peaks at a mid cell
    size (4 < 8 < 16 > 32) and unsigned gradients beat signed -- a mirror
    preserves unsigned orientation but flips signed gradient direction. Block
    size and stride scale with the cell (2x2 cells per block, 1-cell stride).
    """

    def __init__(self, cell_size=16, nbins=18, signed_gradient=False,
                 win_size=64, debug=False):
        self.cell_size = cell_size
        self.nbins = nbins
        self.signed_gradient = signed_gradient
        self.win_size = win_size
        self.debug = debug

    def _calculate(self, bgr_image):
        half_1, half_2 = bgr_image, np.fliplr(bgr_image)

        hog_1 = self._calc_hog(half_1)
        hog_2 = self._calc_hog(half_2)
        score = np.dot(hog_1, hog_2) / \
            (np.linalg.norm(hog_1) * np.linalg.norm(hog_2))

        return score

    def _calc_hog(self, img):
        h, w = img.shape[:2]
        win, cell = self.win_size, self.cell_size
        if w < win or h < win:
            # Upscale crops smaller than the HOG window. The image and its mirror
            # are scaled identically, so the descriptor comparison stays valid.
            s = max(win / w, win / h)
            img = cv2.resize(img, (max(win, int(round(w * s))), max(win, int(round(h * s)))),
                             interpolation=cv2.INTER_LINEAR)
        hog = cv2.HOGDescriptor(_winSize=(win, win), _blockSize=(2 * cell, 2 * cell),
                                _blockStride=(cell, cell), _cellSize=(cell, cell),
                                _nbins=self.nbins, _signedGradient=self.signed_gradient,
                                _gammaCorrection=True)
        feature_descriptior = hog.compute(img, locations=None)
        return feature_descriptior


class PHOGCalculator(SymmetryCalculator):
    """
    PHOG-based measure: cosine similarity between the Pyramid HOG descriptor of
    the image and that of its mirror. Extends HOGCalculator with a spatial
    pyramid (Canny edges + Sobel orientation, levels 0..pyramid_height).

    References:
    - Renero-C, F.-J., Romero-H, R.-A., & Peregrina-B, H. (2017). Extracting
      the symmetry of the human face from digital photographs. Bio-Algorithms
      and Med-Systems, 13(2), 103-109. https://doi.org/10.1515/bams-2017-0002
      (HOG-descriptor-vs-reflected-descriptor symmetry measure; see Eq. (6).)
    - Bosch, A., Zisserman, A., & Munoz, X. (2007). Representing shape with a
      spatial pyramid kernel. ACM CIVR 2007, pp. 401-408.
      https://doi.org/10.1145/1282280.1282340 (the PHOG descriptor itself.)

    The pyramid (PHOG) extension of the Renero-C et al. HOG symmetry measure is
    an in-house combination; PHOG self-similarity for aesthetic/symmetry
    scoring also appears in the Redies group's work.
    """

    def __init__(self, nbins=18, pyramid_height=3, debug=False):
        self.nbins = nbins
        self.pyramid_height = pyramid_height
        self.debug = debug

    def _calculate(self, bgr_image):
        half1, half2 = bgr_image, np.fliplr(bgr_image)

        d1 = self._compute_phog(half1)
        d2 = self._compute_phog(half2)

        score = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
        if self.debug:
            print(f"PHOG1 norm: {np.linalg.norm(d1):.2f}, PHOG2 norm: {np.linalg.norm(d2):.2f}, score: {score:.2f}")
        return score

    def _compute_phog(self, img):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        edges = cv2.Canny(cv2.blur(gray, (3, 3)), 0.66*mean, 1.33*mean)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.abs(gx) + np.abs(gy)
        ori = cv2.phase(gx, gy, angleInDegrees=True)
        ori = (ori / (360.0/self.nbins)).astype(np.float32)

        h, w = gray.shape
        desc = []
        for level in range(self.pyramid_height+1):
            cells = 2**level
            cell_h = h // cells
            cell_w = w // cells
            for i in range(cells):
                for j in range(cells):
                    sx, sy = i*cell_h, j*cell_w
                    hist = self._get_histogram(edges, ori, mag, sx, sy,
                                               cell_w if j < cells-1 else w - sy,
                                               cell_h if i < cells-1 else h - sx)
                    desc.append(hist)
        desc = np.hstack(desc).astype(np.float32)
        s = desc.sum()
        return desc/s if s > 0 else desc

    def _get_histogram(self, edges, ors, mag, startX, startY, width, height):
        if NUMBA_AVAILABLE:
            return _get_histogram_numba(edges, ors, mag, startX, startY, width, height, self.nbins)
        else:
            hist = np.zeros(self.nbins, dtype=np.float32)
            for x in range(startX, startX + height):
                for y in range(startY, startY + width):
                    if edges[x, y] > 0:
                        b = int(math.floor(ors[x, y])) % self.nbins
                        hist[b] += mag[x, y]
            return hist


@jit(nopython=True)
def _get_histogram_numba(edges, ors, mag, startX, startY, width, height, nbins):
    hist = np.zeros(nbins, dtype=np.float32)
    for x in range(startX, startX + height):
        for y in range(startY, startY + width):
            if edges[x, y] > 0:
                b = int(np.floor(ors[x, y])) % nbins
                hist[b] += mag[x, y]
    return hist


class GaborCalculator(SymmetryCalculator):
    """
    Gabor-filter measure: a bank of Gabor filters over orientations 0..165 in
    15-degree steps (12 directions, the paper's stated span; equivalent to
    range(0, 180, 15)) produces magnitude responses that are max-normalized and
    thresholded (tau = 0.2, the paper's selected value) into binary partial
    images. Each orientation alpha is paired with 180 - alpha (0 and 90 are
    self-paired), and block (m x m, m = width/8 per the paper) energies are
    compared between a partial image and the mirrored partial of its paired
    orientation. The asymmetry is the mean normalized energy difference;
    symmetry = 1 - asymmetry (the final clamp to >= 0 is ours). The source
    defines the filter bank only symbolically (its Eq. 1 leaves sigma_x,
    sigma_y and the center frequency unstated; filtering is done via FFT);
    the kernel parameterization here (ksize=21, sigma=3.0, lambd=5.0,
    gamma=0.5, quadrature-pair magnitude) is our instantiation.

    References:
    - Shaker, F., & Monadjemi, A. H. (2015). A new symmetry measure based on
      Gabor filters. 2015 23rd Iranian Conference on Electrical Engineering
      (ICEE), p. 705. IEEE.
    """

    def __init__(self, orientations=None, tau=0.2, m_fraction=1/8, ksize=21, sigma=3.0, lambd=5.0, gamma=0.5, psi=0):
        self.orientations = orientations if orientations is not None else list(range(0, 180, 15))
        self.tau = tau
        self.m_fraction = m_fraction
        self.ksize = ksize
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        self.filters = self._create_gabor_filters()

    def _calculate(self, bgr_image):
        gray = self._to_grayscale(bgr_image)
        partials = self._apply_gabor_filters(gray)
        bin_partials = self._normalize_and_threshold(partials)

        pairs = []
        used_indices = set()
        for i, alpha in enumerate(self.orientations):
            if i in used_indices:
                continue
            comp_angle = 180 - alpha
            if comp_angle in self.orientations:
                j = self.orientations.index(comp_angle)
                pairs.append((i, j))
                used_indices.add(i)
                used_indices.add(j)
            else:
                pairs.append((i, i))
                used_indices.add(i)
        h, w = gray.shape
        m = int(w * self.m_fraction)
        m = max(m, 1)
        asym_values = []
        for idxA, idxB in pairs:
            imgA = bin_partials[idxA]
            imgB = bin_partials[idxB][:, ::-1]
            energiesAB = self._compute_paired_energy_same_location(imgA, imgB, m)
            numerator = 0.0
            denominator = 0.0
            E_imgA = np.sum(imgA**2) / (imgA.shape[0]*imgA.shape[1])
            E_imgB = np.sum(imgB**2) / (imgB.shape[0]*imgB.shape[1])
            norm_factor = E_imgA + E_imgB
            for E_A, E_B in energiesAB:
                numerator += abs(E_A - E_B)
                denominator += (E_A + E_B)
            if norm_factor == 0 or not energiesAB:
                asym_d = 0
            else:
                asym_d = (numerator / len(energiesAB)) / norm_factor
            asym_values.append(asym_d)
        average_asym = np.mean(asym_values) if asym_values else 0
        symmetry = 1 - average_asym
        symmetry = max(symmetry, 0)
        return symmetry

    def _create_gabor_filters(self):
        filters = []
        for theta in self.orientations:
            theta_rad = theta * np.pi / 180.0
            real = cv2.getGaborKernel((self.ksize, self.ksize), self.sigma, theta_rad,
                                      self.lambd, self.gamma, self.psi, ktype=cv2.CV_64F)
            imag = cv2.getGaborKernel((self.ksize, self.ksize), self.sigma, theta_rad,
                                      self.lambd, self.gamma, self.psi + np.pi/2, ktype=cv2.CV_64F)
            filters.append((real, imag))
        return filters

    def _apply_gabor_filters(self, image):
        partial_images = []
        for real, imag in self.filters:
            real_response = cv2.filter2D(image, cv2.CV_64F, real)
            imag_response = cv2.filter2D(image, cv2.CV_64F, imag)
            magnitude = np.sqrt(real_response**2 + imag_response**2)
            partial_images.append(magnitude)
        return partial_images

    def _normalize_and_threshold(self, partial_images):
        binary_images = []
        for img in partial_images:
            img_norm = img / np.max(img) if np.max(img) != 0 else img
            _, img_thresh = cv2.threshold(img_norm, self.tau, 1, cv2.THRESH_BINARY)
            binary_images.append(img_thresh)
        return binary_images

    def _compute_paired_energy_same_location(self, imgA, imgB, m):
        h, w = imgA.shape
        num_squares_y = h // m
        num_squares_x = w // m
        energies = []
        for iy in range(num_squares_y):
            for ix in range(num_squares_x):
                patchA = imgA[iy*m:(iy+1)*m, ix*m:(ix+1)*m]
                patchB = imgB[iy*m:(iy+1)*m, ix*m:(ix+1)*m]
                E_A = np.sum(patchA**2) / (m * m)
                E_B = np.sum(patchB**2) / (m * m)
                energies.append((E_A, E_B))
        return energies


class AlexNetCalculator(SymmetryCalculator):
    """
    AlexNet-based symmetry calculation using CNN features, after Brachmann &
    Redies: filter responses of a pretrained AlexNet layer are adaptive-max-
    pooled to a patches x patches grid for the image and its mirror and
    compared as score = 1 - sum|M1 - M2| / sum max(M1, M2) (their Eqs. 1-2).

    POLICY: reimplemented methods run at the source's recommended configuration
    at its reported optimum. Brachmann & Redies evaluate conv1..conv5 and
    recommend SECOND-layer features for arbitrary images (conv5 fit their
    CD-cover ratings best, r=0.90, but is called out as training-data
    specific); their reported conv2 optimum is an 11x11 patch grid (Spearman
    0.85). Defaults are therefore layer=2, patches=11. Their layer-1 optimum
    (layer=1, patches=17, r=0.80 in their study) is retained for the
    layer-robustness check (results/alexnet_layer_check.md);
    DeepFeatureCalculator generalizes the layer/backbone choice.

    References:
    - Brachmann, A., & Redies, C. (2016). Using convolutional neural network
      filters to measure left-right mirror symmetry in images. Symmetry,
      8(12), 144. https://doi.org/10.3390/sym8120144

    Implementation notes: weights are torchvision's AlexNet (the source used
    the closely related CaffeNet, whose LRN layers torchvision omits); the
    ReLU rectification standard to AlexNet is applied (the source does not
    name the nonlinearity).
    """

    def __init__(self, patches=11, layer=2, target_size=512):
        try:
            import torch
            import torch.nn.functional as F
            from torchvision import models
            from torchvision.models.alexnet import AlexNet_Weights

            self.torch = torch
            self.F = F
            self.patches = patches
            self.layer = layer
            self.target_size = target_size

            model = models.alexnet(weights=AlexNet_Weights.DEFAULT).eval()
            # feature-stack cut points: conv1+relu / conv1..conv2+relu
            cut = {1: 2, 2: 5}[layer]
            self.stack = model.features[:cut]

        except ImportError:
            raise ImportError("PyTorch and torchvision are required for AlexNet calculator. "
                              "Install with: pip install imgsym[deep]")

    def _calculate(self, bgr_image):
        flipped_image = np.fliplr(bgr_image)
        return self._compare_feature_maps(bgr_image, flipped_image)

    def _compare_feature_maps(self, img1, img2):
        feat1 = self._get_features(img1)
        feat2 = self._get_features(img2)
        map1, _ = self._get_max_maps(feat1)
        map2, _ = self._get_max_maps(feat2)
        return self._feature_symmetry_score(map1, map2)

    def _preprocess(self, bgr_image):
        rgb = bgr_image[..., ::-1]
        rgb = np.ascontiguousarray(rgb)
        t = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = self.F.interpolate(t, size=(self.target_size, self.target_size),
                               mode='bilinear', align_corners=False)
        mean = self.torch.tensor([0.485, 0.456, 0.406], device=t.device)[None, :, None, None]
        std = self.torch.tensor([0.229, 0.224, 0.225], device=t.device)[None, :, None, None]
        return (t - mean) / std

    def _get_features(self, bgr_image):
        x = self._preprocess(bgr_image)
        with self.torch.no_grad():
            feat = self.stack(x)
        return feat

    def _get_max_maps(self, feature):
        with self.torch.no_grad():
            mp = self.F.adaptive_max_pool2d(feature, (self.patches, self.patches)).squeeze(0)
        mp_np = mp.cpu().numpy().transpose(1, 2, 0)
        sums = mp_np.sum(axis=2, keepdims=True)
        return mp_np, mp_np / (sums + 1e-8)

    def _feature_symmetry_score(self, map1, map2):
        sa = np.abs(map1 - map2).sum()
        sm = np.maximum(map1, map2).sum()
        return float(1.0 - sa/(sm + 1e-8))


class DeepFeatureCalculator(SymmetryCalculator):
    """
    Deep learning feature-based symmetry calculation using any timm model.
    Supports Vision Transformers, ConvNeXt, MambaOut, ResNet, EfficientNet and
    other architectures.

    Extracts a feature map (``feature_stage``) of a pretrained backbone for the
    image and its mirror and scores them as 1 - sum|F1 - F2| / sum max(F1, F2).
    The score is a global elementwise reduction over corresponding feature
    elements, so a backbone's channels-last feature layout does not affect the
    result. ``feature_stage`` defaults to 1 (a low/mid-level stage): mirror
    symmetry is a low-level geometric property, and low/mid features discriminate
    it markedly better than the deepest semantic map, which is spatially coarse
    and invariant to the detail symmetry depends on (the last stage was the
    weakest of MambaOut's four in a per-stage benchmark).

    The default backbone is ``mambaout_base`` (the ``mambaout_base.in1k``
    weights, 224 px): MambaOut is a gated-convolution, ConvNeXt-like network
    with no state-space (Mamba) token mixer, chosen for its strong orientation
    sensitivity in the author's rotation-estimation study.

    References:
    - Brachmann, A., & Redies, C. (2016). Using convolutional neural network
      filters to measure left-right mirror symmetry in images. Symmetry,
      8(12), 144. https://doi.org/10.3390/sym8120144 (The CNN-filter symmetry
      measure this generalizes -- see AlexNetCalculator -- from AlexNet conv1 to
      arbitrary modern timm backbones; the backbone generalization is in-house.)
    - Yu, W., & Wang, X. (2024). MambaOut: Do We Really Need Mamba for Vision?
      arXiv:2405.07992. (The default backbone architecture.)
    - Woehrer, M. (2026). Image Rotation Angle Estimation: Comparing
      Circular-Aware Methods. arXiv:2603.25351. (Motivates the MambaOut Base
      default; ~1.24 deg mean absolute error on rotation estimation.)
    """

    def __init__(self, model_name='mambaout_base', feature_stage=1):
        try:
            import torch
            import timm
            import timm.data

            self.torch = torch
            self.feature_stage = feature_stage

            self.model = timm.create_model(model_name, pretrained=True, features_only=True)
            self.model.eval()

            temp_model = timm.create_model(model_name, pretrained=False)
            data_config = timm.data.resolve_data_config({}, model=temp_model)
            del temp_model

            self.input_size = data_config.get('input_size', (3, 224, 224))
            self.mean = data_config.get('mean', (0.485, 0.456, 0.406))
            self.std = data_config.get('std', (0.229, 0.224, 0.225))

        except ImportError:
            raise ImportError("PyTorch and timm are required for DeepFeatureCalculator. "
                              "Install with: pip install imgsym[deep]")

    def _calculate(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        flipped_image = np.fliplr(rgb_image)
        return self._compare_features(rgb_image, flipped_image)

    def _compare_features(self, img1, img2):
        feat1 = self._extract_features(img1)
        feat2 = self._extract_features(img2)
        return self._get_differences(feat1, feat2)

    def _preprocess_image(self, rgb_image):
        target_h, target_w = self.input_size[1], self.input_size[2]
        resized = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_float = resized.astype(np.float32) / 255.0
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        normalized = (img_float - mean) / std
        img_tensor = self.torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return img_tensor

    def _extract_features(self, rgb_image):
        img_tensor = self._preprocess_image(rgb_image)
        with self.torch.no_grad():
            feature_list = self.model(img_tensor)
        idx = self.feature_stage
        if idx >= len(feature_list):
            idx = len(feature_list) - 1
        features = feature_list[idx]
        # The final score is a global elementwise reduction over the whole tensor
        # (see _get_differences), so the axis order is irrelevant -- this permute
        # is a no-op for the score and the channels-last layout of some backbones
        # (e.g. MambaOut) does not affect the result.
        features = features.squeeze(0)
        features = features.permute(1, 2, 0)
        return features.abs().cpu().numpy()

    def _get_differences(self, feat_orig, feat_flip):
        assert feat_orig.shape == feat_flip.shape
        sum_abs = np.sum(np.abs(feat_orig - feat_flip))
        sum_max = np.sum(np.maximum(feat_orig, feat_flip))
        return float(1.0 - (sum_abs / (sum_max + 1e-8)))


class WeightedBinarySymmetryCalculator(SymmetryCalculator):
    """
    Weighted binary symmetry calculator. The image is Otsu-binarized, then
    horizontal, vertical and (for square images) main- and minor-diagonal
    symmetries are measured by counting matching mirrored foreground pixels
    with a distance weight w = 1 + (j - 1) / (n - 1) that gives pixels nearer
    the axis more influence; each axis score is normalized by 2 / (3 m n) and
    the mean over the axes is scaled to a 0-100 percentage.

    References:
    - Bauerly, M., & Liu, Y. (2006). Computational modeling and experimental
      investigation of effects of compositional elements on interface and
      design aesthetics. International Journal of Human-Computer Studies,
      64(8), 670-682. https://doi.org/10.1016/j.ijhcs.2006.01.002
      Originator of the distance-weighted reflectional symmetry formula.
    - Gartus, A., & Leder, H. (2017). Predicting perceived visual complexity of
      abstract patterns using computational measures: The influence of mirror
      symmetry on complexity perception. PLoS ONE, 12(11), e0185276.
      https://doi.org/10.1371/journal.pone.0185276
      States the exact measure implemented here: their Eq. (1), the four-axis
      mean (horizontal, vertical, both diagonals) scaled by 100.
    - Hubner, R., & Fillinger, M. G. (2016). Comparison of objective measures
      for predicting perceptual balance and visual aesthetic preference.
      Frontiers in Psychology, 7, 335.
      https://doi.org/10.3389/fpsyg.2016.00335
      Same measure, attributed in-text to Bauerly and Liu (2006).

    The Otsu binarization step is an addition not present in the cited sources;
    matches are counted only where both mirrored pixels are foreground.

    The ``axes`` parameter selects which reflection axes enter the score:
    ``"all"`` (default) reproduces the cited horizontal/vertical(/diagonal) mean;
    ``"v"`` returns the vertical (left-right) component only -- the apples-to-
    apples choice when benchmarking left-right mirror symmetry against the other
    calculators (it also removes the square-only diagonal dependence); ``"h"``
    returns the horizontal (top-bottom) component only. The benchmark always uses
    ``"v"`` because extraction canonicalizes the target axis to vertical.
    """

    def __init__(self, axes="all"):
        if axes not in ("all", "v", "h"):
            raise ValueError("axes must be 'all', 'v' or 'h'")
        self.axes = axes

    def _calculate(self, bgr_image):
        gray = self._to_grayscale(bgr_image)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        H, W = gray.shape

        _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        bw = bw.astype(np.uint8)

        Sv = self._calculate_vertical_symmetry(bw, H, W)
        if self.axes == "v":
            return float(Sv * 100.0)

        Sh = self._calculate_horizontal_symmetry(bw, H, W)
        if self.axes == "h":
            return float(Sh * 100.0)

        if H == W and H > 1:
            Smd = self._calculate_major_diagonal_symmetry(bw, H)
            Sad = self._calculate_minor_diagonal_symmetry(bw, H)
            ms = float(((Sh + Sv + Smd + Sad) / 4.0) * 100.0)
        else:
            ms = float(((Sh + Sv) / 2.0) * 100.0)

        return ms

    def _calculate_horizontal_symmetry(self, bw, H, W):
        h2 = H // 2
        if h2:
            top = bw[:h2, :]
            bottom = bw[-h2:, :][::-1, :]
            n1 = h2 - 1
            w_row = (1.0 + (np.arange(h2) / n1)) if n1 > 0 else np.ones(h2)
            sym_h = (top * bottom * w_row[:, None]).sum(dtype=np.float64)
            return float(sym_h * (2.0 / (3.0 * W * h2)))
        else:
            return 0.0

    def _calculate_vertical_symmetry(self, bw, H, W):
        w2 = W // 2
        if w2:
            left = bw[:, :w2]
            right = bw[:, -w2:][:, ::-1]
            n1 = w2 - 1
            w_col = (1.0 + (np.arange(w2) / n1)) if n1 > 0 else np.ones(w2)
            sym_v = (left * right * w_col[None, :]).sum(dtype=np.float64)
            return float(sym_v * (2.0 / (3.0 * H * w2)))
        else:
            return 0.0

    def _calculate_major_diagonal_symmetry(self, bw, H):
        sym_md = 0.0
        for i in range(1, H):
            base = (bw[i, :i] * bw[:i, i]).astype(np.float64)
            weights = 1.0 + (np.arange(1, i + 1) / i)
            sym_md += (base * weights).sum()
        return float(sym_md * (4.0 / (3.0 * H * (H - 1))))

    def _calculate_minor_diagonal_symmetry(self, bw, H):
        bwh = cv2.flip(bw, 1)
        sym_ad = 0.0
        for i in range(1, H):
            base = (bwh[i, :i] * bwh[:i, i]).astype(np.float64)
            weights = 1.0 + (np.arange(1, i + 1) / i)
            sym_ad += (base * weights).sum()
        return float(sym_ad * (4.0 / (3.0 * H * (H - 1))))


class LocalGlobalSymmetryCalculator(SymmetryCalculator):
    """
    Local and global symmetry calculator using patch descriptors and nearest
    neighbor matching. Square patches sampled in mirrored pairs about a
    vertical candidate axis are turned into descriptors of [patch pixels,
    weighted (folded-x, y) position], z-score normalized over both sides; the
    symmetry cost is the mean bidirectional nearest-neighbor distance between
    the left and right descriptor sets, mapped to 1 / (1 + cost). The bounded
    1/(1+cost) mapping is ours -- the source reports the raw cost S directly
    (low S = symmetric); the mapping is monotone, so rank-based evaluation is
    unaffected.

    References:
    - Hogeweg, L., Sanchez, C. I., Maduskar, P., Philipsen, R. H. H. M., &
      van Ginneken, B. (2017). Fast and effective quantification of symmetry in
      medical images for pathology detection: Application to chest radiography.
      Medical Physics, 44(6), 2242-2256. https://doi.org/10.1002/mp.12127

    Implements their local/global symmetry measure: the folded position
    coordinate (their Eq. 4), the [f_img, w * f_pos] descriptor with position
    weight w, z-scored before weighting (their Eq. 6), and the bidirectional
    nearest-neighbor symmetry cost (their Eq. 7). Uses an exact sklearn
    brute-force nearest neighbor instead of the approximate kd-tree
    (Arya & Mount) of the original.

    POLICY: reimplemented methods run at the source's recommended configuration
    at its reported optimum. Defaults are therefore the source's grid-searched
    optimum m=15, w=17.7, at its kappa=16 subsampling (stride k=4 =
    sqrt(kappa); their kappa is a squared subsample factor). The source's
    full-sampling default (m=9, w=10, kappa=1) is intractable under the exact
    nearest-neighbor search used here. An earlier in-house configuration
    (m=31, w=31) scores identically on the single-axis benchmark (0.65 vs
    0.65; the source default at kappa=16 scores 0.63), and the failed global
    sanity check persists at every configuration
    (results/patchnn_param_check.md).
    """

    def __init__(self, m=15, k=4, w=17.7):
        if m % 2 == 0:
            raise ValueError("Patch size m must be odd.")
        self.m = int(m)
        self.h = self.m // 2
        self.k = int(k)
        self.w = float(w)

    def _calculate(self, bgr_image):
        gray = self._to_grayscale(bgr_image)
        gray_float = gray.astype(np.float32)
        if gray_float.max() > 1.0:
            gray_float /= 255.0

        H, W = gray_float.shape
        x_s = (W - 1) / 2.0

        return self._compute_symmetry(gray_float, x_s)

    def _compute_symmetry(self, image, x_s):
        self.image = image
        self.H, self.W = image.shape
        self.x_s = x_s

        P_L, P_R = self._paired_samples()
        if not P_L:
            return 0.0

        P_L_arr = np.asarray(P_L, dtype=np.int32)
        P_R_arr = np.asarray(P_R, dtype=np.int32)

        pL = self._extract_patches(P_L_arr, flip=False)
        pR = self._extract_patches(P_R_arr, flip=True)

        try:
            dist_L, dist_R = self._nn_distances(P_L_arr, P_R_arr, pL, pR)
        except ImportError:
            return self._simple_patch_symmetry(pL, pR)

        S_cost = 0.5 * (float(dist_L.mean()) + float(dist_R.mean()))
        return 1.0 / (1.0 + S_cost)

    def _nn_distances(self, P_L_arr, P_R_arr, pL, pR):
        """Per-patch bidirectional NN distances (the source's Eq. 7 terms):
        for every left descriptor its distance to the nearest right descriptor,
        and vice versa. Requires scikit-learn (exact brute-force NN)."""
        xL = P_L_arr[:, 0].astype(np.float32)
        yL = P_L_arr[:, 1].astype(np.float32)
        xR = P_R_arr[:, 0].astype(np.float32)
        yR = P_R_arr[:, 1].astype(np.float32)
        pos_L = np.stack([self._x_ell(xL), yL], axis=1)
        pos_R = np.stack([self._x_ell(xR), yR], axis=1)

        dL0 = np.hstack([pL, pos_L]).astype(np.float32)
        dR0 = np.hstack([pR, pos_R]).astype(np.float32)

        Nl, Nr = dL0.shape[0], dR0.shape[0]
        N = Nl + Nr
        sum_all = dL0.sum(axis=0, dtype=np.float64) + dR0.sum(axis=0, dtype=np.float64)
        mu = (sum_all / N).astype(np.float32)

        sumsq_all = (dL0.astype(np.float64) ** 2).sum(axis=0) + (dR0.astype(np.float64) ** 2).sum(axis=0)
        var = (sumsq_all / N) - (mu.astype(np.float64) ** 2)
        sd = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)

        dL = (dL0 - mu) / sd
        dR = (dR0 - mu) / sd

        dL[:, -2:] *= self.w
        dR[:, -2:] *= self.w

        from sklearn.neighbors import NearestNeighbors

        nn_R = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean").fit(dR)
        dist_L, _ = nn_R.kneighbors(dL, return_distance=True)
        nn_L = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean").fit(dL)
        dist_R, _ = nn_L.kneighbors(dR, return_distance=True)

        return dist_L.ravel().astype(np.float32), dist_R.ravel().astype(np.float32)

    def calculate_heatmap(self, bgr_image):
        """Per-patch symmetry heatmap about the vertical centerline.

        Returns ``(score, heatmap)``: ``score`` is exactly
        :meth:`calculate_score`, and ``heatmap`` is an (H, W) float32 map where
        every sampled m x m patch is splatted with its own ``1 / (1 + d)``
        (``d`` = the patch's mirror nearest-neighbor distance; overlapping
        patches average). Higher = that region mirror-matches the other side
        better; ``np.nan`` where no patch was sampled. Requires scikit-learn
        (there is no approximate fallback for the per-patch map).

        The exact NN search is O(n^2) in the number of sampled patches --
        downscale large inputs (<= ~600 px on the long side) first."""
        gray = self._to_grayscale(bgr_image)
        gray_float = gray.astype(np.float32)
        if gray_float.max() > 1.0:
            gray_float /= 255.0

        self.image = gray_float
        self.H, self.W = gray_float.shape
        self.x_s = (self.W - 1) / 2.0

        heat = np.full((self.H, self.W), np.nan, dtype=np.float32)
        P_L, P_R = self._paired_samples()
        if not P_L:
            return 0.0, heat

        P_L_arr = np.asarray(P_L, dtype=np.int32)
        P_R_arr = np.asarray(P_R, dtype=np.int32)
        pL = self._extract_patches(P_L_arr, flip=False)
        pR = self._extract_patches(P_R_arr, flip=True)
        dist_L, dist_R = self._nn_distances(P_L_arr, P_R_arr, pL, pR)

        S_cost = 0.5 * (float(dist_L.mean()) + float(dist_R.mean()))
        score = 1.0 / (1.0 + S_cost)

        acc = np.zeros((self.H, self.W), dtype=np.float64)
        cnt = np.zeros((self.H, self.W), dtype=np.int32)
        h = self.h
        centers = np.concatenate([P_L_arr, P_R_arr])
        values = 1.0 / (1.0 + np.concatenate([dist_L, dist_R]).astype(np.float64))
        for (cx, cy), v in zip(centers, values):
            acc[cy - h:cy + h + 1, cx - h:cx + h + 1] += v
            cnt[cy - h:cy + h + 1, cx - h:cx + h + 1] += 1
        m = cnt > 0
        heat[m] = (acc[m] / cnt[m]).astype(np.float32)
        return score, heat

    def _simple_patch_symmetry(self, pL, pR):
        if len(pL) == 0 or len(pR) == 0:
            return 0.0
        correlations = []
        for i, patch_l in enumerate(pL):
            if i < len(pR):
                patch_r = pR[i]
                corr = np.corrcoef(patch_l, patch_r)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        if correlations:
            return float(np.mean(correlations))
        else:
            return 0.0

    def _paired_samples(self):
        xs = self.x_s
        P_L = []
        P_R = []
        ys = range(self.h, self.H - self.h, self.k)

        xL_list = []
        xR_list = []
        n = 0
        while True:
            xL = int(round(xs - self.h - n * self.k))
            xR = int(round(2 * xs - xL))
            if not (self._valid_center(xL, self.h) and self._valid_center(xR, self.h)):
                break
            xL_list.append(xL)
            xR_list.append(xR)
            n += 1

        for y in ys:
            for xL, xR in zip(xL_list, xR_list):
                P_L.append((xL, y))
                P_R.append((xR, y))

        return P_L, P_R

    def _valid_center(self, x, y):
        return (self.h <= x < self.W - self.h) and (self.h <= y < self.H - self.h)

    def _extract_patches(self, centers, flip):
        xs = centers[:, 0]
        ys = centers[:, 1]

        off = np.arange(-self.h, self.h + 1, dtype=np.int32)
        Y = ys[:, None] + off[None, :]
        X = xs[:, None] + off[None, :]

        patches = self.image[Y[:, :, None], X[:, None, :]]
        if flip:
            patches = patches[:, :, ::-1]

        return patches.reshape(len(centers), -1).astype(np.float32)

    def _x_ell(self, x):
        return np.minimum(x, (2.0 * self.x_s) - x).astype(np.float32)


# Factory class for creating calculators
class SymmetryCalculatorFactory:
    _calculators = {
        "pixel_correlation": PixelCorrelationCalculator,
        "sliding_window": SlidingWindowCalculator,
        "dct": DCTCalculator,
        "eros": ErosCalculator,
        "gradient": GradientCalculator,
        "multi_scale_gradient": MultiScaleGradientCalculator,
        "hog": HOGCalculator,
        "phog": PHOGCalculator,
        "gabor": GaborCalculator,
        "alexnet": AlexNetCalculator,
        "deep_features": DeepFeatureCalculator,
        "weighted_binary": WeightedBinarySymmetryCalculator,
        "local_global": LocalGlobalSymmetryCalculator,
    }

    @staticmethod
    def create(method, **kwargs):
        if method not in SymmetryCalculatorFactory._calculators:
            available = ", ".join(SymmetryCalculatorFactory._calculators.keys())
            raise ValueError(f"Unknown method: {method}. Available methods: {available}")
        return SymmetryCalculatorFactory._calculators[method](**kwargs)

    @staticmethod
    def calculate(bgr_image, method, **kwargs):
        calculator = SymmetryCalculatorFactory.create(method, **kwargs)
        return calculator.calculate_score(bgr_image)

    @staticmethod
    def list_methods():
        return list(SymmetryCalculatorFactory._calculators.keys())


def calculate_symmetry_score(bgr_image, method="multi_scale_gradient", **kwargs):
    """
    Main entry point for symmetry score calculation.

    Args:
        bgr_image: Input image in BGR format (H x W x 3)
        method: Calculation method name (see list_symmetry_methods() for options)
        **kwargs: Method-specific parameters

    Returns:
        float: Symmetry score (higher generally means more symmetric)
    """
    return SymmetryCalculatorFactory.calculate(bgr_image, method, **kwargs)


def list_symmetry_methods():
    """Get list of available symmetry calculation methods"""
    return SymmetryCalculatorFactory.list_methods()
