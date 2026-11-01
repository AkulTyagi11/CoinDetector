"""
Robust Automated Coin Detection Engine using OpenCV.
Features Illumination Normalization (Morphological Top-Hat/Black-Hat),
Watershed Separation for touching coins, High-Circularity Geometry Filtering,
and Circumferential Edge Coherence Verification to reject textured backgrounds (e.g. wood grain).
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from core.config import DEFAULT_CV_PARAMS

class CoinDetector:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(DEFAULT_CV_PARAMS)
        if params:
            self.params.update(params)

    def normalize_illumination(self, gray: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Removes uneven lighting gradients, shadows, and background textures (like wood grain)
        using an adaptive morphological structuring element (Top-Hat for dark/medium tables,
        Black-Hat for bright paper/white surfaces).
        """
        h, w = gray.shape[:2]
        bg_ratio = float(self.params.get("bg_kernel_ratio", 0.13))
        k_size = int(max(h, w) * bg_ratio)
        if k_size % 2 == 0:
            k_size += 1
        k_size = max(45, min(k_size, 181))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        # Check background brightness from image borders
        border = np.concatenate([
            gray[:20, :].ravel(),
            gray[-20:, :].ravel(),
            gray[:, :20].ravel(),
            gray[:, -20:].ravel()
        ])
        bg_mean = float(border.mean())
        bg_is_bright = bg_mean > 160.0

        if bg_is_bright:
            # Coins are darker than background (e.g. white paper, bright table)
            norm = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            # Coins are brighter/reflective compared to background (e.g. wood, cloth, desk)
            norm = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        return norm, bg_is_bright

    def verify_edge_coherence(
        self,
        grad_mag: np.ndarray,
        cx: float,
        cy: float,
        r: float,
        num_samples: int = 36,
        grad_thresh: float = 35.0
    ) -> Tuple[bool, float]:
        """
        Validates whether a candidate circle (cx, cy, r) has a genuine circular boundary.
        Samples perimeter points and measures gradient magnitude step.
        Spurious detections in wood grain have low gradient step along circular arcs.
        """
        h, w = grad_mag.shape[:2]
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        xs = np.clip((cx + r * np.cos(angles)).astype(int), 0, w - 1)
        ys = np.clip((cy + r * np.sin(angles)).astype(int), 0, h - 1)

        vals = grad_mag[ys, xs]
        coherence = float(np.mean(vals > grad_thresh))
        mean_val = float(np.mean(vals))

        thresh = float(self.params.get("edge_coherence_threshold", 0.60))
        # Real coins pass with high coherence or very high average gradient
        is_coherent = (coherence >= thresh) or (mean_val >= 90.0)
        return is_coherent, coherence

    def detect_morphological_contours(
        self,
        norm_img: np.ndarray,
        grad_mag: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float]], np.ndarray, np.ndarray]:
        """
        Extracts circular coin candidates from the illumination-normalized image.
        Uses Otsu thresholding, Distance Transform for touching separation,
        and strict Circularity / Solidity / Edge Coherence filtering.
        """
        h, w = norm_img.shape[:2]
        total_pixels = h * w

        # Gaussian smoothing to suppress internal engravings
        ksize = int(self.params.get("blur_ksize", 9))
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(norm_img, (ksize, ksize), 2)

        # Otsu threshold on normalized image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological Closing to fill small text/holes inside coin body
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=2)

        # Distance Transform to separate touching coins
        dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist > 0:
            dist_norm = (dist_transform / max_dist * 255).astype(np.uint8)
        else:
            dist_norm = np.zeros_like(norm_img, dtype=np.uint8)

        # Contours
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area_ratio = float(self.params.get("min_area_ratio", 0.0018))
        max_area_ratio = float(self.params.get("max_area_ratio", 0.15))
        min_area = total_pixels * min_area_ratio
        max_area = total_pixels * max_area_ratio

        circ_thresh = float(self.params.get("circularity_threshold", 0.65))
        solidity_thresh = float(self.params.get("solidity_threshold", 0.80))
        area_ratio_thresh = float(self.params.get("area_ratio_threshold", 0.65))

        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)

            # If area is unusually large, it represents touching/clustered coins -> split via distance transform
            expected_coin_area = total_pixels * 0.02
            if area > expected_coin_area * 1.5:
                mask_cnt = np.zeros_like(closed)
                cv2.drawContours(mask_cnt, [c], -1, 255, -1)
                dt_cluster = cv2.distanceTransform(mask_cnt, cv2.DIST_L2, 5)
                dt_smooth = cv2.GaussianBlur(dt_cluster, (5, 5), 0)
                dt_max = dt_smooth.max()
                if dt_max > 12:
                    _, sure_fg = cv2.threshold(dt_smooth, 0.45 * dt_max, 255, 0)
                    sure_fg = np.uint8(sure_fg)
                    sub_cnts, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for sc in sub_cnts:
                        (sub_cx, sub_cy), sub_r = cv2.minEnclosingCircle(sc)
                        py = min(max(0, int(sub_cy)), h - 1)
                        px = min(max(0, int(sub_cx)), w - 1)
                        exp_r = dt_cluster[py, px]
                        r_final = max(sub_r, exp_r)
                        if r_final >= 15:
                            candidates.append((float(sub_cx), float(sub_cy), float(r_final)))
                continue

            if area < min_area:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            (cx, cy), r = cv2.minEnclosingCircle(c)
            circle_area = np.pi * (r ** 2)
            area_ratio = area / circle_area if circle_area > 0 else 0

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Edge Coherence Check
            if self.params.get("enable_edge_coherence", True):
                is_coherent, _ = self.verify_edge_coherence(grad_mag, cx, cy, r)
                if not is_coherent:
                    continue

            if circularity >= circ_thresh and area_ratio >= area_ratio_thresh and solidity >= solidity_thresh:
                candidates.append((float(cx), float(cy), float(r)))

        return candidates, closed, dist_norm

    def detect_masked_hough_circles(
        self,
        norm_img: np.ndarray,
        binary_mask: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Runs Hough Circle Transform ONLY on the foreground-masked, normalized image.
        This prevents Hough from firing hundreds of false circles on background wood grain.
        """
        masked = cv2.bitwise_and(norm_img, norm_img, mask=binary_mask)
        blurred = cv2.GaussianBlur(masked, (9, 9), 2)

        dp = float(self.params.get("hough_dp", 1.2))
        min_dist = float(self.params.get("hough_min_dist", 45))
        param1 = float(self.params.get("hough_param1", 70))
        param2 = float(self.params.get("hough_param2", 38))
        min_radius = int(self.params.get("min_radius", 20))
        max_radius = int(self.params.get("max_radius", 140))

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                detected.append((float(i[0]), float(i[1]), float(i[2])))

        return detected

    def non_max_suppression_circles(
        self,
        circles: List[Tuple[float, float, float]],
        overlap_dist_ratio: float = 0.55
    ) -> List[Tuple[float, float, float]]:
        """
        Suppresses redundant, concentric, or heavily overlapping circles.
        """
        if not circles:
            return []

        # Sort by radius descending
        sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)
        keep = []

        for (x1, y1, r1) in sorted_circles:
            suppressed = False
            for (x2, y2, r2) in keep:
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < overlap_dist_ratio * max(r1, r2):
                    suppressed = True
                    break
            if not suppressed:
                keep.append((x1, y1, r1))

        return keep

    def detect_coins(
        self,
        image_bgr: np.ndarray,
        method: str = "adaptive"
    ) -> Tuple[List[Tuple[float, float, float]], Dict[str, np.ndarray]]:
        """
        High-level coin detection dispatcher.
        Methods:
            - 'adaptive' (Default): Illumination Normalization + High-Circularity Geometry + Circumferential Edge Verification.
            - 'hough': Masked Hough Circle Transform on normalized image.
            - 'hybrid': Ensemble of adaptive contour detector and masked Hough circles.
        """
        # 1. Grayscale
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr.copy()

        # 2. Gradient Magnitude Map for edge coherence
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 3. Illumination Normalization
        norm_img, bg_is_bright = self.normalize_illumination(gray)

        # 4. Adaptive Morphological Detection
        contour_circles, binary_thresh, dist_norm = self.detect_morphological_contours(norm_img, grad_mag)

        # 5. Canny edge map for visual inspection
        canny_high = int(self.params.get("canny_high", 110))
        edges = cv2.Canny(norm_img, int(canny_high * 0.4), canny_high)

        # Method Dispatch
        if method == "hough":
            # Masked Hough on foreground
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated_mask = cv2.dilate(binary_thresh, dilate_k, iterations=1)
            hough_raw = self.detect_masked_hough_circles(norm_img, dilated_mask)
            # Filter Hough circles by edge coherence
            hough_valid = []
            for (hx, hy, hr) in hough_raw:
                is_coh, _ = self.verify_edge_coherence(grad_mag, hx, hy, hr)
                if is_coh:
                    hough_valid.append((hx, hy, hr))
            final_circles = self.non_max_suppression_circles(hough_valid)

        elif method == "hybrid":
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated_mask = cv2.dilate(binary_thresh, dilate_k, iterations=1)
            hough_raw = self.detect_masked_hough_circles(norm_img, dilated_mask)
            combined = contour_circles + hough_raw
            final_circles = self.non_max_suppression_circles(combined)

        else:  # 'adaptive' (Default)
            final_circles = self.non_max_suppression_circles(contour_circles)

        stages = {
            "original": image_bgr,
            "grayscale": gray,
            "normalized": norm_img,
            "blurred": cv2.GaussianBlur(norm_img, (9, 9), 2),
            "binary_thresh": binary_thresh,
            "distance_transform": dist_norm,
            "edges": edges
        }

        return final_circles, stages
