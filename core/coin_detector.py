"""
Automated Coin Detection Engine using OpenCV.
Implements Hough Circles, Contour Analysis with Circularity Filtering,
Watershed Segmentation for touching coins, and Hybrid Ensemble detection.
Generates step-by-step pipeline visualizations for academic/viva presentations.
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

    def preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocesses input image: Grayscale conversion, CLAHE contrast enhancement,
        and Gaussian / Bilateral filtering to suppress noise while preserving coin rims.
        """
        # Convert to Grayscale
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.params.get("clahe_enabled", True):
            clip_limit = float(self.params.get("clahe_clip_limit", 2.0))
            grid_size = int(self.params.get("clahe_grid_size", 8))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            enhanced = clahe.apply(gray)
        else:
            enhanced = gray.copy()

        # Smoothing / Denoising
        blur_method = self.params.get("blur_method", "gaussian")
        ksize = int(self.params.get("blur_ksize", 9))
        if ksize % 2 == 0:
            ksize += 1

        if blur_method == "bilateral":
            blurred = cv2.bilateralFilter(
                enhanced,
                d=ksize,
                sigmaColor=self.params.get("bilateral_sigma_color", 75),
                sigmaSpace=self.params.get("bilateral_sigma_space", 75)
            )
        elif blur_method == "median":
            blurred = cv2.medianBlur(enhanced, ksize)
        else:  # Gaussian
            blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), 2)

        return gray, enhanced, blurred

    def detect_hough_circles(self, blurred: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Detects circular coin outlines using OpenCV Hough Circle Transform.
        Returns: list of (center_x, center_y, radius)
        """
        dp = float(self.params.get("hough_dp", 1.2))
        min_dist = float(self.params.get("hough_min_dist", 45))
        param1 = float(self.params.get("hough_param1", 70))
        param2 = float(self.params.get("hough_param2", 32))
        min_radius = int(self.params.get("min_radius", 18))
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

        return self.non_max_suppression_circles(detected)

    def detect_contours_and_watershed(
        self,
        image_bgr: np.ndarray,
        blurred: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float]], np.ndarray, np.ndarray]:
        """
        Detects coins using Adaptive/Otsu Thresholding, Distance Transform,
        Watershed Segmentation (to split touching coins), and Circularity Shape Filtering.
        """
        h, w = blurred.shape[:2]

        # Otsu thresholding or adaptive thresholding
        block_size = int(self.params.get("adaptive_block_size", 25))
        if block_size % 2 == 0:
            block_size += 1
        c_val = int(self.params.get("adaptive_c", 3))

        # Determine if background is light or dark by sampling image borders
        border_pixels = np.concatenate([
            blurred[:15, :].ravel(),
            blurred[-15:, :].ravel(),
            blurred[:, :15].ravel(),
            blurred[:, -15:].ravel()
        ])
        bg_is_light = float(np.mean(border_pixels)) > 125.0
        inv_flag = cv2.THRESH_BINARY_INV if bg_is_light else cv2.THRESH_BINARY

        # Otsu threshold
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, inv_flag + cv2.THRESH_OTSU)

        # Adaptive threshold
        thresh_adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, inv_flag, block_size, c_val
        )

        # Use Otsu or Adaptive based on foreground ratio
        fg_ratio = np.count_nonzero(thresh_otsu) / (h * w)
        if 0.02 <= fg_ratio <= 0.65:
            binary = thresh_otsu
        else:
            binary = thresh_adapt

        # Morphological operations to remove small noise and close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Distance Transform to separate touching coins
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist > 0:
            dist_norm = (dist_transform / max_dist * 255).astype(np.uint8)
            _, sure_fg = cv2.threshold(dist_transform, 0.45 * max_dist, 255, 0)
            sure_fg = np.uint8(sure_fg)
        else:
            dist_norm = np.zeros_like(blurred, dtype=np.uint8)
            sure_fg = np.zeros_like(blurred, dtype=np.uint8)

        # Unknown boundary region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Connected components markers for Watershed
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply Watershed
        watershed_img = image_bgr.copy()
        markers = cv2.watershed(watershed_img, markers)

        # Extract coins from segmented contours
        min_area = float(self.params.get("min_contour_area", 500))
        max_area = float(self.params.get("max_contour_area", 85000))
        circ_thresh = float(self.params.get("circularity_threshold", 0.65))

        detected_circles = []
        unique_markers = np.unique(markers)

        for marker_id in unique_markers:
            if marker_id <= 1:  # 0 is boundary, 1 is background
                continue

            # Mask for current coin segment
            coin_mask = np.uint8(markers == marker_id)
            contours, _ = cv2.findContours(coin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                # Circularity: 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
                if circularity < circ_thresh:
                    continue

                # Minimum Enclosing Circle
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                min_r = int(self.params.get("min_radius", 18))
                max_r = int(self.params.get("max_radius", 140))

                if min_r <= radius <= max_r:
                    detected_circles.append((float(cx), float(cy), float(radius)))

        # Edge-based contour extraction for reflective coins
        canny_low = int(self.params.get("canny_low", 35))
        canny_high = int(self.params.get("canny_high", 110))
        edges = cv2.Canny(blurred, canny_low, canny_high)
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k, iterations=2)
        edge_cnts, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in edge_cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity >= circ_thresh:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                min_r = int(self.params.get("min_radius", 18))
                max_r = int(self.params.get("max_radius", 140))
                if min_r <= radius <= max_r:
                    detected_circles.append((float(cx), float(cy), float(radius)))

        return self.non_max_suppression_circles(detected_circles), binary, dist_norm

    def non_max_suppression_circles(
        self,
        circles: List[Tuple[float, float, float]],
        overlap_thresh: float = 0.50
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
                # If centers are closer than 55% of the larger radius, it's a duplicate/concentric detection
                if dist < 0.55 * max(r1, r2):
                    suppressed = True
                    break
            if not suppressed:
                keep.append((x1, y1, r1))

        return keep

    def detect_coins(
        self,
        image_bgr: np.ndarray,
        method: str = "hybrid"
    ) -> Tuple[List[Tuple[float, float, float]], Dict[str, np.ndarray]]:
        """
        High-level detection dispatcher.
        Methods:
            - 'hough': Hough Circle Transform
            - 'contour': Contour & Watershed Segmentation
            - 'hybrid': Ensemble of both with NMS deduplication

        Returns:
            circles: list of (cx, cy, r)
            stages: dictionary of intermediate pipeline images
        """
        gray, enhanced, blurred = self.preprocess(image_bgr)

        # Canny edge detection for pipeline visualization
        canny_low = int(self.params.get("canny_low", 35))
        canny_high = int(self.params.get("canny_high", 110))
        edges = cv2.Canny(blurred, canny_low, canny_high)

        # Contour and watershed data
        contour_circles, binary_thresh, dist_norm = self.detect_contours_and_watershed(image_bgr, blurred)

        # Hough circles data
        hough_circles = self.detect_hough_circles(blurred)

        if method == "hough":
            final_circles = hough_circles
        elif method == "contour":
            final_circles = contour_circles
        else:  # hybrid
            # Combine both and run NMS
            combined = hough_circles + contour_circles
            final_circles = self.non_max_suppression_circles(combined)

        stages = {
            "original": image_bgr,
            "grayscale": gray,
            "enhanced": enhanced,
            "blurred": blurred,
            "edges": edges,
            "binary_thresh": binary_thresh,
            "distance_transform": dist_norm
        }

        return final_circles, stages
