"""
Coin Denomination Classification and Feature Reasoning.
Analyzes physical dimensions, radial HSV/BGR color profiles, bimetallic contrast,
and surface engraving texture (Laplacian variance) to accurately classify Indian Rupee coins.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.config import COIN_SPECS, CoinSpec

class CoinClassifier:
    def __init__(self, default_ppm: float = 4.5):
        self.default_ppm = default_ppm
        self.coin_specs = COIN_SPECS

    def analyze_coin_features(
        self,
        image_bgr: np.ndarray,
        x: int,
        y: int,
        r: int
    ) -> Dict[str, Any]:
        """
        Extracts color profiles, bimetallic radial contrast, and surface engraving texture
        from a circular coin ROI.
        """
        h, w = image_bgr.shape[:2]
        r = max(int(r), 6)
        x, y = int(x), int(y)

        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r)
        y2 = min(h, y + r)

        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 6 or roi.shape[1] < 6:
            return {
                "tone": "silver",
                "is_golden": False,
                "is_bimetallic": False,
                "laplacian_var": 400.0,
                "sat_mean": 20.0,
                "hue_mean": 0.0
            }

        roi_h, roi_w = roi.shape[:2]
        roi_cx = x - x1
        roi_cy = y - y1

        Y, X = np.ogrid[:roi_h, :roi_w]
        dist_from_center = np.sqrt((X - roi_cx) ** 2 + (Y - roi_cy) ** 2)

        # Concentric regions
        inner_mask = dist_from_center <= 0.45 * r
        outer_mask = (dist_from_center >= 0.65 * r) & (dist_from_center <= 0.90 * r)
        full_mask = dist_from_center <= 0.92 * r

        # HSV Analysis
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_channel = roi_hsv[:, :, 0]
        s_channel = roi_hsv[:, :, 1]
        v_channel = roi_hsv[:, :, 2]

        full_s_mean = float(np.mean(s_channel[full_mask])) if np.any(full_mask) else 30.0
        full_h_mean = float(np.mean(h_channel[full_mask])) if np.any(full_mask) else 20.0
        full_v_mean = float(np.mean(v_channel[full_mask])) if np.any(full_mask) else 150.0

        inner_s = float(np.mean(s_channel[inner_mask])) if np.any(inner_mask) else full_s_mean
        outer_s = float(np.mean(s_channel[outer_mask])) if np.any(outer_mask) else full_s_mean

        # BGR Analysis for Bimetallic Contrast (Outer bronze rim vs Inner silver core)
        b_ch = roi[:, :, 0].astype(float)
        g_ch = roi[:, :, 1].astype(float)

        inner_bg_ratio = float(np.mean(b_ch[inner_mask]) / (np.mean(g_ch[inner_mask]) + 1e-5)) if np.any(inner_mask) else 1.0
        outer_bg_ratio = float(np.mean(b_ch[outer_mask]) / (np.mean(g_ch[outer_mask]) + 1e-5)) if np.any(outer_mask) else 1.0
        bg_diff = inner_bg_ratio - outer_bg_ratio

        # Bimetallic detection: Outer ring is bronze (drop in Blue/Green ratio), inner core is silver
        # Or large saturation contrast between outer rim and inner core
        is_bimetallic = (bg_diff > 0.07) or (outer_s > 75 and inner_s < 50 and (outer_s - inner_s) > 40)

        # Golden detection (Brass ₹5 coin):
        # Golden hue in HSV is ~10 to 42, with notable saturation (> 40) across whole coin
        is_golden = (10 <= full_h_mean <= 42) and (full_s_mean >= 40) and not is_bimetallic

        if is_bimetallic:
            tone = "bimetallic"
        elif is_golden:
            tone = "golden"
        else:
            tone = "silver"

        # Texture analysis: Laplacian variance of inner face
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inner_gray = roi_gray[inner_mask] if np.any(inner_mask) else roi_gray
        laplacian_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())

        return {
            "tone": tone,
            "is_golden": is_golden,
            "is_bimetallic": is_bimetallic,
            "sat_mean": round(full_s_mean, 1),
            "hue_mean": round(full_h_mean, 1),
            "val_mean": round(full_v_mean, 1),
            "inner_bg_ratio": round(inner_bg_ratio, 3),
            "outer_bg_ratio": round(outer_bg_ratio, 3),
            "bg_diff": round(bg_diff, 3),
            "laplacian_var": round(laplacian_var, 1)
        }

    def calibrate_ppm_from_coins(
        self,
        coins: List[Dict[str, Any]],
        ref_denomination: int = 10,
        manual_ppm: Optional[float] = None
    ) -> float:
        """
        Determines the pixels-per-millimeter (PPM) scale.
        Can use manual PPM, designated reference coin, or automatic multi-feature heuristic.
        """
        if manual_ppm and manual_ppm > 0:
            return manual_ppm

        if not coins:
            return self.default_ppm

        # Check if user marked a specific reference coin
        for coin in coins:
            if coin.get("is_reference") and coin.get("reference_diameter_mm"):
                ref_mm = coin["reference_diameter_mm"]
                return (2.0 * coin["radius"]) / ref_mm

        # Auto Heuristic:
        # 1. If bimetallic coin detected, anchor to ₹10 (27.0 mm)
        bimetallic_coins = [c for c in coins if c.get("features", {}).get("is_bimetallic")]
        if bimetallic_coins:
            largest_bimetallic = max(bimetallic_coins, key=lambda c: c["radius"])
            return (2.0 * largest_bimetallic["radius"]) / 27.0

        # 2. If golden coin detected, anchor to ₹5 (23.0 mm)
        golden_coins = [c for c in coins if c.get("features", {}).get("is_golden")]
        if golden_coins:
            med_golden = sorted(golden_coins, key=lambda c: c["radius"])[len(golden_coins) // 2]
            return (2.0 * med_golden["radius"]) / 23.0

        # 3. Fallback: Anchor largest coin to default reference (default: ₹10 = 27mm)
        largest_coin = max(coins, key=lambda c: c["radius"])
        ref_spec = self.coin_specs.get(ref_denomination, self.coin_specs[10])
        return (2.0 * largest_coin["radius"]) / ref_spec.diameter_mm

    def classify_coin(
        self,
        diameter_mm: float,
        features: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        Classifies physical diameter (mm) and visual features into an Indian Rupee denomination:
        - ₹10 (27mm, Bimetallic)
        - ₹20 (27mm, Bimetallic / 12-sided)
        - ₹5 (23mm, Golden)
        - ₹2 (25mm, Silver, textured Rupee symbol)
        - ₹1 (20mm, Silver 2011 series OR 25mm smooth Hastamudra thumbs-up series)
        """
        is_bimetallic = features.get("is_bimetallic", False)
        is_golden = features.get("is_golden", False)
        lap_var = features.get("laplacian_var", 500.0)

        # 1. Bimetallic coins (₹10 / ₹20)
        if is_bimetallic and diameter_mm >= 25.0:
            conf = min(1.0, max(0.70, 1.0 - abs(diameter_mm - 27.0) / 10.0))
            return 10, round(conf, 2)

        # 2. Golden / Brass coins (₹5)
        if is_golden and 21.0 <= diameter_mm <= 25.5:
            conf = min(1.0, max(0.70, 1.0 - abs(diameter_mm - 23.0) / 8.0))
            return 5, round(conf, 2)

        # 3. Silver Coins:
        # A. Small silver coin (< 22.4 mm) -> ₹1 (2011 Rupee Symbol series, 20.0 mm)
        if diameter_mm < 22.4:
            conf = min(1.0, max(0.65, 1.0 - abs(diameter_mm - 20.0) / 6.0))
            return 1, round(conf, 2)

        # B. Medium silver coin (22.4 to 26.5 mm):
        # Could be ₹2 (25 mm) OR ₹1 Hastamudra Thumbs-Up series (25 mm)!
        if 22.4 <= diameter_mm < 26.5:
            # Hastamudra ₹1 coin has a plain smooth field with low Laplacian variance (< 680)
            # 2011 ₹2 coin has dense floral motifs and Rupee symbol (Laplacian variance >= 680)
            texture_thresh = 680.0
            if lap_var < texture_thresh:
                return 1, 0.90
            else:
                conf = min(1.0, max(0.65, 1.0 - abs(diameter_mm - 25.0) / 6.0))
                return 2, round(conf, 2)

        # C. Large silver or generic (>= 26.5 mm) -> ₹10 fallback
        conf = min(1.0, max(0.60, 1.0 - abs(diameter_mm - 27.0) / 8.0))
        return 10, round(conf, 2)

    def process_and_classify_all(
        self,
        image_bgr: np.ndarray,
        raw_circles: List[Tuple[float, float, float]],
        calibration_mode: str = "auto",
        ref_denomination: int = 10,
        manual_ppm: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Processes candidate circles, extracts multimodal features,
        calibrates scale, and classifies all coins.
        """
        if not raw_circles:
            return [], {
                "total_count": 0,
                "total_amount": 0,
                "breakdown": {1: 0, 2: 0, 5: 0, 10: 0, 20: 0},
                "pixels_per_mm": self.default_ppm
            }

        # Step 1: Feature Extraction
        candidates = []
        for idx, (x, y, r) in enumerate(raw_circles):
            features = self.analyze_coin_features(image_bgr, int(x), int(y), int(r))
            candidates.append({
                "id": idx + 1,
                "center": (int(x), int(y)),
                "radius": float(r),
                "diameter_px": float(2.0 * r),
                "features": features
            })

        # Step 2: Calibrate PPM
        ppm = self.calibrate_ppm_from_coins(
            candidates,
            ref_denomination=ref_denomination,
            manual_ppm=manual_ppm if calibration_mode == "manual_ppm" else None
        )
        if ppm <= 0.1:
            ppm = self.default_ppm

        # Step 3: Classify each coin
        classified_coins = []
        breakdown = {1: 0, 2: 0, 5: 0, 10: 0, 20: 0}
        total_amount = 0

        for coin in candidates:
            diameter_mm = coin["diameter_px"] / ppm
            denom, conf = self.classify_coin(diameter_mm, coin["features"])

            spec = self.coin_specs.get(denom, self.coin_specs[1])
            breakdown[denom] = breakdown.get(denom, 0) + 1
            total_amount += denom

            coin.update({
                "diameter_mm": round(diameter_mm, 2),
                "denomination": denom,
                "denomination_str": spec.symbol,
                "coin_name": spec.name,
                "confidence": conf,
                "color_tone": coin["features"]["tone"].capitalize(),
                "display_color_bgr": spec.display_color_bgr,
                "display_color_hex": spec.display_color_hex
            })
            classified_coins.append(coin)

        summary = {
            "total_count": len(classified_coins),
            "total_amount": total_amount,
            "breakdown": breakdown,
            "pixels_per_mm": round(ppm, 3)
        }

        return classified_coins, summary
