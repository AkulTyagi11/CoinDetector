"""
Coin Denomination Classification and Feature Extraction.
Analyzes geometric dimensions, HSV color profiles, and bimetallic signatures
to classify coins into Indian Rupee (INR) denominations (₹1, ₹2, ₹5, ₹10, ₹20).
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.config import COIN_SPECS, CoinSpec

class CoinClassifier:
    def __init__(self, default_ppm: float = 5.0):
        self.default_ppm = default_ppm
        self.coin_specs = COIN_SPECS

    def analyze_coin_color(self, image_bgr: np.ndarray, x: int, y: int, r: int) -> Dict[str, Any]:
        """
        Extracts color profile and bimetallic features from a circular coin ROI.
        Analyzes the inner core (0 to 0.5r) vs outer ring (0.65r to 0.95r).
        """
        h, w = image_bgr.shape[:2]
        r = max(int(r), 5)
        x, y = int(x), int(y)

        # Check bounds
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r)
        y2 = min(h, y + r)

        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return {"tone": "silver", "is_golden": False, "is_bimetallic": False, "sat_mean": 20, "hue_mean": 0}

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_h, roi_w = roi.shape[:2]
        roi_cx = x - x1
        roi_cy = y - y1

        # Create coordinate grids
        Y, X = np.ogrid[:roi_h, :roi_w]
        dist_from_center = np.sqrt((X - roi_cx) ** 2 + (Y - roi_cy) ** 2)

        # Inner core mask (0 to 0.48 * r)
        inner_mask = (dist_from_center <= 0.48 * r) & (dist_from_center >= 0.05 * r)
        # Outer ring mask (0.65 to 0.92 * r)
        outer_mask = (dist_from_center >= 0.65 * r) & (dist_from_center <= 0.92 * r)
        # Full coin mask
        full_mask = dist_from_center <= 0.90 * r

        h_channel = roi_hsv[:, :, 0]
        s_channel = roi_hsv[:, :, 1]
        v_channel = roi_hsv[:, :, 2]

        full_s_mean = float(np.mean(s_channel[full_mask])) if np.any(full_mask) else 30.0
        full_h_mean = float(np.mean(h_channel[full_mask])) if np.any(full_mask) else 20.0
        full_v_mean = float(np.mean(v_channel[full_mask])) if np.any(full_mask) else 150.0

        inner_s = float(np.mean(s_channel[inner_mask])) if np.any(inner_mask) else full_s_mean
        outer_s = float(np.mean(s_channel[outer_mask])) if np.any(outer_mask) else full_s_mean
        inner_h = float(np.mean(h_channel[inner_mask])) if np.any(inner_mask) else full_h_mean
        outer_h = float(np.mean(h_channel[outer_mask])) if np.any(outer_mask) else full_h_mean

        # Bimetallic detection (e.g. ₹10 coin: outer ring is brass/golden S>75, inner core is silver S<50)
        sat_contrast = abs(outer_s - inner_s)
        is_bimetallic = (inner_s < 50 and outer_s > 75 and (outer_s - inner_s) > 45)

        # Golden detection (Brass ₹5 coin):
        # Golden hue in OpenCV HSV is ~10 to 42, with notable saturation (> 40) both inner and outer
        is_golden = (10 <= full_h_mean <= 42) and (full_s_mean >= 40) and not is_bimetallic

        if is_bimetallic:
            tone = "bimetallic"
        elif is_golden:
            tone = "golden"
        else:
            tone = "silver"

        return {
            "tone": tone,
            "is_golden": is_golden,
            "is_bimetallic": is_bimetallic,
            "sat_mean": round(full_s_mean, 1),
            "hue_mean": round(full_h_mean, 1),
            "val_mean": round(full_v_mean, 1),
            "sat_contrast": round(sat_contrast, 1)
        }

    def calibrate_ppm_from_coins(
        self,
        coins: List[Dict[str, Any]],
        ref_denomination: int = 10,
        manual_ppm: Optional[float] = None
    ) -> float:
        """
        Determines the pixels-per-millimeter (PPM) scale.
        Can use manual PPM, a designated reference coin, or an automatic heuristic.
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
        # If there's a bimetallic coin detected, it's very likely ₹10 (27mm)
        bimetallic_coins = [c for c in coins if c.get("color_info", {}).get("is_bimetallic")]
        if bimetallic_coins:
            # Anchor on largest bimetallic coin
            largest_bimetallic = max(bimetallic_coins, key=lambda c: c["radius"])
            return (2.0 * largest_bimetallic["radius"]) / 27.0

        # Golden coin detected: likely ₹5 (23mm)
        golden_coins = [c for c in coins if c.get("color_info", {}).get("is_golden")]
        if golden_coins and len(coins) > 1:
            # Anchor to ₹5 (23mm)
            med_golden = sorted(golden_coins, key=lambda c: c["radius"])[len(golden_coins) // 2]
            return (2.0 * med_golden["radius"]) / 23.0

        # Fallback: Assume the largest coin in the frame is the reference denomination (default: ₹10 = 27mm)
        largest_coin = max(coins, key=lambda c: c["radius"])
        ref_spec = self.coin_specs.get(ref_denomination, self.coin_specs[10])
        return (2.0 * largest_coin["radius"]) / ref_spec.diameter_mm

    def classify_coin(
        self,
        diameter_mm: float,
        color_info: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        Matches a physical diameter (mm) and color information to the best fitting Indian coin denomination.
        Returns: (denomination, confidence_score)
        """
        tone = color_info.get("tone", "silver")
        is_bimetallic = color_info.get("is_bimetallic", False)
        is_golden = color_info.get("is_golden", False)

        # Standard INR Diameters:
        # ₹1: 20.0 mm (Silver)
        # ₹5: 23.0 mm (Golden / Brass)
        # ₹2: 25.0 mm (Silver)
        # ₹10: 27.0 mm (Bimetallic)
        # ₹20: 27.0 mm (Bimetallic / 12-sided)

        # Direct strong multimodal signals
        if is_bimetallic:
            # Definitely ₹10 or ₹20
            conf = min(1.0, max(0.65, 1.0 - abs(diameter_mm - 27.0) / 10.0))
            return 10, round(conf, 2)

        if is_golden:
            # Characteristic ₹5 coin
            conf = min(1.0, max(0.65, 1.0 - abs(diameter_mm - 23.0) / 8.0))
            return 5, round(conf, 2)

        # For Silver coins (₹1 vs ₹2, or uncolored ₹5):
        # ₹1 is ~20mm, ₹2 is ~25mm
        diff_1 = abs(diameter_mm - 20.0)
        diff_2 = abs(diameter_mm - 25.0)
        diff_5 = abs(diameter_mm - 23.0)

        # If it's silver, ₹1 or ₹2 are primary candidates
        if tone == "silver":
            if diff_1 < diff_2:
                conf = max(0.5, 1.0 - diff_1 / 6.0)
                return 1, round(min(1.0, conf), 2)
            else:
                conf = max(0.5, 1.0 - diff_2 / 6.0)
                return 2, round(min(1.0, conf), 2)

        # Minimum distance to any standard coin
        best_denom = 1
        min_diff = 999.0
        for denom, spec in self.coin_specs.items():
            diff = abs(diameter_mm - spec.diameter_mm)
            if diff < min_diff:
                min_diff = diff
                best_denom = denom

        conf = max(0.4, 1.0 - min_diff / 8.0)
        return best_denom, round(min(1.0, conf), 2)

    def process_and_classify_all(
        self,
        image_bgr: np.ndarray,
        raw_circles: List[Tuple[float, float, float]],
        calibration_mode: str = "auto",
        ref_denomination: int = 10,
        manual_ppm: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Takes raw detected circle coordinates [(x, y, r)], analyzes features,
        calibrates scale, and classifies all coins.

        Returns:
            coins: list of enriched coin dicts
            summary: aggregate statistics (total value, count by denomination, etc.)
        """
        if not raw_circles:
            return [], {
                "total_count": 0,
                "total_amount": 0,
                "breakdown": {1: 0, 2: 0, 5: 0, 10: 0, 20: 0},
                "pixels_per_mm": self.default_ppm
            }

        # Step 1: Feature Extraction (Geometry + Color)
        candidates = []
        for idx, (x, y, r) in enumerate(raw_circles):
            color_info = self.analyze_coin_color(image_bgr, int(x), int(y), int(r))
            candidates.append({
                "id": idx + 1,
                "center": (int(x), int(y)),
                "radius": float(r),
                "diameter_px": float(2.0 * r),
                "color_info": color_info
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
            denom, conf = self.classify_coin(diameter_mm, coin["color_info"])

            spec = self.coin_specs.get(denom, self.coin_specs[1])
            breakdown[denom] = breakdown.get(denom, 0) + 1
            total_amount += denom

            coin.update({
                "diameter_mm": round(diameter_mm, 2),
                "denomination": denom,
                "denomination_str": spec.symbol,
                "coin_name": spec.name,
                "confidence": conf,
                "color_tone": coin["color_info"]["tone"].capitalize(),
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
