"""
Visualization and Annotation Utilities for Coin Detection.
Draws high-contrast bounding circles, denomination tags, HUD summaries,
and builds multi-stage pipeline montages for CV presentations and reports.
"""

import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Tuple, Optional

class CoinVisualizer:
    @staticmethod
    def draw_rounded_rectangle(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = -1,
        radius: int = 5
    ) -> None:
        """Draws a rectangle with rounded corners on the image."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

        if thickness < 0:
            # Filled rounded rectangle
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        else:
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    @classmethod
    def annotate_frame(
        cls,
        image_bgr: np.ndarray,
        coins: List[Dict[str, Any]],
        summary: Dict[str, Any],
        show_hud: bool = True
    ) -> np.ndarray:
        """
        Draws circles, labels, and an informative HUD banner on the image.
        """
        annotated = image_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw each detected coin
        for coin in coins:
            cx, cy = coin["center"]
            r = int(coin["radius"])
            color_bgr = coin.get("display_color_bgr", (0, 255, 0))
            denom_str = coin.get("denomination_str", "Coin")
            diam_mm = coin.get("diameter_mm", 0.0)
            cid = coin.get("id", "")

            # 1. Outer circle
            cv2.circle(annotated, (cx, cy), r, color_bgr, 3, cv2.LINE_AA)

            # 2. Outer glow / highlight ring
            cv2.circle(annotated, (cx, cy), r + 2, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Center point
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
            cv2.line(annotated, (cx - 6, cy), (cx + 6, cy), (0, 0, 255), 1)
            cv2.line(annotated, (cx, cy - 6), (cx, cy + 6), (0, 0, 255), 1)

            # 4. Denomination badge (above coin)
            label = f"#{cid} {denom_str}"
            sublabel = f"{diam_mm}mm" if diam_mm > 0 else f"r={r}px"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.45, min(0.65, r / 60.0))
            thickness = 1

            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (stw, sth), _ = cv2.getTextSize(sublabel, font, font_scale * 0.8, thickness)

            badge_w = max(tw, stw) + 16
            badge_h = th + sth + 12

            # Position badge
            bx1 = cx - badge_w // 2
            by1 = cy - r - badge_h - 6
            if by1 < 10:  # If near top edge, place inside coin
                by1 = cy - badge_h // 2
            bx2 = bx1 + badge_w
            by2 = by1 + badge_h

            # Draw semi-transparent dark pill background
            overlay = annotated.copy()
            cls.draw_rounded_rectangle(overlay, (bx1, by1), (bx2, by2), (20, 24, 33), -1, radius=6)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # Border for badge in denomination color
            cls.draw_rounded_rectangle(annotated, (bx1, by1), (bx2, by2), color_bgr, 1, radius=6)

            # Text
            text_x = bx1 + 8
            text_y1 = by1 + th + 3
            text_y2 = text_y1 + sth + 4

            cv2.putText(annotated, label, (text_x, text_y1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(annotated, sublabel, (text_x, text_y2), font, font_scale * 0.8, (200, 220, 240), thickness, cv2.LINE_AA)

        # Draw HUD Summary Banner at Top if requested
        if show_hud and w >= 400:
            total_count = summary.get("total_count", 0)
            total_amt = summary.get("total_amount", 0)
            breakdown = summary.get("breakdown", {})
            ppm = summary.get("pixels_per_mm", 0.0)

            hud_h = 58
            hud_overlay = annotated.copy()
            cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (15, 20, 28), -1)
            cv2.addWeighted(hud_overlay, 0.88, annotated, 0.12, 0, annotated)
            cv2.line(annotated, (0, hud_h), (w, hud_h), (0, 215, 255), 2)

            # Left stats: Total Coins & Total Amount
            cv2.putText(
                annotated,
                f"Coins: {total_count}  |  Total: Rs.{total_amt}",
                (16, 26),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (0, 225, 255),
                1,
                cv2.LINE_AA
            )

            # Breakdown text
            breakdown_parts = []
            for d in [10, 5, 2, 1]:
                c = breakdown.get(d, 0)
                if c > 0:
                    breakdown_parts.append(f"Rs.{d}x{c}")
            breakdown_str = "  ".join(breakdown_parts) if breakdown_parts else "No coins"

            cv2.putText(
                annotated,
                f"Breakdown: {breakdown_str}",
                (16, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 225, 230),
                1,
                cv2.LINE_AA
            )

            # Right stats: Scale
            scale_text = f"Scale: {ppm:.1f} px/mm"
            (stw, _), _ = cv2.getTextSize(scale_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(
                annotated,
                scale_text,
                (w - stw - 16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 180, 200),
                1,
                cv2.LINE_AA
            )

        return annotated

    @classmethod
    def create_pipeline_montage(
        cls,
        stages: Dict[str, np.ndarray],
        annotated: np.ndarray,
        tile_size: Tuple[int, int] = (400, 300)
    ) -> np.ndarray:
        """
        Creates a clean 2x3 academic montage showing all 6 steps of the Computer Vision pipeline:
        1. Original Input
        2. Grayscale & Contrast
        3. Gaussian / Bilateral Filter
        4. Canny Edge Map
        5. Binary / Distance Transform
        6. Final Annotated Detection
        """
        tw, th = tile_size

        def prepare_tile(img: np.ndarray, title: str) -> np.ndarray:
            if len(img.shape) == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img.copy()

            resized = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)

            # Title bar on top
            cv2.rectangle(resized, (0, 0), (tw, 28), (24, 28, 38), -1)
            cv2.putText(
                resized,
                title,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 215, 255),
                1,
                cv2.LINE_AA
            )
            # Border
            cv2.rectangle(resized, (0, 0), (tw - 1, th - 1), (60, 70, 85), 1)
            return resized

        t1 = prepare_tile(stages.get("original", annotated), "1. Original Frame")
        t2 = prepare_tile(stages.get("enhanced", stages.get("grayscale", annotated)), "2. Grayscale & CLAHE")
        t3 = prepare_tile(stages.get("blurred", annotated), "3. Bilateral/Gaussian Filter")
        t4 = prepare_tile(stages.get("edges", annotated), "4. Canny Edge Detection")
        t5 = prepare_tile(stages.get("distance_transform", stages.get("binary_thresh", annotated)), "5. Watershed / Distance Map")
        t6 = prepare_tile(annotated, "6. Detection & Classification")

        row1 = np.hstack([t1, t2, t3])
        row2 = np.hstack([t4, t5, t6])
        montage = np.vstack([row1, row2])

        return montage

    @staticmethod
    def encode_image_base64(image_bgr: np.ndarray, quality: int = 85) -> str:
        """Encodes an OpenCV BGR image into a Base64 JPEG string for web transfer."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image_bgr, encode_params)
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")
