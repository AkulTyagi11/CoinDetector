"""
Configuration and Physical Specifications for Indian Rupee (INR) Coins
and OpenCV Image Processing Pipelines.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class CoinSpec:
    denomination: int
    name: str
    symbol: str
    diameter_mm: float
    thickness_mm: float
    color_type: str  # 'silver', 'golden', 'bimetallic'
    material: str
    display_color_bgr: Tuple[int, int, int]  # BGR for OpenCV drawing
    display_color_hex: str

# Official RBI Specifications for Indian Circulating Coins (2011 & 2019 Series)
COIN_SPECS: Dict[int, CoinSpec] = {
    1: CoinSpec(
        denomination=1,
        name="1 Rupee",
        symbol="₹1",
        diameter_mm=20.0,
        thickness_mm=1.45,
        color_type="silver",
        material="Ferritic Stainless Steel",
        display_color_bgr=(255, 191, 0),     # Deep sky blue / Cyan-ish
        display_color_hex="#00bfff"
    ),
    2: CoinSpec(
        denomination=2,
        name="2 Rupees",
        symbol="₹2",
        diameter_mm=25.0,
        thickness_mm=1.56,
        color_type="silver",
        material="Ferritic Stainless Steel",
        display_color_bgr=(238, 104, 123),   # Royal / Medium slate blue
        display_color_hex="#7b68ee"
    ),
    5: CoinSpec(
        denomination=5,
        name="5 Rupees",
        symbol="₹5",
        diameter_mm=23.0,
        thickness_mm=1.90,
        color_type="golden",
        material="Nickel-Brass / Brass",
        display_color_bgr=(0, 215, 255),     # Bright Gold / Yellow
        display_color_hex="#ffd700"
    ),
    10: CoinSpec(
        denomination=10,
        name="10 Rupees",
        symbol="₹10",
        diameter_mm=27.0,
        thickness_mm=1.80,
        color_type="bimetallic",
        material="Bimetallic (Outer: Al-Bronze, Core: Ni-Silver)",
        display_color_bgr=(42, 142, 255),    # Vibrant Orange
        display_color_hex="#ff8e2a"
    ),
    20: CoinSpec(
        denomination=20,
        name="20 Rupees",
        symbol="₹20",
        diameter_mm=27.0,
        thickness_mm=1.90,
        color_type="bimetallic",
        material="12-Sided Bimetallic",
        display_color_bgr=(180, 50, 230),    # Magenta / Violet
        display_color_hex="#e632b4"
    )
}

# Default Image Processing Parameters
DEFAULT_CV_PARAMS = {
    # Preprocessing
    "blur_method": "gaussian",        # 'gaussian', 'bilateral', 'median'
    "blur_ksize": 9,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75,
    "clahe_enabled": True,
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": 8,

    # Canny Edge Detection
    "canny_low": 35,
    "canny_high": 110,

    # Hough Circles
    "hough_dp": 1.2,
    "hough_min_dist": 50,
    "hough_param1": 75,               # Upper threshold for Canny
    "hough_param2": 38,               # Accumulator threshold (lower = more circles)
    "min_radius": 20,
    "max_radius": 95,

    # Contours & Watershed
    "min_contour_area": 500,
    "max_contour_area": 85000,
    "circularity_threshold": 0.65,    # 4 * pi * Area / Perimeter^2
    "adaptive_block_size": 25,
    "adaptive_c": 3,

    # Calibration & Denomination
    "calibration_mode": "auto",       # 'auto', 'reference_coin', 'manual_ppm'
    "reference_denomination": 10,     # Default reference coin (₹10 = 27mm)
    "pixels_per_mm": 5.0,             # Initial estimate if unknown
    "size_tolerance_ratio": 0.08      # 8% tolerance for diameter matching
}
