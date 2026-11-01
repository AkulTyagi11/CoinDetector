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

# Official RBI Specifications for Indian Circulating Coins (2007, 2011 & 2019 Series)
COIN_SPECS: Dict[int, CoinSpec] = {
    1: CoinSpec(
        denomination=1,
        name="1 Rupee",
        symbol="₹1",
        diameter_mm=20.0,
        thickness_mm=1.45,
        color_type="silver",
        material="Ferritic Stainless Steel",
        display_color_bgr=(255, 191, 0),     # Cyan / Sky Blue
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
        display_color_bgr=(238, 104, 123),   # Medium slate blue / Indigo
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
    # Detection Engine
    "method": "adaptive",             # 'adaptive' (Morphological + Edge Coherence), 'hough', 'watershed'
    
    # Illumination Normalization
    "enable_bg_normalization": True,
    "bg_kernel_ratio": 0.13,          # Ratio of max dimension for morphological structuring element
    "blur_ksize": 9,
    
    # Shape & Contour Filtering
    "circularity_threshold": 0.65,    # 4 * pi * Area / Perimeter^2
    "solidity_threshold": 0.82,       # Area / Convex Hull Area
    "area_ratio_threshold": 0.65,     # Area / Enclosing Circle Area
    "min_area_ratio": 0.0018,         # Min area as fraction of image size
    "max_area_ratio": 0.15,           # Max area as fraction of image size
    
    # Circumferential Edge Coherence
    "enable_edge_coherence": True,
    "edge_coherence_threshold": 0.48, # Fraction of perimeter samples with strong gradient

    # Hough Circles (when enabled or guiding)
    "hough_dp": 1.2,
    "hough_min_dist": 45,
    "hough_param1": 70,
    "hough_param2": 38,
    "min_radius": 20,
    "max_radius": 140,
    "canny_high": 110,

    # Calibration & Denomination
    "calibration_mode": "auto",       # 'auto', 'reference_coin', 'manual_ppm'
    "reference_denomination": 10,     # Default reference coin (₹10 = 27mm)
    "pixels_per_mm": 4.5,
    "texture_split_threshold": 680.0  # Laplacian variance separating Hastamudra Rs.1 vs Rs.2
}
