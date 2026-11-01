"""
Core computer vision modules for coin detection and counting.
"""

from core.coin_detector import CoinDetector
from core.classifier import CoinClassifier
from core.visualizer import CoinVisualizer
from core.config import COIN_SPECS, DEFAULT_CV_PARAMS

__all__ = [
    "CoinDetector",
    "CoinClassifier",
    "CoinVisualizer",
    "COIN_SPECS",
    "DEFAULT_CV_PARAMS"
]
