"""
Automated Unit and Integration Tests for Coin Detection and Counting System.
Validates OpenCV pipelines, classification logic, and Flask API endpoints
using Python's standard library unittest module.
"""

import os
import cv2
import unittest
import numpy as np

from app import app
from core.coin_detector import CoinDetector
from core.classifier import CoinClassifier
from core.visualizer import CoinVisualizer
from core.config import COIN_SPECS, DEFAULT_CV_PARAMS

class TestCoinDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mix_path = os.path.join("static", "samples", "sample_mix.jpg")
        cls.touching_path = os.path.join("static", "samples", "sample_touching.jpg")
        assert os.path.exists(cls.mix_path), f"Sample file not found at {cls.mix_path}"
        assert os.path.exists(cls.touching_path), f"Sample file not found at {cls.touching_path}"
        cls.sample_mix_image = cv2.imread(cls.mix_path)
        cls.sample_touching_image = cv2.imread(cls.touching_path)

        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_preprocess_pipeline(self):
        """Verifies preprocessing generates proper Grayscale, CLAHE, and Blurred arrays."""
        detector = CoinDetector()
        gray, enhanced, blurred = detector.preprocess(self.sample_mix_image)

        self.assertEqual(gray.shape, self.sample_mix_image.shape[:2])
        self.assertEqual(enhanced.shape, gray.shape)
        self.assertEqual(blurred.shape, gray.shape)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(blurred.dtype, np.uint8)

    def test_hough_detection_mix(self):
        """Verifies that Hough Circle transform accurately detects coins in sample_mix.jpg."""
        detector = CoinDetector()
        circles, stages = detector.detect_coins(self.sample_mix_image, method="hough")

        self.assertGreaterEqual(len(circles), 5)
        self.assertIn("edges", stages)
        self.assertIn("blurred", stages)

    def test_classification_and_counting(self):
        """Verifies denomination classification, scale calibration, and total value aggregation."""
        detector = CoinDetector()
        circles, _ = detector.detect_coins(self.sample_mix_image, method="hough")

        classifier = CoinClassifier()
        coins, summary = classifier.process_and_classify_all(self.sample_mix_image, circles)

        self.assertGreaterEqual(summary["total_count"], 5)
        self.assertGreater(summary["total_amount"], 0)
        self.assertGreater(summary["pixels_per_mm"], 1.0)

        for denom in [1, 2, 5, 10]:
            self.assertIn(denom, summary["breakdown"])

        for c in coins:
            self.assertIn("center", c)
            self.assertIn("radius", c)
            self.assertIn("diameter_mm", c)
            self.assertIn("denomination", c)
            self.assertIn("confidence", c)
            self.assertIn(c["denomination"], [1, 2, 5, 10, 20])

    def test_touching_coins_watershed(self):
        """Verifies that touching coins are detected and separated."""
        detector = CoinDetector()
        circles, _ = detector.detect_coins(self.sample_touching_image, method="hybrid")
        self.assertGreaterEqual(len(circles), 4)

    def test_visualizer_annotation(self):
        """Verifies that annotated image and 6-stage montage render correctly."""
        detector = CoinDetector()
        circles, stages = detector.detect_coins(self.sample_mix_image, method="hough")
        classifier = CoinClassifier()
        coins, summary = classifier.process_and_classify_all(self.sample_mix_image, circles)

        annotated = CoinVisualizer.annotate_frame(self.sample_mix_image, coins, summary)
        self.assertEqual(annotated.shape, self.sample_mix_image.shape)

        montage = CoinVisualizer.create_pipeline_montage(stages, annotated)
        self.assertEqual(len(montage.shape), 3)
        self.assertGreater(montage.shape[0], 0)
        self.assertGreater(montage.shape[1], 0)

    def test_api_index(self):
        """Tests the home page route."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CoinDetector CV", response.data)

    def test_api_system_info(self):
        """Tests the system info diagnostic route."""
        response = self.client.get("/api/system_info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("opencv_version", data)
        self.assertIn("coin_specs", data)
        self.assertIn("10", data["coin_specs"])

    def test_api_load_sample(self):
        """Tests loading a bundled sample via API."""
        response = self.client.get("/api/sample/sample_mix.jpg")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["coin_count"], 5)
        self.assertGreater(data["total_amount"], 0)
        self.assertIn("annotated_image", data)
        self.assertIn("stages", data)

    def test_api_process_frame_missing_payload(self):
        """Tests error handling when payload is missing."""
        response = self.client.post("/api/process_frame", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])

if __name__ == "__main__":
    unittest.main()
