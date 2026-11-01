"""
Automated Unit and Integration Tests for Coin Detection and Counting System.
Validates OpenCV pipelines, classification logic, and Flask API endpoints
including tests on real-world photos with background wood grain textures.
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
        cls.real_path = os.path.join("static", "samples", "sample_real_user.jpg")
        cls.mix_path = os.path.join("static", "samples", "sample_mix.jpg")
        cls.scattered_path = os.path.join("static", "samples", "sample_scattered.jpg")

        assert os.path.exists(cls.real_path), f"Sample file not found at {cls.real_path}"
        assert os.path.exists(cls.mix_path), f"Sample file not found at {cls.mix_path}"
        assert os.path.exists(cls.scattered_path), f"Sample file not found at {cls.scattered_path}"

        cls.sample_real_image = cv2.imread(cls.real_path)
        cls.sample_mix_image = cv2.imread(cls.mix_path)
        cls.sample_scattered_image = cv2.imread(cls.scattered_path)

        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_real_world_user_image(self):
        """Verifies 100% accuracy on real-world photo with wood grain texture (6 coins = Rs 17)."""
        detector = CoinDetector()
        circles, stages = detector.detect_coins(self.sample_real_image)

        self.assertEqual(len(circles), 6, f"Expected 6 coins on user photo, got {len(circles)}")

        classifier = CoinClassifier()
        coins, summary = classifier.process_and_classify_all(self.sample_real_image, circles)

        self.assertEqual(summary["total_count"], 6)
        self.assertEqual(summary["total_amount"], 17)
        self.assertEqual(summary["breakdown"][1], 3)
        self.assertEqual(summary["breakdown"][2], 2)
        self.assertEqual(summary["breakdown"][10], 1)

    def test_preprocess_pipeline(self):
        """Verifies preprocessing generates proper Grayscale and Normalized arrays."""
        detector = CoinDetector()
        gray = cv2.cvtColor(self.sample_mix_image, cv2.COLOR_BGR2GRAY)
        norm, is_bright = detector.normalize_illumination(gray)

        self.assertEqual(norm.shape, gray.shape)
        self.assertEqual(norm.dtype, np.uint8)

    def test_hough_detection_mix(self):
        """Verifies that coins in sample_mix.jpg are detected."""
        detector = CoinDetector()
        circles, stages = detector.detect_coins(self.sample_mix_image)

        self.assertGreaterEqual(len(circles), 5)
        self.assertIn("edges", stages)
        self.assertIn("normalized", stages)

    def test_classification_and_counting_mix(self):
        """Verifies denomination classification and aggregation on sample_mix.jpg."""
        detector = CoinDetector()
        circles, _ = detector.detect_coins(self.sample_mix_image)

        classifier = CoinClassifier()
        coins, summary = classifier.process_and_classify_all(self.sample_mix_image, circles)

        self.assertGreaterEqual(summary["total_count"], 5)
        self.assertGreater(summary["total_amount"], 0)
        self.assertGreater(summary["pixels_per_mm"], 1.0)

        for c in coins:
            self.assertIn("center", c)
            self.assertIn("radius", c)
            self.assertIn("diameter_mm", c)
            self.assertIn("denomination", c)
            self.assertIn(c["denomination"], [1, 2, 5, 10, 20])

    def test_scattered_coins_detection(self):
        """Verifies detection on scattered coins dataset."""
        detector = CoinDetector()
        circles, _ = detector.detect_coins(self.sample_scattered_image)
        self.assertGreaterEqual(len(circles), 8)

    def test_visualizer_annotation(self):
        """Verifies that annotated image and 6-stage montage render correctly."""
        detector = CoinDetector()
        circles, stages = detector.detect_coins(self.sample_real_image)
        classifier = CoinClassifier()
        coins, summary = classifier.process_and_classify_all(self.sample_real_image, circles)

        annotated = CoinVisualizer.annotate_frame(self.sample_real_image, coins, summary)
        self.assertEqual(annotated.shape, self.sample_real_image.shape)

        montage = CoinVisualizer.create_pipeline_montage(stages, annotated)
        self.assertEqual(len(montage.shape), 3)
        self.assertGreater(montage.shape[0], 0)
        self.assertGreater(montage.shape[1], 0)

    def test_api_index(self):
        """Tests the home page route."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CoinDetector CV", response.data)

    def test_api_load_sample(self):
        """Tests loading the user sample via API."""
        response = self.client.get("/api/sample/sample_real_user.jpg")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["coin_count"], 6)
        self.assertEqual(data["total_amount"], 17)
        self.assertIn("annotated_image", data)

    def test_api_process_frame_missing_payload(self):
        """Tests error handling when payload is missing."""
        response = self.client.post("/api/process_frame", json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])

if __name__ == "__main__":
    unittest.main()
