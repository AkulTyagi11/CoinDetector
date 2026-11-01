"""
Standalone OpenCV Desktop Application with Live Trackbars.
Runs directly with OpenCV GUI windows (cv2.imshow) on webcam feeds or static images.

Usage:
    python cli.py                       # Runs on default webcam 0
    python cli.py --camera 1            # Runs on external USB camera 1
    python cli.py --image path/to/img   # Runs on a static image file
"""

import sys
import argparse
import cv2
import numpy as np

from core.coin_detector import CoinDetector
from core.classifier import CoinClassifier
from core.visualizer import CoinVisualizer
from core.config import DEFAULT_CV_PARAMS

def nothing(x):
    pass

def main():
    parser = argparse.ArgumentParser(description="Automated Coin Detection and Counting (OpenCV CLI/Desktop)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--image", type=str, default="", help="Path to static image (optional)")
    parser.add_argument("--method", type=str, default="hybrid", choices=["hough", "contour", "hybrid"], help="Detection method")
    args = parser.parse_args()

    window_name = "Automated Coin Detector & Counter (Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    # Trackbars for real-time OpenCV parameter tuning
    cv2.createTrackbar("Param2 (Accum)", window_name, DEFAULT_CV_PARAMS.get("hough_param2", 38), 80, nothing)
    cv2.createTrackbar("Min Radius", window_name, DEFAULT_CV_PARAMS.get("min_radius", 20), 100, nothing)
    cv2.createTrackbar("Max Radius", window_name, DEFAULT_CV_PARAMS.get("max_radius", 95), 200, nothing)
    cv2.createTrackbar("Canny High", window_name, DEFAULT_CV_PARAMS.get("canny_high", 110), 255, nothing)

    # Initialize CV components
    classifier = CoinClassifier()
    current_method = args.method

    is_static = bool(args.image)
    if is_static:
        static_img = cv2.imread(args.image)
        if static_img is None:
            print(f"Error: Could not open image file: {args.image}")
            sys.exit(1)
        cap = None
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Warning: Could not open camera {args.camera}. Falling back to sample image.")
            sample_fallback = "static/samples/sample_mix.jpg"
            static_img = cv2.imread(sample_fallback)
            is_static = True
            cap = None

    print("\n" + "=" * 65)
    print(" Automated Coin Detection and Counting (OpenCV Desktop)")
    print("=" * 65)
    print(" Interactive Controls:")
    print("   [Q] / [ESC] : Quit application")
    print("   [S]         : Save annotated frame to disk")
    print("   [M]         : Toggle method (Hough -> Contour -> Hybrid)")
    print("   [P]         : Toggle 6-stage CV Pipeline Montage view")
    print("=" * 65 + "\n")

    show_montage = False
    methods_cycle = ["hybrid", "hough", "contour"]
    method_idx = methods_cycle.index(current_method)

    while True:
        if is_static:
            frame = static_img.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

        # Read current trackbar values
        param2 = max(10, cv2.getTrackbarPos("Param2 (Accum)", window_name))
        min_r = max(5, cv2.getTrackbarPos("Min Radius", window_name))
        max_r = max(min_r + 5, cv2.getTrackbarPos("Max Radius", window_name))
        canny_h = max(20, cv2.getTrackbarPos("Canny High", window_name))

        runtime_params = dict(DEFAULT_CV_PARAMS)
        runtime_params["hough_param2"] = param2
        runtime_params["min_radius"] = min_r
        runtime_params["max_radius"] = max_r
        runtime_params["canny_high"] = canny_h

        detector = CoinDetector(runtime_params)

        # 1. Detect
        circles, stages = detector.detect_coins(frame, method=current_method)

        # 2. Classify
        coins, summary = classifier.process_and_classify_all(frame, circles)

        # 3. Visualize
        annotated = CoinVisualizer.annotate_frame(frame, coins, summary, show_hud=True)

        # Overlay active method text
        method_badge = f"Method: {current_method.upper()}"
        cv2.putText(annotated, method_badge, (annotated.shape[1] - 220, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

        if show_montage:
            display_view = CoinVisualizer.create_pipeline_montage(stages, annotated)
        else:
            display_view = annotated

        cv2.imshow(window_name, display_view)

        key = cv2.waitKey(30 if not is_static else 100) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key in [ord('s'), ord('S')]:
            save_name = f"detected_coins_result.jpg"
            cv2.imwrite(save_name, annotated)
            print(f"[Saved] Annotated frame saved to {save_name}")
        elif key in [ord('m'), ord('M')]:
            method_idx = (method_idx + 1) % len(methods_cycle)
            current_method = methods_cycle[method_idx]
            print(f"[Switched] Detection method: {current_method}")
        elif key in [ord('p'), ord('P')]:
            show_montage = not show_montage
            print(f"[Montage] 6-Stage Montage View: {'ON' if show_montage else 'OFF'}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
