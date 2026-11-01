"""
Flask Web Application for Automated Coin Detection and Counting Using OpenCV.
Supports live camera processing (WebRTC / getUserMedia and OpenCV VideoCapture),
image uploads, interactive calibration, parameter tuning, and academic CV pipeline inspection.
"""

import os
import cv2
import json
import base64
from typing import Optional, Dict, Any, List
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

from core.coin_detector import CoinDetector
from core.classifier import CoinClassifier
from core.visualizer import CoinVisualizer
from core.config import COIN_SPECS, DEFAULT_CV_PARAMS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Global detector and classifier instances
classifier = CoinClassifier(default_ppm=DEFAULT_CV_PARAMS.get("pixels_per_mm", 4.5))

def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """Decodes a base64 string (with or without data URI prefix) into an OpenCV BGR image."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            try:
                from PIL import Image, ImageOps
                import io
                pil_img = Image.open(io.BytesIO(img_bytes))
                transposed = ImageOps.exif_transpose(pil_img)
                rgb = np.array(transposed)
                if len(rgb.shape) == 3 and rgb.shape[2] == 3:
                    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        return img_bgr
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def process_image_pipeline(
    image_bgr: np.ndarray,
    cv_params: dict,
    include_stages: bool = False,
    include_montage: bool = False
) -> dict:
    """Executes the full computer vision detection, classification, and visualization pipeline."""
    method = cv_params.get("method", "hybrid")
    detector = CoinDetector(cv_params)

    # 1. Detect Coins
    circles, stages = detector.detect_coins(image_bgr, method=method)

    # 2. Denomination Classification & Scale Calibration
    calib_mode = cv_params.get("calibration_mode", "auto")
    ref_denom = int(cv_params.get("reference_denomination", 10))
    manual_ppm = float(cv_params.get("manual_ppm", 0.0)) if calib_mode == "manual_ppm" else None

    # Handle manual reference coin designation (user clicked a specific coin)
    ref_coin_index = cv_params.get("reference_coin_index")
    if ref_coin_index is not None and 0 <= int(ref_coin_index) < len(circles):
        target_idx = int(ref_coin_index)
        ref_spec = COIN_SPECS.get(ref_denom, COIN_SPECS[10])
        raw_circles = list(circles)
        # Mark target coin as reference
        temp_coins = []
        for idx, c in enumerate(raw_circles):
            is_ref = (idx == target_idx)
            temp_coins.append({
                "radius": c[2],
                "is_reference": is_ref,
                "reference_diameter_mm": ref_spec.diameter_mm if is_ref else None,
                "color_info": classifier.analyze_coin_color(image_bgr, int(c[0]), int(c[1]), int(c[2]))
            })
        ppm = classifier.calibrate_ppm_from_coins(temp_coins, ref_denomination=ref_denom)
        calib_mode = "manual_ppm"
        manual_ppm = ppm

    coins, summary = classifier.process_and_classify_all(
        image_bgr,
        circles,
        calibration_mode=calib_mode,
        ref_denomination=ref_denom,
        manual_ppm=manual_ppm
    )

    # 3. Visualization and Annotation
    annotated = CoinVisualizer.annotate_frame(image_bgr, coins, summary, show_hud=True)
    annotated_b64 = CoinVisualizer.encode_image_base64(annotated)

    response_data = {
        "success": True,
        "coins": coins,
        "summary": summary,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "coin_count": summary["total_count"],
        "total_amount": summary["total_amount"],
        "breakdown": summary["breakdown"],
        "pixels_per_mm": summary["pixels_per_mm"]
    }

    if include_montage:
        montage = CoinVisualizer.create_pipeline_montage(stages, annotated)
        montage_b64 = CoinVisualizer.encode_image_base64(montage)
        response_data["montage_image"] = f"data:image/jpeg;base64,{montage_b64}"

    if include_stages:
        stages_b64 = {}
        for stage_name, stage_img in stages.items():
            stages_b64[stage_name] = f"data:image/jpeg;base64,{CoinVisualizer.encode_image_base64(stage_img)}"
        response_data["stages"] = stages_b64

    return response_data

@app.route("/")
def index():
    """Renders the main single-page web app."""
    return render_template("index.html", default_params=DEFAULT_CV_PARAMS, coin_specs=COIN_SPECS)

@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    """API endpoint for live webcam frame processing from browser client."""
    data = request.get_json(force=True, silent=True) or {}
    frame_b64 = data.get("frame")
    if not frame_b64:
        return jsonify({"success": False, "error": "No image frame provided"}), 400

    img_bgr = decode_base64_image(frame_b64)
    if img_bgr is None:
        return jsonify({"success": False, "error": "Failed to decode frame image"}), 400

    cv_params = data.get("params", {})
    include_stages = bool(data.get("include_stages", False))
    include_montage = bool(data.get("include_montage", False))

    result = process_image_pipeline(
        img_bgr,
        cv_params,
        include_stages=include_stages,
        include_montage=include_montage
    )
    return jsonify(result)

@app.route("/api/upload_image", methods=["POST"])
def upload_image():
    """API endpoint for uploading static images."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"success": False, "error": "Invalid image format"}), 400

    try:
        from PIL import Image, ImageOps
        import io
        pil_img = Image.open(io.BytesIO(file_bytes))
        transposed = ImageOps.exif_transpose(pil_img)
        rgb = np.array(transposed)
        if len(rgb.shape) == 3 and rgb.shape[2] == 3:
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    params_raw = request.form.get("params", "{}")
    try:
        cv_params = json.loads(params_raw)
    except Exception:
        cv_params = {}

    include_stages = request.form.get("include_stages", "true").lower() == "true"
    include_montage = request.form.get("include_montage", "true").lower() == "true"

    result = process_image_pipeline(
        img_bgr,
        cv_params,
        include_stages=include_stages,
        include_montage=include_montage
    )
    return jsonify(result)

@app.route("/api/sample/<filename>", methods=["GET"])
def load_sample(filename: str):
    """Loads a pre-generated sample image and returns detection results."""
    safe_filename = os.path.basename(filename)
    sample_path = os.path.join(app.root_path, "static", "samples", safe_filename)

    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": f"Sample {safe_filename} not found"}), 404

    img_bgr = cv2.imread(sample_path)
    if img_bgr is None:
        return jsonify({"success": False, "error": "Failed to read sample image"}), 500

    method = request.args.get("method", "adaptive")
    cv_params = dict(DEFAULT_CV_PARAMS)
    cv_params["method"] = method

    result = process_image_pipeline(
        img_bgr,
        cv_params,
        include_stages=True,
        include_montage=True
    )
    result["sample_filename"] = safe_filename
    return jsonify(result)

@app.route("/api/system_info", methods=["GET"])
def system_info():
    """Returns system diagnostic metadata, OpenCV version, and coin specs."""
    specs_dict = {
        k: {
            "denomination": v.denomination,
            "name": v.name,
            "symbol": v.symbol,
            "diameter_mm": v.diameter_mm,
            "material": v.material,
            "color_hex": v.display_color_hex
        }
        for k, v in COIN_SPECS.items()
    }
    return jsonify({
        "success": True,
        "opencv_version": cv2.__version__,
        "default_params": DEFAULT_CV_PARAMS,
        "coin_specs": specs_dict
    })

# Generator for direct OpenCV video feed (Backend camera mode)
def generate_camera_frames():
    cap = cv2.VideoCapture(0)
    detector = CoinDetector(DEFAULT_CV_PARAMS)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            circles, _ = detector.detect_coins(frame, method="hybrid")
            coins, summary = classifier.process_and_classify_all(frame, circles)
            annotated = CoinVisualizer.annotate_frame(frame, coins, summary, show_hud=True)

            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()

@app.route("/video_feed")
def video_feed():
    """MJPEG stream from local backend camera."""
    return Response(
        generate_camera_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print("=" * 60)
    print(" Automated Coin Detection and Counting Using OpenCV")
    print(f" Starting server on http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=debug)
