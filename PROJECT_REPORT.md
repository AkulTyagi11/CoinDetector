# Academic Project Report
## Automated Coin Detection and Counting Using OpenCV

**Course:** Computer Vision / Digital Image Processing  
**Project Title:** Automated Coin Detection and Counting Using OpenCV  
**Repository:** [https://github.com/AkulTyagi11/CoinDetector_CV_Project.git](https://github.com/AkulTyagi11/CoinDetector_CV_Project.git)  

---

### Abstract
Automated physical currency detection and counting is a classic and practical problem in Computer Vision and Digital Image Processing. This project presents an end-to-end automated system capable of detecting, segmenting, classifying, and counting Indian Rupee (INR) coins (₹1, ₹2, ₹5, ₹10, and ₹20) from live camera video streams and static photographic captures. The pipeline integrates Contrast-Limited Adaptive Histogram Equalization (CLAHE), Bilateral filtering, Canny edge detection, Hough Circle Transform, Marker-Controlled Watershed Segmentation, and multimodal feature reasoning combining geometric diameter ratios and HSV color-space analysis. The system is packaged with a high-performance Flask WebRTC web application, a standalone OpenCV desktop application with real-time GUI trackbars, and an educational 6-stage pipeline inspector for academic viva demonstrations.

---

### 1. Problem Statement & Motivation
Manual coin counting and sorting in banking, retail counters, vending machines, and transportation ticketing is labor-intensive, error-prone, and slow. While specialized hardware sorters exist, vision-based software systems offer a cost-effective, non-contact alternative using ubiquitous camera sensors (webcams, smartphone cameras).

Key challenges in vision-based coin detection include:
1. **Specular Reflection and Glare:** Metallic surfaces reflect direct illumination, causing intense highlights and shadows that disrupt traditional thresholding.
2. **Touching and Clustered Coins:** Coins touching edge-to-edge merge into single connected components under standard contour analysis.
3. **Scale Ambiguity:** The pixel radius of a coin depends inversely on the distance $Z$ from the camera lens to the surface ($r_{px} = f \cdot \frac{R_{world}}{Z}$).
4. **Denomination Similarity:** Coins of different denominations may have similar dimensions (e.g. ₹5 at 23 mm vs ₹2 at 25 mm vs ₹1 at 20 mm).

---

### 2. Physical Specifications of Indian Rupee (INR) Coins
The system is calibrated according to the official Reserve Bank of India (RBI) physical standards:

| Denomination | Diameter (mm) | Thickness (mm) | Weight (g) | Material Composition | Color & Visual Profile |
|:---:|:---:|:---:|:---:|:---|:---|
| **₹1** | 20.0 mm | 1.45 mm | 3.79 g | Ferritic Stainless Steel | Monometallic Silver / Grey |
| **₹2** | 25.0 mm | 1.56 mm | 5.62 g | Ferritic Stainless Steel | Monometallic Silver / Grey |
| **₹5** | 23.0 mm | 1.90 mm | 6.00 g | Nickel-Brass | Monometallic Golden / Yellow |
| **₹10** | 27.0 mm | 1.80 mm | 7.74 g | Bimetallic (Al-Bronze rim, Cu-Ni core) | Outer Golden Ring, Inner Silver Core |
| **₹20** | 27.0 mm | 1.90 mm | 8.54 g | Bimetallic (12-sided polygon) | Outer Golden Rim, Inner Silver Core |

---

### 3. Proposed Methodology & Mathematical Foundations

```
+-------------------------------------------------------------------------------+
|                             Processing Pipeline                               |
|                                                                               |
|  [Input Frame] ---> [Grayscale & CLAHE] ---> [Bilateral / Gaussian Filter]    |
|                                                         |                     |
|                                                         v                     |
|                   +-------------------------------------+                     |
|                   |                                     |                     |
|                   v                                     v                     |
|         [Hough Circle Transform]           [Otsu / Adaptive Threshold]        |
|         - Gradient Vector Voting           - Morphological Opening            |
|         - Accumulator Peak Search          - Distance Transform               |
|                   |                        - Watershed Marker Segmentation    |
|                   |                        - Circularity Filtering (4*pi*A/P2)|
|                   +------------------+------------------+                     |
|                                      v                                        |
|                       [Non-Maximum Suppression (NMS)]                         |
|                                      |                                        |
|                                      v                                        |
|                      [HSV Color & Bimetallic Feature]                         |
|                      - Inner Core (0 to 0.45r)                                |
|                      - Outer Rim  (0.65 to 0.90r)                             |
|                                      |                                        |
|                                      v                                        |
|                     [PPM Calibration & Denomination]                          |
|                     - Size Matching (mm)                                      |
|                     - Color Confirmation                                      |
|                                      |                                        |
|                                      v                                        |
|                   [Annotated Frame & Value Aggregation]                       |
+-------------------------------------------------------------------------------+
```

#### 3.1 Preprocessing & Contrast Normalization
1. **Grayscale Conversion:**
   $$I_{gray}(x, y) = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE):**
   Instead of global equalization which over-amplifies specular glare, CLAHE operates on localized $8 \times 8$ pixel tiles with a clip limit $\beta = 2.0$. It redistributes pixel intensity histograms exceeding the threshold evenly across bins before bilinear interpolation across tile boundaries.
3. **Bilateral Filtering:**
   Preserves coin outer circular edges while smoothing internal textures and engravings:
   $$I_{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i) - I(x)\|) g_s(\|x_i - x\|)$$
   where $g_s$ is the spatial Gaussian and $f_r$ is the radiometric (range) Gaussian.

#### 3.2 Circle Detection Algorithms
##### A. Hough Circle Transform (Gradient Method)
The parametric equation of a circle is:
$$(x - a)^2 + (y - b)^2 = r^2$$
To avoid high memory overhead of a 3D accumulator $(a, b, r)$, OpenCV utilizes the **2-1 Hough Gradient Method**:
1. First, compute first-order image gradients $\nabla I = \left(\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right)$ via Sobel operators.
2. For each edge pixel detected by Canny, increment accumulator cells along the gradient line.
3. Local maxima in the 2D accumulator identify candidate circle centers $(a, b)$.
4. For each center candidate, compute distances to edge points to determine the optimal radius $r$.

##### B. Marker-Controlled Watershed Segmentation
For touching or overlapping coins:
1. Compute binary foreground mask $B(x, y)$ via Otsu / Adaptive thresholding.
2. Apply Euclidean Distance Transform:
   $$D(p) = \min_{q \in \text{Background}} \|p - q\|_2$$
3. Threshold the distance transform at $0.45 \cdot \max(D)$ to obtain distinct, disconnected seed markers for each coin core.
4. Apply the topological watershed simulation where pixel values represent elevation, separating touching coins along ridge lines.
5. Filter segmented contours via the **isoperimetric circularity quotient**:
   $$C = \frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2}$$
   Coins satisfy $C \ge 0.65$ (a perfect circle has $C = 1.0$).

##### C. Circle Non-Maximum Suppression (NMS)
To eliminate duplicate concentric detections (e.g., inner engravings or outer rim echoes):
$$\text{Distance}(c_1, c_2) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} < 0.55 \cdot \max(r_1, r_2)$$
Candidate circles failing this criterion are merged/suppressed.

#### 3.3 Multimodal Denomination Classification
Classification utilizes two orthogonal visual feature spaces:

1. **Physical Diameter Estimation via Scale Calibration:**
   $$\text{PPM} = \frac{2 \cdot r_{ref\_px}}{\text{Diameter}_{ref\_mm}}$$
   $$\text{Diameter}_{i\_mm} = \frac{2 \cdot r_{i\_px}}{\text{PPM}}$$
2. **Radial Color-Space Analysis (HSV):**
   Within each detected circle $(x_c, y_c, r)$, two concentric masks are sampled:
   - **Inner Core:** $\rho \in [0.05 \cdot r, 0.48 \cdot r]$
   - **Outer Rim:** $\rho \in [0.65 \cdot r, 0.92 \cdot r]$
   - **Bimetallic Signature (₹10 / ₹20):** High contrast between outer rim saturation ($S_{outer} > 75$, golden) and inner core saturation ($S_{inner} < 50$, silver), with $|S_{outer} - S_{inner}| > 45$.
   - **Golden Signature (₹5):** Mean hue $H \in [10, 42]$ and mean saturation $S \ge 40$ across both inner and outer regions.
   - **Silver Signature (₹1 / ₹2):** Low saturation $S < 35$ across both regions. Tie-breaker between ₹1 (20 mm) and ₹2 (25 mm) is resolved by physical diameter matching.

---

### 4. Software Architecture & Implementation

#### 4.1 Modular Directory Structure
```
CoinDetector_CV_Project/
├── app.py                      # Flask web application & REST API
├── cli.py                      # Standalone OpenCV desktop GUI with trackbars
├── generate_samples.py        # Synthetic test coin generator (accurate mm/px)
├── requirements.txt           # Python dependencies
├── core/
│   ├── config.py              # RBI coin specs, physical dimensions, default CV params
│   ├── coin_detector.py       # Preprocessing, Hough, Contours, Watershed, NMS
│   ├── classifier.py          # Radial color analysis, PPM scale, denomination rules
│   └── visualizer.py          # HUD overlays, rounded pill labels, 6-stage montage
├── static/
│   ├── css/style.css          # Glassmorphism dark/light responsive styling
│   ├── js/main.js             # WebRTC video capture, dynamic controls, Web Audio API
│   └── samples/               # Preloaded realistic test images
├── templates/
│   └── index.html             # Single-page web dashboard & pipeline inspector
└── tests/
    └── test_coin_detector.py  # Automated unit and integration test suite
```

#### 4.2 Application Modes
1. **Live WebRTC Webcam Mode:** Streams client camera frames directly from the browser (`navigator.mediaDevices.getUserMedia`) to the Flask backend with throttled FPS for smooth real-time response.
2. **Static Photo Upload Mode:** Drag-and-drop any user photo (JPG, PNG, WEBP) to inspect detection, itemized breakdown, and export CSV.
3. **Preloaded Test Samples:** Instant 1-click test suites (`sample_mix.jpg`, `sample_scattered.jpg`, `sample_touching.jpg`).
4. **Backend Camera Stream (`/video_feed`):** Native MJPEG stream using local OpenCV `cv2.VideoCapture(0)`.
5. **Interactive Trackbar Desktop App (`cli.py`):** Standalone OpenCV window with live sliders for parameter tuning (`cv2.createTrackbar`).

---

### 5. Experimental Results & Performance

| Test Dataset | Ground Truth Coins | Detected Coins | Ground Truth Total | Calculated Total | Accuracy (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Real-World Photo (`sample_real_user.jpg` on wood table)** | **6 coins (3x ₹1, 2x ₹2, 1x ₹10)** | **6** | **₹17** | **₹17** | **100%** |
| `sample_mix.jpg` | 6 coins (2x ₹10, 2x ₹5, 1x ₹2, 1x ₹1) | 6 | ₹33 | ₹33 | **100%** |
| `sample_scattered.jpg` | 9 coins (2x ₹10, 2x ₹5, 2x ₹2, 3x ₹1) | 9 | ₹37 | ₹37 | **100%** |
| `sample_touching.jpg` | 5 coins (1x ₹10, 2x ₹5, 1x ₹2, 1x ₹1) | 5 | ₹23 | ₹23 | **100%** |

- **Average Processing Latency per Frame:** 18 ms - 35 ms on standard CPU (equivalent to 28-55 FPS).
- **Unit Test Coverage:** 9/9 passing integration and unit tests covering preprocessing, Hough detection, Watershed segmentation, classification, HUD annotation, and REST API endpoints.

---

### 6. Frequently Asked Viva / Oral Examination Questions

1. **Why use CLAHE instead of standard Histogram Equalization?**
   *Standard histogram equalization uses global contrast modification, which heavily amplifies specular reflections on shiny metallic surfaces, washing out coin edges. CLAHE operates on localized contextual tiles and limits contrast amplification via a clipping threshold.*

2. **How does the 2-1 Hough Gradient method differ from the standard Hough Transform?**
   *The standard Hough circle transform requires a 3D accumulator array $(a, b, r)$, leading to high computational and memory complexity $O(N^3)$. The 2-1 Gradient method uses edge gradient direction vectors to constrain circle centers to a 2D accumulator array $(a, b)$, and then calculates radii in a 1D step.*

3. **How does the system separate touching coins?**
   *Touching coins form merged contours. The system applies the Euclidean Distance Transform on the thresholded binary mask to identify localized peaks as foreground markers, and then executes Marker-Controlled Watershed Segmentation to delineate the contact boundaries.*

4. **Why is color alone insufficient for coin classification?**
   *Both ₹1 and ₹2 coins are minted from identical ferritic stainless steel (silver/grey). Similarly, older ₹5 coins and certain commemorative ₹10 coins share metallic compositions. Combining physical diameter ratios with radial HSV inspection ensures unambiguous classification.*

---

### 7. Conclusion
The "Automated Coin Detection and Counting Using OpenCV" project delivers a robust, real-time Computer Vision solution capable of detecting, classifying, and tallying Indian currency coins. By combining classical image processing algorithms (CLAHE, Bilateral filtering, Hough Circle Transform, Watershed segmentation) with multimodal physical reasoning and an interactive web interface, this project bridges theoretical computer vision concepts and practical software engineering.
