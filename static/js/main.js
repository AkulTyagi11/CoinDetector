/**
 * Automated Coin Detection and Counting Using OpenCV
 * Frontend Client Controller: WebRTC Camera, Parameter Tuning,
 * Interactive Calibration, and Real-Time Dashboard Rendering.
 */

class CoinDetectorApp {
  constructor() {
    // State
    this.activeMode = "webcam"; // 'webcam', 'upload', 'backend'
    this.isStreaming = false;
    this.stream = null;
    this.processingFrame = false;
    this.lastFrameTime = 0;
    this.targetFps = 8;
    this.currentImageB64 = null;
    this.lastCoins = [];
    this.lastSummary = null;
    this.audioContext = null;
    this.soundEnabled = true;
    this.lastCoinCount = 0;
    this.referenceCoinIndex = null;

    // DOM Elements
    this.initElements();
    this.initEventListeners();
    this.loadCameraDevices();
    this.initAudio();

    // Auto-load Real User Sample initially so user sees immediate accurate results
    this.loadSample("sample_real_user.jpg");
  }

  initElements() {
    this.webcamVideo = document.getElementById("webcamVideo");
    this.canvasFeed = document.getElementById("canvasFeed");
    this.ctx = this.canvasFeed.getContext("2d");
    this.feedPlaceholder = document.getElementById("feedPlaceholder");

    // Camera Controls
    this.btnStartCam = document.getElementById("btnStartCam");
    this.btnStopCam = document.getElementById("btnStopCam");
    this.btnSnapshot = document.getElementById("btnSnapshot");
    this.cameraSelect = document.getElementById("cameraSelect");

    // Upload & Samples
    this.fileInput = document.getElementById("fileInput");
    this.dropzone = document.getElementById("dropzone");
    this.uploadContainer = document.getElementById("uploadContainer");
    this.feedContainer = document.getElementById("feedContainer");

    // Stats Counters
    this.statTotalAmount = document.getElementById("statTotalAmount");
    this.statCoinCount = document.getElementById("statCoinCount");
    this.statPpm = document.getElementById("statPpm");
    this.coinsTableBody = document.getElementById("coinsTableBody");

    // Denominations
    this.denomCounts = {
      1: document.getElementById("count-1"),
      2: document.getElementById("count-2"),
      5: document.getElementById("count-5"),
      10: document.getElementById("count-10"),
      20: document.getElementById("count-20")
    };

    // Sliders & Controls
    this.methodSelect = document.getElementById("methodSelect");
    this.sliderParam2 = document.getElementById("sliderParam2");
    this.valParam2 = document.getElementById("valParam2");
    this.sliderMinR = document.getElementById("sliderMinR");
    this.valMinR = document.getElementById("valMinR");
    this.sliderMaxR = document.getElementById("sliderMaxR");
    this.valMaxR = document.getElementById("valMaxR");
    this.sliderCanny = document.getElementById("sliderCanny");
    this.valCanny = document.getElementById("valCanny");
    this.calibModeSelect = document.getElementById("calibModeSelect");
    this.refDenomSelect = document.getElementById("refDenomSelect");
    this.soundToggle = document.getElementById("soundToggle");

    // Pipeline Stage Containers
    this.stageOriginal = document.getElementById("stageOriginal");
    this.stageGrayscale = document.getElementById("stageGrayscale");
    this.stageBlurred = document.getElementById("stageBlurred");
    this.stageEdges = document.getElementById("stageEdges");
    this.stageDistance = document.getElementById("stageDistance");
    this.stageFinal = document.getElementById("stageFinal");
  }

  initAudio() {
    try {
      window.AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContext();
    } catch (e) {
      console.warn("Web Audio API not supported in this browser:", e);
    }
  }

  playCoinChime() {
    if (!this.soundEnabled || !this.audioContext) return;
    try {
      if (this.audioContext.state === "suspended") {
        this.audioContext.resume();
      }
      const osc = this.audioContext.createOscillator();
      const gain = this.audioContext.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(880, this.audioContext.currentTime); // A5
      osc.frequency.exponentialRampToValueAtTime(1760, this.audioContext.currentTime + 0.12);

      gain.gain.setValueAtTime(0.2, this.audioContext.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + 0.25);

      osc.connect(gain);
      gain.connect(this.audioContext.destination);

      osc.start();
      osc.stop(this.audioContext.currentTime + 0.25);
    } catch (e) {
      // Audio playback blocked by policy until user interaction
    }
  }

  initEventListeners() {
    // Mode Navigation Tabs
    document.querySelectorAll(".nav-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        this.switchMode(tab.dataset.mode);
      });
    });

    // Camera Controls
    this.btnStartCam.addEventListener("click", () => this.startWebcam());
    this.btnStopCam.addEventListener("click", () => this.stopWebcam());
    this.btnSnapshot.addEventListener("click", () => this.captureSnapshot());
    this.cameraSelect.addEventListener("change", () => {
      if (this.isStreaming) {
        this.startWebcam(this.cameraSelect.value);
      }
    });

    // Upload & Dropzone
    this.dropzone.addEventListener("click", () => this.fileInput.click());
    this.fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        this.handleFileUpload(e.target.files[0]);
      }
    });

    ["dragenter", "dragover"].forEach(eventName => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        this.dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        this.dropzone.classList.remove("dragover");
      });
    });

    this.dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length > 0) {
        this.handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    // Quick Sample Buttons
    document.querySelectorAll(".sample-pill").forEach(pill => {
      pill.addEventListener("click", () => {
        this.loadSample(pill.dataset.file);
      });
    });

    // Sliders Real-Time Update
    const bindSlider = (slider, display, key) => {
      slider.addEventListener("input", () => {
        display.textContent = slider.value;
        this.referenceCoinIndex = null; // Reset manual click calibration
        if (!this.isStreaming && this.currentImageB64) {
          this.reprocessCurrentImage();
        }
      });
    };

    bindSlider(this.sliderParam2, this.valParam2, "hough_param2");
    bindSlider(this.sliderMinR, this.valMinR, "min_radius");
    bindSlider(this.sliderMaxR, this.valMaxR, "max_radius");
    bindSlider(this.sliderCanny, this.valCanny, "canny_high");

    this.methodSelect.addEventListener("change", () => {
      if (!this.isStreaming && this.currentImageB64) {
        this.reprocessCurrentImage();
      }
    });

    this.calibModeSelect.addEventListener("change", () => {
      if (!this.isStreaming && this.currentImageB64) {
        this.reprocessCurrentImage();
      }
    });

    this.refDenomSelect.addEventListener("change", () => {
      if (!this.isStreaming && this.currentImageB64) {
        this.reprocessCurrentImage();
      }
    });

    if (this.soundToggle) {
      this.soundToggle.addEventListener("change", (e) => {
        this.soundEnabled = e.target.checked;
      });
    }

    // Interactive Click on Canvas to Calibrate Reference Coin
    this.canvasFeed.addEventListener("click", (e) => this.handleCanvasClick(e));

    // Export Buttons
    document.getElementById("btnDownloadImage")?.addEventListener("click", () => this.downloadAnnotatedImage());
    document.getElementById("btnDownloadMontage")?.addEventListener("click", () => this.downloadMontageImage());
    document.getElementById("btnExportCsv")?.addEventListener("click", () => this.exportCsvReport());
    document.getElementById("btnPrintReport")?.addEventListener("click", () => window.print());
  }

  getCVParameters() {
    return {
      method: this.methodSelect.value,
      hough_param2: parseInt(this.sliderParam2.value, 10),
      min_radius: parseInt(this.sliderMinR.value, 10),
      max_radius: parseInt(this.sliderMaxR.value, 10),
      canny_high: parseInt(this.sliderCanny.value, 10),
      calibration_mode: this.calibModeSelect.value,
      reference_denomination: parseInt(this.refDenomSelect.value, 10),
      reference_coin_index: this.referenceCoinIndex
    };
  }

  async loadCameraDevices() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        return;
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === "videoinput");

      this.cameraSelect.innerHTML = "";
      if (videoDevices.length === 0) {
        this.cameraSelect.innerHTML = "<option value=''>Default Camera</option>";
        return;
      }

      videoDevices.forEach((device, idx) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `Camera ${idx + 1}`;
        this.cameraSelect.appendChild(option);
      });
    } catch (e) {
      console.warn("Could not enumerate camera devices:", e);
    }
  }

  switchMode(mode) {
    this.activeMode = mode;
    if (mode === "webcam") {
      this.uploadContainer.style.display = "none";
      this.feedContainer.style.display = "flex";
      this.startWebcam();
    } else if (mode === "upload") {
      this.stopWebcam();
      this.uploadContainer.style.display = "block";
      this.feedContainer.style.display = "flex";
    } else if (mode === "backend") {
      this.stopWebcam();
      this.uploadContainer.style.display = "none";
      this.feedContainer.style.display = "flex";
      this.feedPlaceholder.style.display = "none";
      // Point image directly to backend stream
      const img = new Image();
      img.src = "/video_feed";
      img.onload = () => {
        this.canvasFeed.width = img.width;
        this.canvasFeed.height = img.height;
        this.ctx.drawImage(img, 0, 0);
      };
    }
  }

  async startWebcam(deviceId = null) {
    this.stopWebcam();
    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          deviceId: deviceId ? { exact: deviceId } : undefined
        }
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.webcamVideo.srcObject = this.stream;
      await this.webcamVideo.play();

      this.isStreaming = true;
      this.feedPlaceholder.style.display = "none";
      this.btnStartCam.style.display = "none";
      this.btnStopCam.style.display = "inline-flex";

      // Re-enumerate to get labeled device names
      this.loadCameraDevices();

      // Start processing loop
      this.requestFrameLoop();
    } catch (err) {
      console.error("Camera access error:", err);
      alert(`Could not access camera: ${err.message}. Please allow camera permissions in your browser.`);
      this.feedPlaceholder.style.display = "flex";
    }
  }

  stopWebcam() {
    this.isStreaming = false;
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.btnStartCam.style.display = "inline-flex";
    this.btnStopCam.style.display = "none";
  }

  requestFrameLoop() {
    if (!this.isStreaming) return;

    const now = performance.now();
    const interval = 1000 / this.targetFps;

    if (now - this.lastFrameTime >= interval && !this.processingFrame) {
      this.lastFrameTime = now;
      this.processWebcamFrame();
    }

    requestAnimationFrame(() => this.requestFrameLoop());
  }

  async processWebcamFrame() {
    if (!this.isStreaming || this.processingFrame) return;
    if (this.webcamVideo.videoWidth === 0 || this.webcamVideo.videoHeight === 0) return;

    this.processingFrame = true;

    // Offscreen canvas for frame capture
    const offscreen = document.createElement("canvas");
    offscreen.width = this.webcamVideo.videoWidth;
    offscreen.height = this.webcamVideo.videoHeight;
    const offCtx = offscreen.getContext("2d");
    offCtx.drawImage(this.webcamVideo, 0, 0);

    const frameBase64 = offscreen.toDataURL("image/jpeg", 0.75);

    try {
      const response = await fetch("/api/process_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frame: frameBase64,
          params: this.getCVParameters(),
          include_stages: false,
          include_montage: false
        })
      });

      const data = await response.json();
      if (data.success) {
        this.renderDetectionResults(data);
      }
    } catch (e) {
      console.error("Frame processing error:", e);
    } finally {
      this.processingFrame = false;
    }
  }

  captureSnapshot() {
    if (!this.isStreaming) return;
    // Pause stream and fetch high-resolution with all pipeline stages
    const offscreen = document.createElement("canvas");
    offscreen.width = this.webcamVideo.videoWidth;
    offscreen.height = this.webcamVideo.videoHeight;
    const offCtx = offscreen.getContext("2d");
    offCtx.drawImage(this.webcamVideo, 0, 0);

    const frameBase64 = offscreen.toDataURL("image/jpeg", 0.95);
    this.stopWebcam();
    this.processSingleImage(frameBase64);
  }

  async handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("params", JSON.stringify(this.getCVParameters()));
    formData.append("include_stages", "true");
    formData.append("include_montage", "true");

    try {
      const response = await fetch("/api/upload_image", {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (data.success) {
        this.renderDetectionResults(data);
      } else {
        alert("Detection error: " + (data.error || "Unknown error"));
      }
    } catch (e) {
      console.error("Upload error:", e);
      alert("Failed to upload image.");
    }
  }

  async loadSample(filename) {
    try {
      const params = this.getCVParameters();
      const response = await fetch(`/api/sample/${filename}?method=${params.method}`);
      const data = await response.json();
      if (data.success) {
        this.renderDetectionResults(data);
      }
    } catch (e) {
      console.error("Error loading sample:", e);
    }
  }

  async reprocessCurrentImage() {
    if (!this.currentImageB64) return;
    try {
      const response = await fetch("/api/process_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frame: this.currentImageB64,
          params: this.getCVParameters(),
          include_stages: true,
          include_montage: true
        })
      });
      const data = await response.json();
      if (data.success) {
        this.renderDetectionResults(data);
      }
    } catch (e) {
      console.error("Reprocess error:", e);
    }
  }

  renderDetectionResults(data) {
    this.feedPlaceholder.style.display = "none";
    this.lastCoins = data.coins || [];
    this.lastSummary = data.summary || {};
    this.lastMontageB64 = data.montage_image || null;

    if (data.annotated_image) {
      this.currentImageB64 = data.annotated_image;
      const img = new Image();
      img.onload = () => {
        this.canvasFeed.width = img.width;
        this.canvasFeed.height = img.height;
        this.ctx.drawImage(img, 0, 0);
      };
      img.src = data.annotated_image;
    }

    // Play chime when new coins appear
    const currentCount = data.summary.total_count || 0;
    if (currentCount > this.lastCoinCount && currentCount > 0) {
      this.playCoinChime();
    }
    this.lastCoinCount = currentCount;

    // Update Stats Card
    this.statTotalAmount.textContent = `₹${data.summary.total_amount || 0}`;
    this.statCoinCount.textContent = `${currentCount}`;
    this.statPpm.textContent = `${data.summary.pixels_per_mm || 0} px/mm`;

    // Update Denomination Counts
    const breakdown = data.summary.breakdown || {};
    for (const [denom, countEl] of Object.entries(this.denomCounts)) {
      if (countEl) {
        countEl.textContent = breakdown[denom] || 0;
      }
    }

    // Update Table
    this.renderCoinsTable(this.lastCoins);

    // Update Pipeline Stages
    if (data.stages) {
      this.renderPipelineStages(data.stages, data.annotated_image);
    }
  }

  renderCoinsTable(coins) {
    this.coinsTableBody.innerHTML = "";
    if (coins.length === 0) {
      this.coinsTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted); padding:1.5rem;">No coins detected in current frame</td></tr>`;
      return;
    }

    const denomOrder = [1, 2, 5, 10, 20];
    const denomColors = {
      1: "#00bfff",
      2: "#7b68ee",
      5: "#ffd700",
      10: "#ff8e2a",
      20: "#e632b4"
    };

    coins.forEach((coin, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight:700;">#${coin.id}</td>
        <td>(${coin.center[0]}, ${coin.center[1]})</td>
        <td>${coin.radius.toFixed(1)} px</td>
        <td style="font-weight:600; color:var(--accent-cyan);">${coin.diameter_mm} mm</td>
        <td>${coin.color_tone}</td>
        <td>
          <button class="coin-badge-btn" data-coin-idx="${idx}" style="background-color:${coin.display_color_hex || denomColors[coin.denomination] || '#00e5ff'}; border:none; border-radius:4px; padding:0.2rem 0.55rem; color:#fff; font-weight:700; cursor:pointer;" title="Click to change denomination">
            ${coin.denomination_str || ('₹' + coin.denomination)}
          </button>
        </td>
        <td>
          <div style="background:rgba(255,255,255,0.1); border-radius:4px; height:8px; width:70px; overflow:hidden;">
            <div style="background:var(--accent-green); height:100%; width:${coin.confidence * 100}%;"></div>
          </div>
        </td>
      `;
      this.coinsTableBody.appendChild(tr);
    });

    // Event listener for manual denomination toggle
    this.coinsTableBody.querySelectorAll(".coin-badge-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const coinIdx = parseInt(e.target.dataset.coinIdx, 10);
        if (this.lastCoins && this.lastCoins[coinIdx]) {
          const currentDenom = this.lastCoins[coinIdx].denomination;
          const nextIdx = (denomOrder.indexOf(currentDenom) + 1) % denomOrder.length;
          const nextDenom = denomOrder[nextIdx];

          this.lastCoins[coinIdx].denomination = nextDenom;
          this.lastCoins[coinIdx].denomination_str = `₹${nextDenom}`;
          this.lastCoins[coinIdx].display_color_hex = denomColors[nextDenom];

          // Recalculate summary totals
          let newTotal = 0;
          const newBreakdown = { 1: 0, 2: 0, 5: 0, 10: 0, 20: 0 };
          this.lastCoins.forEach(c => {
            newTotal += c.denomination;
            newBreakdown[c.denomination] = (newBreakdown[c.denomination] || 0) + 1;
          });

          this.lastSummary.total_amount = newTotal;
          this.lastSummary.breakdown = newBreakdown;

          this.statTotalAmount.textContent = `₹${newTotal}`;
          for (const [denom, countEl] of Object.entries(this.denomCounts)) {
            if (countEl) countEl.textContent = newBreakdown[denom] || 0;
          }

          this.renderCoinsTable(this.lastCoins);
        }
      });
    });
  }

  renderPipelineStages(stages, annotatedB64) {
    if (stages.original && this.stageOriginal) this.stageOriginal.src = stages.original;
    if (stages.enhanced && this.stageGrayscale) this.stageGrayscale.src = stages.enhanced;
    if (stages.blurred && this.stageBlurred) this.stageBlurred.src = stages.blurred;
    if (stages.edges && this.stageEdges) this.stageEdges.src = stages.edges;
    if (stages.distance_transform && this.stageDistance) this.stageDistance.src = stages.distance_transform;
    if (annotatedB64 && this.stageFinal) this.stageFinal.src = annotatedB64;
  }

  handleCanvasClick(event) {
    if (!this.lastCoins || this.lastCoins.length === 0) return;

    const rect = this.canvasFeed.getBoundingClientRect();
    const scaleX = this.canvasFeed.width / rect.width;
    const scaleY = this.canvasFeed.height / rect.height;

    const clickX = (event.clientX - rect.left) * scaleX;
    const clickY = (event.clientY - rect.top) * scaleY;

    // Find clicked coin
    let clickedCoinIndex = -1;
    for (let i = 0; i < this.lastCoins.length; i++) {
      const c = this.lastCoins[i];
      const dist = Math.sqrt((clickX - c.center[0]) ** 2 + (clickY - c.center[1]) ** 2);
      if (dist <= c.radius) {
        clickedCoinIndex = i;
        break;
      }
    }

    if (clickedCoinIndex !== -1) {
      const selectedCoin = this.lastCoins[clickedCoinIndex];
      this.referenceCoinIndex = clickedCoinIndex;
      alert(`Selected Coin #${selectedCoin.id} as calibration scale reference! Scale recalibrating...`);
      this.reprocessCurrentImage();
    }
  }

  downloadAnnotatedImage() {
    if (!this.canvasFeed) return;
    const link = document.createElement("a");
    link.download = `coin_detection_result_${Date.now()}.jpg`;
    link.href = this.canvasFeed.toDataURL("image/jpeg", 0.95);
    link.click();
  }

  downloadMontageImage() {
    if (!this.lastMontageB64) {
      alert("Montage is generated when an image or snapshot is processed. Load a sample or upload a photo to download the 6-stage montage.");
      return;
    }
    const link = document.createElement("a");
    link.download = `cv_pipeline_montage_${Date.now()}.jpg`;
    link.href = this.lastMontageB64;
    link.click();
  }

  exportCsvReport() {
    if (!this.lastCoins || this.lastCoins.length === 0) {
      alert("No coin data available to export.");
      return;
    }

    let csv = "Coin ID,Center X,Center Y,Radius (px),Diameter (mm),Color Tone,Denomination (INR),Confidence\n";
    this.lastCoins.forEach(c => {
      csv += `${c.id},${c.center[0]},${c.center[1]},${c.radius.toFixed(2)},${c.diameter_mm},${c.color_tone},${c.denomination},${c.confidence}\n`;
    });

    csv += `\nTotal Count,${this.lastSummary.total_count}\n`;
    csv += `Total Amount (INR),${this.lastSummary.total_amount}\n`;
    csv += `Scale (px/mm),${this.lastSummary.pixels_per_mm}\n`;

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `coin_detection_report_${Date.now()}.csv`;
    link.click();
  }
}

// Instantiate on DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
  window.coinApp = new CoinDetectorApp();
});
