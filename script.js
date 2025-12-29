 lucide.createIcons();

    // --- GLOBALS ---
    const video = document.getElementById("webcam");
    const procCanvas = document.getElementById("proc-canvas");
    const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });

    const plotCanvas = document.getElementById("ab-plot");
    const plotCtx = plotCanvas.getContext("2d", { willReadFrequently: true });

    const distCanvas = document.getElementById("dist-canvas");
    const distCtx = distCanvas.getContext("2d");

    const sheetCanvas = document.getElementById("sheet-canvas");
    const sheetCtx = sheetCanvas.getContext("2d");
    const ghostOverlay = document.getElementById("ghost-overlay");

    let activeTab = 'colex';
    let isProcessing = false;

    let savedRefLab = null, savedSamLab = null;
    let bgImageData = null;
    let deltaMode = "2000";

    let videoTrack = null;
    let torchSupported = false;
    let isTorchOn = false;
    let torchOpBusy = false;
    let captureBusy = false;

    let sheetFrozen = false;
    let sheetResults = [];
    let lastOri = { beta: 0, gamma: 0, alpha: 0 };

    let layerStabilizer = {}; // (kept, but tracking below is the main stabilizer)

    // ==============================
    // ✅ GLUE GAP TRACKING (stabilize boxes)
    // ==============================
    let layerTracks = [];
    let nextTrackId = 1;

    const TRACK_MATCH_MAX_DY = 28;     // px in roiH_v
    const TRACK_MAX_MISS = 8;
    const TRACK_MIN_AGE_DRAW = 3;
    const TRACK_MIN_CONF_DRAW = 0.35;

    const EMA_Y = 0.25;
    const EMA_X = 0.20;
    const EMA_W = 0.20;
    const EMA_AREA = 0.25;
    const EMA_CONF = 0.20;

    function clampStep(prev, next, ratioStep = 0.25) {
      if (!prev || prev <= 0) return next;
      const lo = prev * (1 - ratioStep);
      const hi = prev * (1 + ratioStep);
      return clamp(next, lo, hi);
    }

    function deadband(prev, next, eps = 1.0) {
      return (Math.abs(next - prev) < eps) ? prev : next;
    }

    function computeLayerConfidence(peakValue, layerThreshold, gapWidthPx, roiW_v) {
      const pv = Math.max(0, peakValue || 0);
      const lt = Math.max(1e-6, layerThreshold || 1);

      const c1 = clamp(pv / (lt * 3.0), 0, 1);

      const gw = Math.max(0, gapWidthPx || 0);
      const c2 = clamp(gw / Math.max(1, roiW_v * 0.45), 0, 1);

      return clamp(c1 * 0.65 + c2 * 0.35, 0, 1);
    }

    function updateLayerTracks(measures) {
      layerTracks.forEach(t => { t._matched = false; });

      for (const m of measures) {
        let best = null;
        let bestDy = Infinity;

        for (const t of layerTracks) {
          if (t._matched) continue;
          const dy = Math.abs((t.yEma ?? t.y) - m.y);
          if (dy < bestDy && dy <= TRACK_MATCH_MAX_DY) {
            bestDy = dy;
            best = t;
          }
        }

        if (best) {
          best._matched = true;
          best.miss = 0;
          best.age = (best.age || 0) + 1;

          best.yEma = lerp(best.yEma ?? m.y, m.y, EMA_Y);
          best.drawYEma = lerp(best.drawYEma ?? m.drawY, m.drawY, EMA_Y);

          const nextW = clampStep(best.dispWEma ?? m.dispW, m.dispW, 0.25);
          best.dispWEma = lerp(best.dispWEma ?? nextW, nextW, EMA_W);

          const nextX = clampStep(best.dispXEma ?? m.dispX, m.dispX, 0.35);
          best.dispXEma = lerp(best.dispXEma ?? nextX, nextX, EMA_X);

          best.areaEma = lerp(best.areaEma ?? m.areaPx2, m.areaPx2, EMA_AREA);
          best.confEma = lerp(best.confEma ?? m.conf, m.conf, EMA_CONF);

          best.drawYEma = deadband(best.drawYEma, best.drawYEma, 0.6);
          best.dispXEma = deadband(best.dispXEma, best.dispXEma, 0.6);
          best.dispWEma = deadband(best.dispWEma, best.dispWEma, 0.6);

        } else {
          layerTracks.push({
            id: nextTrackId++,
            age: 1,
            miss: 0,
            yEma: m.y,
            drawYEma: m.drawY,
            dispXEma: m.dispX,
            dispWEma: m.dispW,
            areaEma: m.areaPx2,
            confEma: m.conf,
            _matched: true
          });
        }
      }

      for (const t of layerTracks) {
        if (!t._matched) {
          t.miss = (t.miss || 0) + 1;
        }
      }

      layerTracks = layerTracks.filter(t => (t.miss || 0) <= TRACK_MAX_MISS);
      layerTracks.sort((a,b) => (a.yEma ?? 0) - (b.yEma ?? 0));
    }

    // =========================================================
    // ✅ MM CALIBRATION (px² -> mm) @ distance 15 cm
    // =========================================================
    const CAL_PTS_15CM = [
      { a: 742,  mm: 1.00 },
      { a: 1910, mm: 7.00 },
      { a: 2590, mm: 8.00 },
      { a: 2768, mm: 8.25 },
      { a: 3390, mm: 8.50 },
      { a: 3575, mm: 9.00 },
    ].sort((p,q)=>p.a-q.a);

    function areaPx2ToMm_15cm(areaPx2){
      const x = Math.max(0, areaPx2 || 0);
      if (x <= 0) return 0;

      const pts = CAL_PTS_15CM;
      if (pts.length < 2) return 0;

      if (x <= pts[0].a){
        const p0 = pts[0], p1 = pts[1];
        const t = (x - p0.a) / (p1.a - p0.a);
        return p0.mm + t * (p1.mm - p0.mm);
      }
      for (let i=0;i<pts.length-1;i++){
        const p0 = pts[i], p1 = pts[i+1];
        if (x >= p0.a && x <= p1.a){
          const t = (x - p0.a) / (p1.a - p0.a);
          return p0.mm + t * (p1.mm - p0.mm);
        }
      }
      const p0 = pts[pts.length-2], p1 = pts[pts.length-1];
      const t = (x - p0.a) / (p1.a - p0.a);
      return p0.mm + t * (p1.mm - p0.mm);
    }

    // ✅ mm range rule: out of 2..10 => RED
    const MM_MIN_OK = 2.0;
    const MM_MAX_OK = 10.0;

    // =========================================================
    // LIGHT THRESHOLD CONTROL (AUTO + MANUAL)
    // =========================================================
    let lightAuto = false;
    let lightBiasManual = 0;
    let lightBiasEma = 0;
    const LIGHT_BIAS_MIN = -30;
    const LIGHT_BIAS_MAX = 30;
    const LIGHT_AUTO_EMA = 0.15;

    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
    function lerp(a, b, t) { return (1 - t) * a + t * b; }
    function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }

    function getLightBias() {
      return clamp(lightAuto ? lightBiasEma : lightBiasManual, LIGHT_BIAS_MIN, LIGHT_BIAS_MAX);
    }

    function syncLightUI() {
      const slider = document.getElementById("light-slider");
      const val = document.getElementById("light-val");
      const box = document.getElementById("lightctl");
      if (!slider || !val || !box) return;

      val.textContent = String(slider.value);
      lightBiasManual = parseInt(slider.value || "0", 10);

      box.classList.toggle("auto-on", !!lightAuto);
  // ✅ ensure lucide icon renders (for reset button icon)
  lucide.createIcons();
    }

    function toggleLightAuto() {
      lightAuto = !lightAuto;
      syncLightUI();
    }

function resetLightBias() {
  // ✅ ถ้าอยู่หน้า Glue Gap: เคลียร์ทั้งหน้า แล้วเริ่มคำนวณใหม่ทั้งหมด
  if (activeTab === 'gluegap') {
    resetSheetUI();               // ล้าง counts/กราฟ/stabilizer/light/header/overlay
    sheetFrozen = false;          // กันค้าง Freeze
    syncSheetCanvasToDisplay();   // กัน overlay/canvas ค้าง
    applyAutoFitLayout();         // รีเฟรช layout
    lucide.createIcons();
    return;
  }

  // ✅ ถ้าไม่ได้อยู่ Glue Gap: รีเซ็ตแค่ Light ตามเดิม
  lightAuto = false;
  lightBiasManual = 0;
  lightBiasEma = 0;

  const slider = document.getElementById("light-slider");
  const val = document.getElementById("light-val");
  if (slider) slider.value = "0";
  if (val) val.textContent = "0";

  syncLightUI();
}


    document.addEventListener("input", (e) => {
      if (e.target && e.target.id === "light-slider") {
        lightAuto = false;
        syncLightUI();
      }
    });

    // =========================================================
    // Adaptive robust threshold (Auto K) + Multi-frame confirm
    // (kept for stability, but RED is now ONLY by mm-range)
    // =========================================================
    let adaptiveAreaK = 2.2;
    const AREA_K_MIN = 1.2;
    const AREA_K_MAX = 3.2;
    const AREA_K_PERCENTILE = 0.08;
    const AREA_K_EMA = 0.12;
    const AREA_MIN_SAMPLES = 4;

    const AREA_ENTER_FRAMES = 6;
    const AREA_EXIT_FRAMES  = 3;

    const LOW_AREA_RATIO = 0.55;

    // =========================================================
    // COVER-MAPPING
    // =========================================================
    function getCoverMetrics() {
      const vw = video.videoWidth || 0;
      const vh = video.videoHeight || 0;
      if (!vw || !vh) return null;

      const r = video.getBoundingClientRect();
      const dw = r.width || 0;
      const dh = r.height || 0;
      if (!dw || !dh) return null;

      const scale = Math.max(dw / vw, dh / vh);
      const scaledW = vw * scale;
      const scaledH = vh * scale;
      const offX = (dw - scaledW) / 2;
      const offY = (dh - scaledH) / 2;

      return { vw, vh, dw, dh, scale, offX, offY, rect: r };
    }

    function dispToVideoXY(mx, my, m) {
      const x = (mx - m.offX) / m.scale;
      const y = (my - m.offY) / m.scale;
      return { x, y };
    }

    function dispRectToVideoRect(rx, ry, rw, rh, m) {
      const p0 = dispToVideoXY(rx, ry, m);
      const p1 = dispToVideoXY(rx + rw, ry + rh, m);

      let x = Math.min(p0.x, p1.x);
      let y = Math.min(p0.y, p1.y);
      let w = Math.abs(p1.x - p0.x);
      let h = Math.abs(p1.y - p0.y);

      x = clamp(x, 0, m.vw - 1);
      y = clamp(y, 0, m.vh - 1);
      w = clamp(w, 1, m.vw - x);
      h = clamp(h, 1, m.vh - y);

      return { x, y, w, h };
    }

    function syncSheetCanvasToDisplay() {
      const m = getCoverMetrics();
      if (!m) return;
      const nw = Math.max(1, Math.round(m.dw));
      const nh = Math.max(1, Math.round(m.dh));
      if (sheetCanvas.width !== nw || sheetCanvas.height !== nh) {
        sheetCanvas.width = nw;
        sheetCanvas.height = nh;
        sheetCtx.clearRect(0, 0, nw, nh);
      }
    }

    // --- LAYOUT ---
    function applyAutoFitLayout() {
      const frame = document.querySelector(".app-frame");
      if (!frame) return;

      const vvH = window.visualViewport ? window.visualViewport.height : window.innerHeight;
      document.documentElement.style.setProperty("--app-h", `${Math.round(vvH)}px`);

      const frameH = frame.offsetHeight;

      if (activeTab === 'gluegap') {
        const camH = Math.round(frameH * 0.50);
        document.documentElement.style.setProperty("--viewport-h", `${camH}px`);

        const wrap = document.getElementById("dist-wrapper");
        if (wrap && wrap.offsetWidth) {
          const tabBar = document.querySelector(".tab-bar");
          const tabHpx = tabBar ? tabBar.getBoundingClientRect().height : 68;
          const extraPad = 12;

          const w = Math.max(1, Math.floor(wrap.clientWidth));
          const h = Math.max(1, Math.floor(wrap.clientHeight - tabHpx - extraPad));

          distCanvas.width = w;
          distCanvas.height = h;
        }
      } else {
        const header = document.querySelector("header");
        const headerH = header ? header.offsetHeight : 64;
        const tabH = 60;
        const available = frameH - headerH - tabH;

        const standardH = Math.max(200, Math.round(available * 0.38));
        document.documentElement.style.setProperty("--viewport-h", `${standardH}px`);

        const viz = Math.max(110, Math.round(available * 0.24));
        document.documentElement.style.setProperty("--viz-h", `${viz}px`);
      }

      const container = document.querySelector(".graph-box");
      if (container && container.offsetWidth > 0 && container.offsetHeight > 0) {
        plotCanvas.width = container.offsetWidth;
        plotCanvas.height = container.offsetHeight;
        renderColorMap();
        drawABGraph();
      }

      syncSheetCanvasToDisplay();
    }
    window.addEventListener("resize", applyAutoFitLayout);
    if (window.visualViewport) window.visualViewport.addEventListener("resize", applyAutoFitLayout);
    applyAutoFitLayout();

    // --- TABS ---
    function switchTab(mode) {
      activeTab = mode;
      document.querySelectorAll('.tab-item').forEach(el => el.classList.remove('active'));
      document.getElementById(`tab-${mode}`).classList.add('active');

      document.getElementById('panel-colex').style.display = 'none';
      document.getElementById('panel-gluegap').classList.add('display-none');

      document.getElementById('vp-side-panel').classList.add('hidden');
      document.getElementById('vp-camera').classList.add('full-width');
      document.getElementById('ctl-color-algo').classList.add('display-none');

      sheetCtx.clearRect(0, 0, sheetCanvas.width, sheetCanvas.height);
      document.getElementById('target-box').classList.add('display-none');

      const lightCtl = document.getElementById("lightctl");
      if (lightCtl) lightCtl.classList.remove("show");

      if (mode === 'colex') {
        document.getElementById('ctl-color-algo').classList.remove('display-none');
        document.getElementById('vp-side-panel').classList.remove('hidden');
        document.getElementById('vp-camera').classList.remove('full-width');
        document.getElementById('panel-colex').style.display = 'flex';
        document.getElementById('target-box').classList.remove('display-none');
      } else if (mode === 'gluegap') {
        ghostOverlay.classList.remove("active");
        setTimeout(() => { ghostOverlay.src = ""; }, 0);
        document.getElementById('panel-gluegap').classList.remove('display-none');
        document.getElementById('panel-gluegap').style.display = 'flex';
        resetSheetUI();

        if (lightCtl) lightCtl.classList.add("show");
      }

      const pc = document.querySelector('.panel-content');
      if (pc) pc.scrollTop = 0;

      applyAutoFitLayout();
      lucide.createIcons();
      syncLightUI();
    }

    // --- CAMERA ---
    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        video.srcObject = stream;
        videoTrack = stream.getVideoTracks?.()[0] || null;

        try {
          const caps = videoTrack?.getCapabilities?.();
          torchSupported = !!caps?.torch;
        } catch (e) { torchSupported = false; }

        updateTorchUI();

        video.onloadedmetadata = () => {
          syncSheetCanvasToDisplay();
          applyAutoFitLayout();
          isProcessing = true;
          requestAnimationFrame(loop);
          syncLightUI();
        };
      } catch (err) {
        alert("Camera Error: " + err);
      }
    }

    function loop() {
      if (!isProcessing) return;
      if (activeTab === 'gluegap') syncSheetCanvasToDisplay();

      if (activeTab === 'colex') {
        drawABGraph();
      } else if (activeTab === 'gluegap') {
        processSheetFrame();
      }
      requestAnimationFrame(loop);
    }

    // =========================================================
    // CAPTURE LIGHTING
    // =========================================================
    async function setTorch(on) {
      if (!videoTrack || !torchSupported || !videoTrack.applyConstraints) return false;
      try {
        await videoTrack.applyConstraints({ advanced: [{ torch: !!on }] });
        return true;
      } catch (e) {
        return false;
      }
    }

    function pulseScreenFlash() {
      document.body.classList.add("screen-flash");
      setTimeout(() => document.body.classList.remove("screen-flash"), 120);
    }

    async function withCaptureLighting(fn) {
      const wasHold = isTorchOn;
      if (wasHold) {
        pulseScreenFlash();
        return await fn();
      }
      if (torchSupported && videoTrack?.applyConstraints) {
        const okOn = await setTorch(true);
        if (okOn) {
          await wait(60);
          try {
            pulseScreenFlash();
            return await fn();
          } finally {
            await setTorch(false);
          }
        }
      }
      pulseScreenFlash();
      return await fn();
    }

    // --- helper (signal processing) ---
    function movingAverage(arr, radius) {
      const out = new Float32Array(arr.length);
      for (let i = 0; i < arr.length; i++) {
        let sum = 0, cnt = 0;
        for (let r = -radius; r <= radius; r++) {
          const j = i + r;
          if (j >= 0 && j < arr.length) { sum += arr[j]; cnt++; }
        }
        out[i] = cnt ? (sum / cnt) : arr[i];
      }
      return out;
    }

    function medianOfSorted(sorted) {
      const n = sorted.length;
      if (!n) return 0;
      const mid = (n / 2) | 0;
      return (n % 2) ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    function robustStats(arr) {
      const a = Array.from(arr);
      a.sort((x, y) => x - y);
      const med = medianOfSorted(a);
      const dev = a.map(v => Math.abs(v - med));
      dev.sort((x, y) => x - y);
      const mad = medianOfSorted(dev) || 0;
      return { med, mad };
    }

    function quantileSorted(sortedArr, q) {
      const n = sortedArr.length;
      if (n === 0) return 0;
      const pos = (n - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      if (sortedArr[base + 1] === undefined) return sortedArr[base];
      return sortedArr[base] + rest * (sortedArr[base + 1] - sortedArr[base]);
    }

    function updateAdaptiveKFromAreas(areas) {
      if (!areas || areas.length < AREA_MIN_SAMPLES) return adaptiveAreaK;

      const { med, mad } = robustStats(areas);
      if (mad <= 1e-6) return adaptiveAreaK;

      const zs = areas.map(a => 0.6745 * (a - med) / mad).sort((x,y)=>x-y);
      const zLow = quantileSorted(zs, AREA_K_PERCENTILE);
      let kNew = Math.abs(zLow);

      kNew = clamp(kNew, AREA_K_MIN, AREA_K_MAX);
      adaptiveAreaK = lerp(adaptiveAreaK, kNew, AREA_K_EMA);

      return adaptiveAreaK;
    }

    function findPeaksDynamic(arr, prominence, minDist) {
      const candidates = [];
      for (let i = 1; i < arr.length - 1; i++) {
        if (arr[i] > arr[i - 1] && arr[i] > arr[i + 1]) candidates.push({ index: i, value: arr[i] });
      }

      const finalPeaks = [];
      const range = 18;
      for (const cand of candidates) {
        let localMin = Infinity;
        for (let k = -range; k <= range; k++) {
          const idx = cand.index + k;
          if (idx >= 0 && idx < arr.length) localMin = Math.min(localMin, arr[idx]);
        }
        if ((cand.value - localMin) >= prominence) finalPeaks.push(cand);
      }

      finalPeaks.sort((a, b) => b.value - a.value);
      const picked = [];
      for (const p of finalPeaks) {
        const tooClose = picked.some(q => Math.abs(q.index - p.index) < minDist);
        if (!tooClose) picked.push(p);
      }
      picked.sort((a, b) => a.index - b.index);
      return picked;
    }

    // =========================================================
    // Largest contour AREA (px^2) inside each detected "box"
    // =========================================================
    function computeLargestBlobAreaPx2(data, w, h, x0, y0, x1, y1, lightBias) {
      x0 = Math.max(0, Math.min(w - 1, x0));
      x1 = Math.max(0, Math.min(w, x1));
      y0 = Math.max(0, Math.min(h - 1, y0));
      y1 = Math.max(0, Math.min(h, y1));
      if (x1 <= x0 + 2 || y1 <= y0 + 2) return 0;

      const ds = 3;
      const gw = Math.floor((x1 - x0) / ds);
      const gh = Math.floor((y1 - y0) / ds);
      if (gw < 6 || gh < 6) return 0;

      const mask = new Uint8Array(gw * gh);

      let sum = 0, cnt = 0, minL = 255;
      for (let gy = 0; gy < gh; gy++) {
        const py = y0 + gy * ds;
        const row = py * w * 4;
        for (let gx = 0; gx < gw; gx++) {
          const px = x0 + gx * ds;
          const i = row + px * 4;
          const luma = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          sum += luma; cnt++;
          if (luma < minL) minL = luma;
        }
      }
      const mean = cnt ? sum / cnt : 180;

      let thr = Math.min(245, Math.max(10, minL + Math.max(14, (mean - minL) * 0.55)));
      thr = clamp(thr + (lightBias || 0), 5, 250);

      for (let gy = 0; gy < gh; gy++) {
        const py = y0 + gy * ds;
        const row = py * w * 4;
        for (let gx = 0; gx < gw; gx++) {
          const px = x0 + gx * ds;
          const i = row + px * 4;
          const luma = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          if (luma <= thr) mask[gy * gw + gx] = 1;
        }
      }

      const visited = new Uint8Array(gw * gh);
      const qx = new Int16Array(gw * gh);
      const qy = new Int16Array(gw * gh);

      let bestArea = 0;

      for (let gy = 0; gy < gh; gy++) {
        for (let gx = 0; gx < gw; gx++) {
          const idx = gy * gw + gx;
          if (!mask[idx] || visited[idx]) continue;

          let head = 0, tail = 0;
          visited[idx] = 1;
          qx[tail] = gx; qy[tail] = gy; tail++;

          let area = 0;

          while (head < tail) {
            const x = qx[head], y = qy[head]; head++;
            area++;

            const n1 = (y - 1) * gw + x;
            const n2 = (y + 1) * gw + x;
            const n3 = y * gw + (x - 1);
            const n4 = y * gw + (x + 1);

            if (y > 0 && mask[n1] && !visited[n1]) { visited[n1] = 1; qx[tail] = x; qy[tail] = y - 1; tail++; }
            if (y < gh - 1 && mask[n2] && !visited[n2]) { visited[n2] = 1; qx[tail] = x; qy[tail] = y + 1; tail++; }
            if (x > 0 && mask[n3] && !visited[n3]) { visited[n3] = 1; qx[tail] = x - 1; qy[tail] = y; tail++; }
            if (x < gw - 1 && mask[n4] && !visited[n4]) { visited[n4] = 1; qx[tail] = x + 1; qy[tail] = y; tail++; }
          }

          if (area > bestArea) bestArea = area;
        }
      }

      return bestArea * (ds * ds);
    }

    // =========================================================
    // GLUE GAP DETECTION (stabilized)
    // ✅ RED only when gapMM is out of [2..10]
    // =========================================================
    function processSheetFrame() {
      if (sheetFrozen) return;
      if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

      const m = getCoverMetrics();
      if (!m) return;

      const dw = sheetCanvas.width;
      const dh = sheetCanvas.height;
      if (!dw || !dh) return;

      // ROI setup
      const roiW_disp = Math.round(dw * 0.50);
      const roiH_disp = Math.round(dh * 0.70);
      const roiX_disp = Math.round((dw - roiW_disp) / 2);

      const bottomSafe_disp = Math.round(dh * 0.15);
      let roiY_disp = Math.round((dh - roiH_disp) / 2 - bottomSafe_disp * 0.2);
      roiY_disp = clamp(roiY_disp, 0, dh - roiH_disp);

      const roiV = dispRectToVideoRect(roiX_disp, roiY_disp, roiW_disp, roiH_disp, m);
      const roiW_v = Math.max(1, Math.round(roiV.w));
      const roiH_v = Math.max(1, Math.round(roiV.h));
      const roiX_v = Math.round(roiV.x);
      const roiY_v = Math.round(roiV.y);

      // Draw HUD (Overlay)
      sheetCtx.clearRect(0, 0, dw, dh);
      sheetCtx.save();
      sheetCtx.fillStyle = "rgba(2, 6, 23, 0.60)";
      sheetCtx.fillRect(0, 0, dw, dh);
      sheetCtx.clearRect(roiX_disp, roiY_disp, roiW_disp, roiH_disp);
      sheetCtx.strokeStyle = "rgba(52, 211, 153, 0.5)";
      sheetCtx.lineWidth = 2;
      sheetCtx.setLineDash([10, 10]);
      sheetCtx.strokeRect(roiX_disp, roiY_disp, roiW_disp, roiH_disp);

sheetCtx.save();

sheetCtx.strokeStyle = "rgba(239, 68, 68, 0.88)"; // red, stronger
sheetCtx.lineWidth = 3;                           // thicker
sheetCtx.setLineDash([10, 8]);                    // dash pattern

// glow for better visibility on dark background
sheetCtx.shadowColor = "rgba(239, 68, 68, 0.75)";
sheetCtx.shadowBlur = 10;

const midX = roiX_disp + roiW_disp / 2;

sheetCtx.beginPath();
sheetCtx.moveTo(midX, roiY_disp);
sheetCtx.lineTo(midX, roiY_disp + roiH_disp);
sheetCtx.stroke();

sheetCtx.restore();
      sheetCtx.restore();




      // Get Image Data
      if (procCanvas.width !== roiW_v || procCanvas.height !== roiH_v) {
        procCanvas.width = roiW_v; procCanvas.height = roiH_v;
      }
      procCtx.drawImage(video, roiX_v, roiY_v, roiW_v, roiH_v, 0, 0, roiW_v, roiH_v);
      const imgData = procCtx.getImageData(0, 0, roiW_v, roiH_v);
      const data = imgData.data;

      // AUTO light bias
      if (lightAuto) {
        const bandLeft = Math.floor(roiW_v * 0.25);
        const bandRight = Math.floor(roiW_v * 0.75);
        const bandTop = Math.floor(roiH_v * 0.15);
        const bandBot = Math.floor(roiH_v * 0.85);

        let minL = 255, sumL = 0, cnt = 0;
        for (let y = bandTop; y < bandBot; y += 6) {
          const row = y * roiW_v * 4;
          for (let x = bandLeft; x < bandRight; x += 6) {
            const i = row + x * 4;
            const luma = data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114;
            sumL += luma; cnt++;
            if (luma < minL) minL = luma;
          }
        }
        const meanL = cnt ? (sumL / cnt) : 160;
        const contrast = Math.max(0, meanL - minL);

        const target = 35;
        const raw = clamp((target - contrast) * 0.7, LIGHT_BIAS_MIN, LIGHT_BIAS_MAX);
        lightBiasEma = lerp(lightBiasEma, raw, LIGHT_AUTO_EMA);

        const valEl = document.getElementById("light-val");
        if (valEl) valEl.textContent = String(Math.round(lightBiasEma));
      }

      const lightBias = getLightBias();

      // Profile Scan
      const profile = new Float32Array(roiH_v);
      const centerStart = Math.floor(roiW_v * 0.35);
      const centerEnd = Math.floor(roiW_v * 0.65);
      const centerW = centerEnd - centerStart;

      for (let y = 0; y < roiH_v; y++) {
        let sumLuma = 0;
        const rowOffset = y * roiW_v * 4;
        for (let x = centerStart; x < centerEnd; x++) {
          const i = rowOffset + x * 4;
          const luma = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          sumLuma += luma;
        }
        const avgLuma = sumLuma / centerW;
        const darkness = Math.max(0, 255 - avgLuma);
        profile[y] = Math.pow(darkness, 1.5);
      }

      const smoothed = movingAverage(profile, 4);
      const { mad } = robustStats(smoothed);
      const layerThreshold = Math.max(30, mad * 1.5);
      const minDist = Math.floor(roiH_v / 12);

      let peaks = findPeaksDynamic(smoothed, layerThreshold, minDist);

      const sx = roiW_disp / roiW_v;
      const sy = roiH_disp / roiH_v;

      const peakYs = peaks.map(p => p.index);
      const bounds = peakYs.map((y, i) => {
        const yPrev = i === 0 ? 0 : peakYs[i - 1];
        const yNext = i === peakYs.length - 1 ? (roiH_v - 1) : peakYs[i + 1];
        const y0 = Math.floor((yPrev + y) / 2);
        const y1 = Math.floor((y + yNext) / 2);
        return { y, y0, y1 };
      });

      // ==============================
      // PASS 1: build measures (no idx binding)
      // ==============================
      const measures = [];

      bounds.forEach((b) => {
        if (b.y > roiH_v * 0.96) return;

        const localY_v = b.y;
        const drawY = roiY_disp + localY_v * sy;

        const x0 = Math.floor(roiW_v * 0.25);
        const x1 = Math.floor(roiW_v * 0.75);
        const y0 = clamp(b.y0, 0, roiH_v - 1);
        const y1 = clamp(b.y1, 0, roiH_v);

        const areaPx2 = computeLargestBlobAreaPx2(
          data, roiW_v, roiH_v, x0, y0, x1, y1, lightBias
        );

        const gapGeo = analyzeGapWidthRow(data, roiW_v, roiH_v, localY_v, lightBias);
        const dispW = Math.max(40, gapGeo.width * sx);
        const dispX = roiX_disp + (roiW_disp - dispW) / 2;

        const peakValue = smoothed[localY_v];
        const conf = computeLayerConfidence(peakValue, layerThreshold, gapGeo.width, roiW_v);

        measures.push({
          y: localY_v,
          drawY,
          dispX,
          dispW,
          areaPx2,
          peakValue,
          conf
        });
      });

      // keep adaptive K update (UI only)
      const allAreasThisFrame = measures.map(x => x.areaPx2).filter(a => a > 0);
      if (allAreasThisFrame.length >= AREA_MIN_SAMPLES) {
        updateAdaptiveKFromAreas(allAreasThisFrame);
      }

      // ==============================
      // TRACKING: stabilize across frames
      // ==============================
      updateLayerTracks(measures);

      // ==============================
      // PASS 2: render using stable tracks
      // ==============================
      const results = [];

      const drawableTracks = layerTracks.filter(t =>
        (t.age || 0) >= TRACK_MIN_AGE_DRAW &&
        (t.confEma || 0) >= TRACK_MIN_CONF_DRAW
      );

      drawableTracks.forEach((t, i) => {
        const layerNo = i + 1;

        const stableArea = Math.max(0, t.areaEma || 0);

        const gapMM_raw = areaPx2ToMm_15cm(stableArea);
        const gapMM = Number.isFinite(gapMM_raw) ? gapMM_raw : 0;

        const outOfRangeMM = (gapMM < MM_MIN_OK || gapMM > MM_MAX_OK);

        const boxColor = outOfRangeMM ? "#ef4444" : "#34d399";
        const bgColor  = outOfRangeMM ? "rgba(239, 68, 68, 0.18)" : "rgba(52, 211, 153, 0.20)";

        const drawY = t.drawYEma ?? 0;
        const dispX = t.dispXEma ?? 0;
        const dispW = t.dispWEma ?? 40;

        const prefix = `#${layerNo}`;
        const xOffset = 58;
        const labelText = `${prefix}  ${gapMM.toFixed(2)} mm`;

        sheetCtx.fillStyle = bgColor;
        sheetCtx.fillRect(dispX, drawY - 8, dispW, 16);

        sheetCtx.strokeStyle = boxColor;
        sheetCtx.lineWidth = 2;
        sheetCtx.setLineDash([]);
        sheetCtx.strokeRect(dispX, drawY - 8, dispW, 16);

        sheetCtx.fillStyle = boxColor;
        sheetCtx.font = "bold 12px 'JetBrains Mono'";
        sheetCtx.fillText(labelText, roiX_disp - xOffset, drawY + 4);

        results.push({ layer: layerNo, gapMM, outOfRangeMM });
      });

      // Summary counts (ONLY mm-range)
      const expected = results.length;
      const redCount = results.filter(r => r.outOfRangeMM).length;
      const okCount = expected - redCount;

      // Header status
      const statusGroup = document.getElementById("header-status-group");
      const header = document.getElementById("glue-header");

      if (redCount > 0) {
        statusGroup.innerHTML =
          `<div class="gc-status-badge" style="color:#ef4444; background:rgba(239,68,68,0.1); border-color:rgba(248,113,113,0.2)">
             <i data-lucide="x-circle"></i> WARNING
           </div>`;
        header.className = "glue-compact-header status-fail";
      } else if (expected >= 7 && expected <= 8) {
        statusGroup.innerHTML =
          `<div class="gc-status-badge"><i data-lucide="check-circle-2" style="color:#34d399"></i> PASS</div>`;
        header.className = "glue-compact-header status-pass";
      } else {
        statusGroup.innerHTML =
          `<div class="gc-status-badge"><i data-lucide="alert-circle"></i> SCANNING</div>`;
        header.className = "glue-compact-header";
      }

      document.getElementById('sum-total').innerText = expected;
      document.getElementById('sum-gap').innerText = okCount;
      document.getElementById('sum-red').innerText = redCount;

      // --- STATS CALCULATION (GREEN only = in-range mm) ---
      const valid = results.filter(r => !r.outOfRangeMM);
      const validMM = valid.map(r => r.gapMM).filter(v => v > 0);

      let avgMM = 0;
      let uniformity = 100;
      let stdDev = 0;

      if (validMM.length > 0) {
        const total = validMM.reduce((acc, v) => acc + v, 0);
        avgMM = total / validMM.length;

        const variance = validMM.reduce((acc, v) => acc + Math.pow(v - avgMM, 2), 0) / validMM.length;
        stdDev = Math.sqrt(variance);

        const totalDev = validMM.reduce((acc, v) => acc + Math.abs(v - avgMM), 0);
        const avgDev = totalDev / validMM.length;
        uniformity = Math.max(0, 100 - ((avgDev / Math.max(1e-6, avgMM)) * 100));
      }

      // ✅ Bell curve shows ALL values; red points only when out-of-range mm
      const pointsAll = results
        .map(r => ({
          v: r.gapMM,
          red: r.outOfRangeMM
        }))
        .filter(p => p.v > 0);

      // If not enough green values, fall back to all for curve center/spread
      if (validMM.length < 2 && pointsAll.length >= 2) {
        const allVals = pointsAll.map(p => p.v);
        const total = allVals.reduce((a,b)=>a+b,0);
        avgMM = total / allVals.length;
        const variance = allVals.reduce((acc, v) => acc + Math.pow(v - avgMM, 2), 0) / allVals.length;
        stdDev = Math.sqrt(variance);
      }

      const statsHeader = document.getElementById("stats-header");
      if (results.length > 0) {
        statsHeader.style.display = "grid";
        statsHeader.innerHTML = `
          <div class="stat-box">
              <div class="stat-title">Avg Gap</div>
              <div class="stat-val">${avgMM.toFixed(2)}<span class="stat-unit">mm</span></div>
          </div>
          <div class="stat-box">
              <div class="stat-title">Consistency <span style="opacity:.7">(K=${adaptiveAreaK.toFixed(2)})</span></div>
              <div class="stat-val" style="color:${uniformity > 85 ? '#34d399' : '#fbbf24'}">
              ${Math.round(uniformity)}<span class="stat-unit">%</span>
              </div>
          </div>
        `;
        document.getElementById("empty-scan-msg").style.display = "none";
      } else {
        statsHeader.style.display = "none";
        document.getElementById("empty-scan-msg").style.display = "flex";
      }

      drawBellCurveMM(pointsAll, avgMM, stdDev);
      lucide.createIcons();
    }

    // =========================================================
    // ✅ Bell Curve (mm) — show ALL points
    // points = [{ v: number(mm), red: boolean }]
    // =========================================================
    function drawBellCurveMM(points, mean, stdDev) {
      distCtx.clearRect(0, 0, distCanvas.width, distCanvas.height);
      if (!points || points.length < 2) return;

      const values = points.map(p => p.v);
      const w = distCanvas.width;
      const h = distCanvas.height;
      const padX = 20;
      const padY = 20;

      const safeSD = Math.max(stdDev || 0, Math.max(0.08, (mean || 1) * 0.08));
      const minX = mean - 3.5 * safeSD;
      const maxX = mean + 3.5 * safeSD;
      const rangeX = Math.max(1e-6, (maxX - minX));

      const mapX = (val) => padX + ((val - minX) / rangeX) * (w - 2 * padX);

      const getGaussian = (x) => (1 / (safeSD * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / safeSD, 2));
      const maxY = getGaussian(mean);

      // histogram (ALL values)
      const binCount = 8;
      const binWidth = rangeX / binCount;
      const bins = new Array(binCount).fill(0);

      values.forEach(v => {
        const binIdx = Math.floor((v - minX) / binWidth);
        if (binIdx >= 0 && binIdx < binCount) bins[binIdx]++;
      });

      const maxBinVal = Math.max(...bins, 1);
      const barScaleY = (h - 2*padY) * 0.7;

      distCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
      bins.forEach((count, i) => {
        const hBar = (count / maxBinVal) * barScaleY;
        const xBar = mapX(minX + i * binWidth);
        const yBar = h - padY - hBar;
        const wBar = ((w - 2*padX) / binCount) - 2;
        if (hBar > 0) distCtx.fillRect(xBar, yBar, wBar, hBar);
      });

      const mapY = (prob) => h - padY - (prob / maxY) * (h - 2 * padY);

      // bell curve
      distCtx.beginPath();
      distCtx.moveTo(mapX(minX), h - padY);
      const steps = 60;
      for (let i = 0; i <= steps; i++) {
        const xVal = minX + (rangeX * (i / steps));
        const yVal = getGaussian(xVal);
        distCtx.lineTo(mapX(xVal), mapY(yVal));
      }
      distCtx.lineTo(mapX(maxX), h - padY);
      distCtx.closePath();

      const grad = distCtx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, "rgba(59, 130, 246, 0.4)");
      grad.addColorStop(1, "rgba(59, 130, 246, 0.05)");
      distCtx.fillStyle = grad;
      distCtx.fill();

      distCtx.strokeStyle = "#3b82f6";
      distCtx.lineWidth = 2;
      distCtx.stroke();

      // mean line
      const cx = mapX(mean);
      distCtx.beginPath();
      distCtx.strokeStyle = "rgba(255,255,255,0.3)";
      distCtx.setLineDash([4, 4]);
      distCtx.moveTo(cx, padY);
      distCtx.lineTo(cx, h - padY);
      distCtx.stroke();
      distCtx.setLineDash([]);

      // points (ALL values)
      points.forEach(p => {
        const val = p.v;
        const vx = mapX(val);
        const vy = mapY(getGaussian(val));
        const dev = Math.abs(val - mean);

        let color = "#34d399";
        if (p.red) color = "#ef4444";
        else if (dev > 1.5 * safeSD) color = "#fbbf24";

        distCtx.beginPath();
        distCtx.fillStyle = color;
        distCtx.arc(vx, vy, 4, 0, Math.PI * 2);
        distCtx.fill();

        distCtx.shadowColor = color;
        distCtx.shadowBlur = 6;
        distCtx.strokeStyle = "rgba(0,0,0,0.45)";
        distCtx.lineWidth = 1;
        distCtx.stroke();
        distCtx.shadowBlur = 0;
      });

      // mean label (mm)
      distCtx.fillStyle = "#64748b";
      distCtx.font = "10px Inter";
      distCtx.textAlign = "center";
      distCtx.fillText(`${mean.toFixed(2)} mm`, cx, h - 5);
    }

    function analyzeGapWidthRow(data, w, h, cy, lightBias) {
      const rowStart = cy * w * 4;

      const bandLeft = Math.floor(w * 0.25);
      const bandRight = Math.floor(w * 0.75);

      let minLuma = 255;
      let minX = ((bandLeft + bandRight) >> 1);
      let sumL = 0;
      let cnt = 0;

      for (let x = bandLeft; x < bandRight; x += 2) {
        const i = rowStart + x * 4;
        const luma = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        sumL += luma; cnt++;
        if (luma < minLuma) { minLuma = luma; minX = x; }
      }

      const meanL = cnt ? (sumL / cnt) : minLuma;
      const contrast = Math.max(0, meanL - minLuma);

      let thr = clamp(minLuma + Math.max(18, contrast * 0.55), 0, 245);
      thr = clamp(thr + (lightBias || 0), 0, 250);

      let left = minX;
      while (left > 0) {
        const i = rowStart + left * 4;
        const luma = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        if (luma > thr) break;
        left--;
      }

      let right = minX;
      while (right < w - 1) {
        const i = rowStart + right * 4;
        const luma = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        if (luma > thr) break;
        right++;
      }

      return { left, right, width: Math.max(0, right - left) };
    }

    function toggleSheetCapture() {
      const btn = document.getElementById("btn-sheet-cap");
      if (sheetFrozen) {
        sheetFrozen = false;
        btn.innerHTML = '<i data-lucide="camera"></i>';
        btn.classList.remove("frozen");
      } else {
        sheetFrozen = true;
        btn.innerHTML = '<i data-lucide="rotate-ccw"></i>';
        btn.classList.add("frozen");
      }
      lucide.createIcons();
    }

    function resetSheetUI() {
      sheetFrozen = false;
      sheetResults = [];
      layerStabilizer = {};
      adaptiveAreaK = 2.2;

      // ✅ reset tracking
      layerTracks = [];
      nextTrackId = 1;

      lightAuto = false;
      lightBiasManual = 0;
      lightBiasEma = 0;
      const slider = document.getElementById("light-slider");
      if (slider) slider.value = "0";
      syncLightUI();

      sheetCtx.clearRect(0, 0, sheetCanvas.width, sheetCanvas.height);
      distCtx.clearRect(0, 0, distCanvas.width, distCanvas.height);

      document.getElementById('sum-total').innerText = "0";
      document.getElementById('sum-gap').innerText = "0";
      document.getElementById('sum-red').innerText = "0";

      const header = document.getElementById("glue-header");
      header.className = "glue-compact-header";
      document.getElementById("header-status-group").innerHTML =
        `<div class="gc-status-badge"><i data-lucide="minus"></i> READY</div>`;

      document.getElementById("stats-header").style.display = "none";
      document.getElementById("empty-scan-msg").style.display = "flex";

      const btn = document.getElementById("btn-sheet-cap");
      btn.innerHTML = '<i data-lucide="camera"></i>';
      btn.classList.remove("frozen");

      lucide.createIcons();
    }

    // --- COLEX LOGIC ---
    window.addEventListener('deviceorientation', (event) => {
      lastOri = { beta: event.beta || 0, gamma: event.gamma || 0, alpha: event.alpha || 0 };
    });

    function getSmartAverage(arr) {
      if (arr.length === 0) return 0;
      arr.sort((a, b) => a - b);
      const trimAmt = Math.floor(arr.length * 0.25);
      const target = arr.slice(trimAmt, arr.length - trimAmt);
      if (target.length === 0) return arr[Math.floor(arr.length / 2)];
      const sum = target.reduce((a, b) => a + b, 0);
      return Math.round(sum / target.length);
    }

    function readOneSample() {
      if (video.readyState !== video.HAVE_ENOUGH_DATA) return null;

      const m = getCoverMetrics();
      if (!m) return null;

      const tb = document.getElementById("target-box");
      const tbR = tb.getBoundingClientRect();
      const vR = m.rect;

      const cx_disp = (tbR.left + tbR.right) / 2 - vR.left;
      const cy_disp = (tbR.top + tbR.bottom) / 2 - vR.top;

      const p = dispToVideoXY(cx_disp, cy_disp, m);

      const W = 100, H = 100;
      if (procCanvas.width !== W) { procCanvas.width = W; procCanvas.height = H; }

      let sx = Math.round(p.x - W / 2);
      let sy = Math.round(p.y - H / 2);

      sx = clamp(sx, 0, m.vw - W);
      sy = clamp(sy, 0, m.vh - H);

      procCtx.drawImage(video, sx, sy, W, H, 0, 0, W, H);
      const frame = procCtx.getImageData(0, 0, W, H);
      const d = frame.data;

      let rList = [], gList = [], bList = [];
      for (let i = 0; i < d.length; i += 20) {
        rList.push(d[i]); gList.push(d[i + 1]); bList.push(d[i + 2]);
      }
      const r = getSmartAverage(rList);
      const g = getSmartAverage(gList);
      const b = getSmartAverage(bList);

      const rgb = `rgb(${r},${g},${b})`;
      const lab = rgbToLab(r, g, b);
      return { rgb, lab };
    }

    async function captureSmartSample() {
      if (captureBusy) return null;
      captureBusy = true;
      try {
        const frames = 10;
        let sumL = 0, sumA = 0, sumB = 0;
        let lastRGB = "";

        for (let i = 0; i < frames; i++) {
          const sample = readOneSample();
          if (sample) {
            sumL += parseFloat(sample.lab.L);
            sumA += parseFloat(sample.lab.a);
            sumB += parseFloat(sample.lab.b);
            lastRGB = sample.rgb;
          }
          await wait(30);
        }

        return {
          rgb: lastRGB,
          lab: {
            L: (sumL / frames).toFixed(1),
            a: (sumA / frames).toFixed(1),
            b: (sumB / frames).toFixed(1)
          }
        };
      } finally {
        captureBusy = false;
      }
    }

    async function lockRef() {
      const btn = document.getElementById("btn-lock-ref");
      if (btn.disabled || captureBusy) return;
      btn.disabled = true;

      const ghostData = captureGhostImage();
      ghostOverlay.src = ghostData;
      ghostOverlay.classList.add("active");

      const sample = await captureSmartSample();

      btn.disabled = false;
      if (!sample) return;

      savedRefLab = { ...sample.lab };
      updateCard("ref", savedRefLab, sample.rgb);
      document.getElementById("card-ref").classList.add("active-ref");
      tryCalculate();
    }

    async function lockSam() {
      const btn = document.getElementById("btn-lock-sam");
      if (btn.disabled || captureBusy) return;
      btn.disabled = true;

      const sample = await captureSmartSample();

      btn.disabled = false;
      if (!sample) return;

      savedSamLab = { ...sample.lab };
      updateCard("sam", savedSamLab, sample.rgb);
      document.getElementById("card-sam").classList.add("active-sam");
      tryCalculate();
    }

    function captureGhostImage() {
      const c = document.createElement("canvas");
      c.width = video.videoWidth; c.height = video.videoHeight;
      c.getContext("2d").drawImage(video, 0, 0);
      return c.toDataURL("image/jpeg", 0.8);
    }

    function updateCard(type, lab, rgb) {
      document.getElementById(`preview-${type}`).style.background = rgb;
      document.getElementById(`${type}-l`).innerText = lab.L;
      document.getElementById(`${type}-a`).innerText = lab.a;
      document.getElementById(`${type}-b`).innerText = lab.b;

      const lVal = Math.max(0, Math.min(100, parseFloat(lab.L)));
      const marker = document.getElementById(`marker-${type}`);
      marker.classList.add("visible");
      marker.style.bottom = lVal + "%";
    }

    function resetAll() {
      savedRefLab = null; savedSamLab = null;

      ["ref", "sam"].forEach((type) => {
        document.getElementById(`card-${type}`).classList.remove("active-ref", "active-sam");
        document.getElementById(`preview-${type}`).style.background = "#334155";
        document.getElementById(`${type}-l`).innerText = "-";
        document.getElementById(`${type}-a`).innerText = "-";
        document.getElementById(`${type}-b`).innerText = "-";
        document.getElementById(`marker-${type}`).classList.remove("visible");
      });

      ghostOverlay.classList.remove("active");
      setTimeout(() => { ghostOverlay.src = ""; }, 300);

      document.getElementById("delta-val").innerText = "--";
      document.getElementById("result-dashboard").className = "result-dashboard";
      document.getElementById("delta-msg").innerText = "READY";
      document.getElementById("advice-content").innerHTML = "";
      document.getElementById("advice-placeholder").style.display = "flex";
    }

    function tryCalculate() {
      if (savedRefLab && savedSamLab) {
        let samForCalc = { ...savedSamLab };
        let dE = 0;
        if (deltaMode === "2000") dE = calculateDeltaE2000(savedRefLab, samForCalc);
        else if (deltaMode === "94") dE = calculateDeltaE94(savedRefLab, samForCalc);
        else dE = calculateDeltaE76(savedRefLab, samForCalc);

        const dash = document.getElementById("result-dashboard");
        document.getElementById("delta-val").innerText = dE.toFixed(2);

        dash.classList.remove("status-match", "status-close", "status-diff");
        if (dE <= 1.0) { dash.classList.add("status-match"); document.getElementById("delta-msg").innerText = "EXACT"; }
        else if (dE <= 2.5) { dash.classList.add("status-close"); document.getElementById("delta-msg").innerText = "CLOSE"; }
        else { dash.classList.add("status-diff"); document.getElementById("delta-msg").innerText = "DIFF"; }

        generateAdvice(savedRefLab, samForCalc);
      }
    }

    function generateAdvice(ref, curr) {
      const dL = parseFloat(ref.L) - parseFloat(curr.L);
      const da = parseFloat(ref.a) - parseFloat(curr.a);
      const db = parseFloat(ref.b) - parseFloat(curr.b);
      const th = 1.0;

      const container = document.getElementById("advice-content");
      const placeholder = document.getElementById("advice-placeholder");
      container.innerHTML = "";

      let actions = [];
      if (dL > th) actions.push({ icon: "sun", text: "Lighten", det: "+White", col: "#fff" });
      else if (dL < -th) actions.push({ icon: "moon", text: "Darken", det: "+Black", col: "#94a3b8" });

      if (da > th) actions.push({ icon: "droplet", text: "Add Red", det: "+Red", col: "#f87171" });
      else if (da < -th) actions.push({ icon: "droplet", text: "Add Green", det: "+Grn", col: "#4ade80" });

      if (db > th) actions.push({ icon: "droplet", text: "Add Yellow", det: "+Yel", col: "#facc15" });
      else if (db < -th) actions.push({ icon: "droplet", text: "Add Blue", det: "+Blu", col: "#60a5fa" });

      if (actions.length === 0) actions.push({ icon: "check-circle", text: "Perfect", det: "Good Match", col: "#10b981" });

      placeholder.style.display = "none";
      actions.forEach((a) => {
        const div = document.createElement("div");
        div.className = "advice-item";
        div.innerHTML = `<div class="adv-icon-row" style="color:${a.col}"><i data-lucide="${a.icon}" width="14"></i> ${a.text}</div><div class="adv-detail">${a.det}</div>`;
        container.appendChild(div);
      });

      lucide.createIcons();
    }

    function drawABGraph() {
      const w = plotCanvas.width, h = plotCanvas.height;
      const cx = w / 2, cy = h / 2;

      if (bgImageData) plotCtx.putImageData(bgImageData, 0, 0);
      else plotCtx.clearRect(0, 0, w, h);

      if (savedRefLab) drawDot(savedRefLab, "#000", "#3b82f6");
      if (savedSamLab) drawDot(savedSamLab, "#000", "#ffffff");

      if (savedRefLab && savedSamLab) {
        const rx = cx + parseFloat(savedRefLab.a), ry = cy - parseFloat(savedRefLab.b);
        const sx = cx + parseFloat(savedSamLab.a), sy = cy - parseFloat(savedSamLab.b);
        plotCtx.beginPath();
        plotCtx.strokeStyle = "rgba(0,0,0,0.5)";
        plotCtx.lineWidth = 1;
        plotCtx.setLineDash([2, 2]);
        plotCtx.moveTo(rx, ry); plotCtx.lineTo(sx, sy);
        plotCtx.stroke();
        plotCtx.setLineDash([]);
      }
    }

    function drawDot(lab, stroke, fill) {
      const cx = plotCanvas.width / 2, cy = plotCanvas.height / 2;
      const x = cx + parseFloat(lab.a), y = cy - parseFloat(lab.b);
      plotCtx.beginPath();
      plotCtx.fillStyle = fill;
      plotCtx.arc(x, y, 6, 0, Math.PI * 2);
      plotCtx.fill();
      plotCtx.lineWidth = 2;
      plotCtx.strokeStyle = stroke;
      plotCtx.stroke();
    }

    function renderColorMap() {
      const w = plotCanvas.width, h = plotCanvas.height;
      const cx = w / 2, cy = h / 2;
      const imgData = plotCtx.createImageData(w, h);
      const d = imgData.data;
      const maxDist = Math.min(w, h) / 2;

      for (let y = 0; y < h; y += 2) {
        for (let x = 0; x < w; x += 2) {
          const dx = x - cx, dy = y - cy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const normDist = Math.min(1, dist / maxDist);
          const angleRad = Math.atan2(-dy, dx);
          let angleDeg = angleRad * (180 / Math.PI);
          if (angleDeg < 0) angleDeg += 360;

          const rgb = hslToRgb(angleDeg, normDist * 100, 100 - normDist * 55);
          const idx = (y * w + x) * 4;

          const setPx = (i) => { d[i] = rgb.r; d[i + 1] = rgb.g; d[i + 2] = rgb.b; d[i + 3] = 255; };
          if (idx < d.length) setPx(idx);
          const idx2 = ((y + 1) * w + (x + 1)) * 4;
          if (idx2 < d.length) setPx(idx2);
        }
      }

      for (let x = 0; x < w; x++) {
        const idx = (Math.floor(cy) * w + x) * 4;
        d[idx] = 255; d[idx + 1] = 255; d[idx + 2] = 255; d[idx + 3] = 40;
      }
      for (let y = 0; y < h; y++) {
        const idx = (y * w + Math.floor(cx)) * 4;
        d[idx] = 255; d[idx + 1] = 255; d[idx + 2] = 255; d[idx + 3] = 40;
      }

      bgImageData = imgData;
    }

    function hslToRgb(h, s, l) {
      s /= 100; l /= 100;
      const k = (n) => (n + h / 30) % 12;
      const a = s * Math.min(l, 1 - l);
      const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
      return { r: Math.round(255 * f(0)), g: Math.round(255 * f(8)), b: Math.round(255 * f(4)) };
    }

    function changeMode(m) { deltaMode = m; document.getElementById("algo-lbl").innerText = m; tryCalculate(); }

    function rgbToLab(r, g, b) {
      let R = r / 255, G = g / 255, B = b / 255;
      R = (R > 0.04045) ? Math.pow((R + 0.055) / 1.055, 2.4) : R / 12.92;
      G = (G > 0.04045) ? Math.pow((G + 0.055) / 1.055, 2.4) : G / 12.92;
      B = (B > 0.04045) ? Math.pow((B + 0.055) / 1.055, 2.4) : B / 12.92;

      let X = (R * 0.4124 + G * 0.3576 + B * 0.1805) * 100;
      let Y = (R * 0.2126 + G * 0.7152 + B * 0.0722) * 100;
      let Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) * 100;

      X /= 95.047; Y /= 100.000; Z /= 108.883;
      X = (X > 0.008856) ? Math.pow(X, 1 / 3) : (7.787 * X) + 16 / 116;
      Y = (Y > 0.008856) ? Math.pow(Y, 1 / 3) : (7.787 * Y) + 16 / 116;
      Z = (Z > 0.008856) ? Math.pow(Z, 1 / 3) : (7.787 * Z) + 16 / 116;

      return { L: ((116 * Y) - 16).toFixed(1), a: (500 * (X - Y)).toFixed(1), b: (200 * (Y - Z)).toFixed(1) };
    }

    function calculateDeltaE76(l1, l2) {
      return Math.sqrt(Math.pow(l2.L - l1.L, 2) + Math.pow(l2.a - l1.a, 2) + Math.pow(l2.b - l1.b, 2));
    }

    function calculateDeltaE94(l1, l2) {
      const L1 = parseFloat(l1.L), a1 = parseFloat(l1.a), b1 = parseFloat(l1.b);
      const L2 = parseFloat(l2.L), a2 = parseFloat(l2.a), b2 = parseFloat(l2.b);
      const C1 = Math.sqrt(a1 ** 2 + b1 ** 2), C2 = Math.sqrt(a2 ** 2 + b2 ** 2);
      const dL = L1 - L2, dC = C1 - C2, da = a1 - a2, db = b1 - b2;
      const dH = Math.sqrt(Math.max(0, da ** 2 + db ** 2 - dC ** 2));
      const SC = 1 + 0.045 * C1, SH = 1 + 0.015 * C1;
      return Math.sqrt(dL ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2);
    }

    function calculateDeltaE2000(l1, l2) {
      const L1 = parseFloat(l1.L), a1 = parseFloat(l1.a), b1 = parseFloat(l1.b);
      const L2 = parseFloat(l2.L), a2 = parseFloat(l2.a), b2 = parseFloat(l2.b);
      const deg2rad = d => d * Math.PI / 180, rad2deg = r => r * 180 / Math.PI;

      const C1 = Math.sqrt(a1 ** 2 + b1 ** 2), C2 = Math.sqrt(a2 ** 2 + b2 ** 2), avgC = (C1 + C2) / 2;
      const G = 0.5 * (1 - Math.sqrt(avgC ** 7 / (avgC ** 7 + 25 ** 7)));
      const a1p = (1 + G) * a1, a2p = (1 + G) * a2;

      const C1p = Math.sqrt(a1p ** 2 + b1 ** 2), C2p = Math.sqrt(a2p ** 2 + b2 ** 2);
      const h1p = (a1p === 0 && b1 === 0) ? 0 : (rad2deg(Math.atan2(b1, a1p)) + (rad2deg(Math.atan2(b1, a1p)) < 0 ? 360 : 0));
      const h2p = (a2p === 0 && b2 === 0) ? 0 : (rad2deg(Math.atan2(b2, a2p)) + (rad2deg(Math.atan2(b2, a2p)) < 0 ? 360 : 0));

      const dLp = L2 - L1, dCp = C2p - C1p;
      let dhp = 0;
      if (C1p * C2p !== 0) dhp = (Math.abs(h2p - h1p) <= 180) ? h2p - h1p : (h2p - h1p > 180 ? h2p - h1p - 360 : h2p - h1p + 360);
      const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(deg2rad(dhp / 2));

      const avgLp = (L1 + L2) / 2, avgCp = (C1p + C2p) / 2;
      let avghp = 0;
      if (C1p * C2p !== 0) avghp = (Math.abs(h1p - h2p) <= 180) ? (h1p + h2p) / 2 : ((h1p + h2p < 360) ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2);

      const T = 1 - 0.17 * Math.cos(deg2rad(avghp - 30)) + 0.24 * Math.cos(deg2rad(2 * avghp)) + 0.32 * Math.cos(deg2rad(3 * avghp + 6)) - 0.20 * Math.cos(deg2rad(4 * avghp - 63));
      const SL = 1 + ((0.015 * (avgLp - 50) ** 2) / Math.sqrt(20 + (avgLp - 50) ** 2));
      const SC = 1 + 0.045 * avgCp;
      const SH = 1 + 0.015 * avgCp * T;

      const dTheta = 30 * Math.exp(-Math.pow((avghp - 275) / 25, 2));
      const RC = 2 * Math.sqrt(avgCp ** 7 / (avgCp ** 7 + 25 ** 7));
      const RT = -Math.sin(deg2rad(2 * dTheta)) * RC;

      return Math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH));
    }

    // --- TORCH (hold button) ---
    function updateTorchUI() {
      const btn = document.getElementById("btn-torch");
      if (!btn) return;

      if (isTorchOn) {
        btn.innerHTML = '<i data-lucide="flashlight-off" style="width:18px;"></i>';
        btn.style.borderColor = "rgba(255,255,255,0.35)";
      } else {
        btn.innerHTML = '<i data-lucide="flashlight" style="width:18px;"></i>';
        btn.style.borderColor = "rgba(255,255,255,0.12)";
      }
      lucide.createIcons();
    }

    async function toggleTorchHold() {
      if (torchOpBusy || captureBusy) return;
      torchOpBusy = true;

      const next = !isTorchOn;
      try {
        if (torchSupported && videoTrack?.applyConstraints) {
          const ok = await setTorch(next);
          if (!ok && next) {
            document.body.classList.add("screen-lamp");
            isTorchOn = true;
            updateTorchUI();
            return;
          }
          isTorchOn = next;
          if (!isTorchOn) document.body.classList.remove("screen-lamp");
          updateTorchUI();
          return;
        }

        isTorchOn = next;
        document.body.classList.toggle("screen-lamp", isTorchOn);
        updateTorchUI();
      } finally {
        torchOpBusy = false;
      }
    }

    initCamera();
