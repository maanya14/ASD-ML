const $ = (id) => document.getElementById(id);

const state = {
  sessionId: null,
  stream: null,
  audioContext: null,
  analyser: null,
  audioData: null,
  timer: null,
  lastVideo: null,
  lastBrightness: null,
  samples: 0,
  running: false,
};

const els = {
  startBtn: $("startBtn"),
  stopBtn: $("stopBtn"),
  subjectSelect: $("subjectSelect"),
  sampleRate: $("sampleRate"),
  video: $("video"),
  canvas: $("sampleCanvas"),
  trend: $("trendCanvas"),
  toast: $("toast"),
};

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => els.toast.classList.remove("show"), 2800);
}

function pct(value) {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function levelColor(prob) {
  if (prob < 0.25) return "#55d187";
  if (prob < 0.5) return "#ffbc58";
  if (prob < 0.75) return "#ff875c";
  return "#ff5f70";
}

function setBar(id, value, color = levelColor(value)) {
  const bar = $(id);
  bar.style.width = pct(value);
  bar.style.background = color;
}

function setText(id, text) {
  $(id).textContent = text;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${body}`);
  }
  return res.json();
}

async function boot() {
  try {
    const health = await api("/api/health");
    setText("apiState", "Ready");

    const subjects = health.subjects || [];
    els.subjectSelect.innerHTML = "";
    for (const subject of subjects) {
      const option = document.createElement("option");
      option.value = subject;
      option.textContent = subject;
      els.subjectSelect.appendChild(option);
    }

    const session = await api("/api/session", {
      method: "POST",
      body: JSON.stringify({ subject_id: subjects[0] || null }),
    });
    state.sessionId = session.session_id;
  } catch (error) {
    setText("apiState", "Offline");
    showToast(`API error: ${error.message}`);
  }
}

async function resetSession() {
  const subjectId = els.subjectSelect.value || null;
  const session = await api("/api/session", {
    method: "POST",
    body: JSON.stringify({ subject_id: subjectId }),
  });
  state.sessionId = session.session_id;
  state.samples = 0;
  state.lastVideo = null;
  state.lastBrightness = null;
  drawTrend([]);
}

async function start() {
  if (state.running) return;
  await resetSession();

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    els.video.srcObject = state.stream;
    await els.video.play();
    setText("cameraState", "Live");
    setText("micState", "Live");
  } catch (error) {
    showToast("Camera or microphone permission is required.");
    throw error;
  }

  state.audioContext = new AudioContext();
  const source = state.audioContext.createMediaStreamSource(state.stream);
  state.analyser = state.audioContext.createAnalyser();
  state.analyser.fftSize = 2048;
  source.connect(state.analyser);
  state.audioData = new Float32Array(state.analyser.fftSize);

  state.running = true;
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  setText("framesSeen", "0 samples");

  const runOnce = async () => {
    if (!state.running) return;
    const started = performance.now();
    try {
      const payload = {
        session_id: state.sessionId,
        audio: collectAudioFeatures(),
        video: collectVideoFeatures(),
        timestamp_ms: Math.round(performance.now()),
      };
      const result = await api("/api/analyze", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderResult(result, performance.now() - started);
    } catch (error) {
      showToast(`Live analysis failed: ${error.message}`);
    }
  };

  await runOnce();
  state.timer = window.setInterval(runOnce, Number(els.sampleRate.value));
}

function stop() {
  state.running = false;
  window.clearInterval(state.timer);
  state.timer = null;

  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
  }
  if (state.audioContext) {
    state.audioContext.close();
  }

  state.stream = null;
  state.audioContext = null;
  state.analyser = null;
  state.audioData = null;
  els.video.srcObject = null;
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  setText("cameraState", "Idle");
  setText("micState", "Idle");
  setText("videoReadout", "Video offline");
  setText("audioReadout", "Audio offline");
}

function collectAudioFeatures() {
  if (!state.analyser || !state.audioData) {
    return { rms: 0, peak: 0, zero_crossing_rate: 0, spectral_centroid: 0 };
  }

  state.analyser.getFloatTimeDomainData(state.audioData);
  let sumSq = 0;
  let peak = 0;
  let crossings = 0;
  let weighted = 0;
  let energy = 0;

  for (let i = 0; i < state.audioData.length; i += 1) {
    const sample = state.audioData[i];
    const abs = Math.abs(sample);
    sumSq += sample * sample;
    peak = Math.max(peak, abs);
    energy += abs;
    weighted += abs * (i / state.audioData.length);
    if (i > 0 && Math.sign(sample) !== Math.sign(state.audioData[i - 1])) crossings += 1;
  }

  const rms = Math.sqrt(sumSq / state.audioData.length);
  const zcr = crossings / state.audioData.length;
  const centroid = energy > 0 ? weighted / energy : 0;
  setText("audioReadout", `rms ${rms.toFixed(3)} peak ${peak.toFixed(2)}`);

  return {
    rms: Math.min(rms, 1),
    peak: Math.min(peak, 1),
    zero_crossing_rate: Math.min(zcr * 18, 1),
    spectral_centroid: Math.min(centroid * 1.8, 1),
  };
}

function collectVideoFeatures() {
  const video = els.video;
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  if (!video.videoWidth || !video.videoHeight) {
    return { brightness: 0, contrast: 0, motion: 0, flicker: 0, face_presence: 0 };
  }

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const gray = new Float32Array(canvas.width * canvas.height);
  let sum = 0;

  for (let i = 0, j = 0; i < image.length; i += 4, j += 1) {
    const value = (image[i] * 0.2126) + (image[i + 1] * 0.7152) + (image[i + 2] * 0.0722);
    gray[j] = value;
    sum += value;
  }

  const brightness = sum / gray.length;
  let variance = 0;
  let motion = 0;

  for (let i = 0; i < gray.length; i += 1) {
    const diff = gray[i] - brightness;
    variance += diff * diff;
    if (state.lastVideo) motion += Math.abs(gray[i] - state.lastVideo[i]);
  }

  const contrast = Math.sqrt(variance / gray.length);
  motion = state.lastVideo ? motion / gray.length : 0;
  const flicker = state.lastBrightness == null ? 0 : Math.abs(brightness - state.lastBrightness);
  const facePresence = estimateFacePresence(gray, canvas.width, canvas.height, brightness);

  state.lastVideo = gray;
  state.lastBrightness = brightness;
  setText("videoReadout", `motion ${motion.toFixed(1)} flicker ${flicker.toFixed(1)}`);

  return {
    brightness: Math.min(brightness, 255),
    contrast: Math.min(contrast, 128),
    motion: Math.min(motion, 255),
    flicker: Math.min(flicker, 255),
    face_presence: facePresence,
  };
}

function estimateFacePresence(gray, width, height, brightness) {
  const x0 = Math.floor(width * 0.28);
  const x1 = Math.floor(width * 0.72);
  const y0 = Math.floor(height * 0.16);
  const y1 = Math.floor(height * 0.78);
  let center = 0;
  let count = 0;

  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      center += gray[y * width + x];
      count += 1;
    }
  }

  const centerAvg = center / Math.max(1, count);
  return Math.max(0, Math.min(1, (centerAvg - brightness + 24) / 48));
}

function renderResult(result, latency) {
  state.samples += 1;
  setText("latency", `${Math.round(latency)} ms`);
  setText("framesSeen", `${state.samples} samples`);
  setText("fusionScore", result.fusion_probability.toFixed(2));
  setText("fusionLevel", result.fusion_level);
  setText("confidence", pct(result.confidence));
  setBar("confidenceBar", result.confidence, "#3dd6c6");

  const degrees = Math.round(result.fusion_probability * 360);
  $("scoreRing").style.background =
    `radial-gradient(circle at center, #10141c 57%, transparent 58%), conic-gradient(${levelColor(result.fusion_probability)} ${degrees}deg, #26303e 0deg)`;

  for (const item of result.modalities) {
    const key = item.name;
    setText(`${key}Score`, item.probability.toFixed(2));
    setText(`${key}Level`, item.level);
    setBar(`${key}Bar`, item.probability);
    if (key === "physio") {
      setText("physioTime", `Subject ${item.signal.subject_id} at ${Math.round(item.signal.center_time_sec)}s`);
    }
  }

  setWeight("audio", result.weights.audio, "#3dd6c6");
  setWeight("video", result.weights.video, "#ffbc58");
  setWeight("physio", result.weights.physio, "#8ea7ff");
  drawTrend(result.timeline);

  const last = result.modalities.reduce((acc, item) => {
    acc[item.name] = item.signal;
    return acc;
  }, {});
  $("featureLog").textContent = JSON.stringify(last, null, 2);
}

function setWeight(name, value, color) {
  setText(`${name}WeightText`, pct(value));
  setBar(`${name}Weight`, value, color);
}

function drawTrend(timeline) {
  const canvas = els.trend;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0d13";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#222b38";
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i += 1) {
    const y = (h / 5) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const series = [
    ["audio", "#3dd6c6"],
    ["video", "#ffbc58"],
    ["physio", "#8ea7ff"],
    ["fusion", "#f7f8fb"],
  ];

  for (const [key, color] of series) {
    ctx.strokeStyle = color;
    ctx.lineWidth = key === "fusion" ? 3 : 2;
    ctx.beginPath();
    timeline.forEach((point, index) => {
      const x = timeline.length <= 1 ? 0 : (index / (timeline.length - 1)) * w;
      const y = h - (point[key] * h);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

els.startBtn.addEventListener("click", () => start().catch((error) => showToast(error.message)));
els.stopBtn.addEventListener("click", stop);
els.subjectSelect.addEventListener("change", () => {
  if (!state.running) resetSession().catch((error) => showToast(error.message));
});
els.sampleRate.addEventListener("change", () => {
  if (state.running) {
    window.clearInterval(state.timer);
    state.timer = window.setInterval(async () => {
      const started = performance.now();
      const payload = {
        session_id: state.sessionId,
        audio: collectAudioFeatures(),
        video: collectVideoFeatures(),
        timestamp_ms: Math.round(performance.now()),
      };
      const result = await api("/api/analyze", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderResult(result, performance.now() - started);
    }, Number(els.sampleRate.value));
  }
});
window.addEventListener("resize", () => drawTrend([]));
window.addEventListener("beforeunload", stop);

boot();
