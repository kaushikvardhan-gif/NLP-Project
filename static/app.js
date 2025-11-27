/* ===========================================
   GLOBALS
=========================================== */
let mediaRecorder = null;
let chunks = [];
let recording = false;
let silenceTimer = null;
let audioContext = null;
let analyser = null;
let microphoneSource = null;
let micStream = null;

const micBtn = document.getElementById("micBtn");
const sentenceEl = document.getElementById("sentence");
const predList = document.getElementById("predList");
const loadingOverlay = document.getElementById("loadingOverlay");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");

// ⭐ Default MIC animation state
micBtn.classList.add("idle");


/* ===========================================
   COLOR MAP
=========================================== */
const COLORS = {
  joy: "#16a34a",
  amusement: "#22c55e",
  love: "#e11d48",
  excitement: "#fb923c",
  surprise: "#f59e0b",
  anger: "#ef4444",
  disgust: "#a16207",
  fear: "#7c3aed",
  sadness: "#2563eb",
  neutral: "#94a3b8",
  default: "#10b981"
};

function showLoading() { loadingOverlay.classList.remove("hidden"); }
function hideLoading() { loadingOverlay.classList.add("hidden"); }


/* ===========================================
   TEXT ANALYSIS
=========================================== */
async function analyzeText() {
  const text = sentenceEl.value.trim();
  if (!text) {
    predList.innerHTML = "<p class='hint'>Type or speak something first.</p>";
    return;
  }

  showLoading();

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence: text })
  });

  const data = await res.json();
  hideLoading();

  renderPredictions(data.predictions);
}

analyzeBtn.onclick = analyzeText;

clearBtn.onclick = () => {
  sentenceEl.value = "";
  predList.innerHTML = "<p class='hint'>Cleared.</p>";
};


/* ===========================================
   RENDER EMOTIONS
=========================================== */
function escapeHTML(s) {
  return s.replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
  }[c]));
}

function renderPredictions(preds) {
  predList.innerHTML = "";

  if (!preds || preds.length === 0) {
    predList.innerHTML = "<p class='hint'>No emotions detected.</p>";
    return;
  }

  preds.sort((a, b) => b[1] - a[1]);

  preds.forEach(([label, prob]) => {
    const pct = (prob * 100).toFixed(2);
    const div = document.createElement("div");
    div.className = "pred-item";
    div.innerHTML = `
      <div class="pred-left">
        <span class="label">${escapeHTML(label)}</span>
        <span class="pct">${pct}%</span>
      </div>
      <div class="bar-wrap"><div class="bar" style="width:0%"></div></div>
    `;

    predList.appendChild(div);

    requestAnimationFrame(() => {
      div.querySelector(".bar").style.width = pct + "%";
    });
  });
}


/* ===========================================
   🎤 START RECORDING
=========================================== */
async function startRec() {
  try {

    // ⭐ MIC animation change
    micBtn.classList.remove("idle");
    micBtn.classList.add("recording");

    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    chunks = [];
    mediaRecorder = new MediaRecorder(micStream, { mimeType: "audio/webm" });
    mediaRecorder.start();

    recording = true;

    // Silence detection setup
    audioContext = new AudioContext();
    microphoneSource = audioContext.createMediaStreamSource(micStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    microphoneSource.connect(analyser);

    detectSilence();
    mediaRecorder.ondataavailable = e => chunks.push(e.data);

  } catch (err) {
    alert("Microphone access blocked or unavailable.");
    console.error(err);
  }
}


/* ===========================================
   SILENCE DETECTION
=========================================== */
function detectSilence() {
  const buffer = new Uint8Array(analyser.frequencyBinCount);

  function analyze() {
    analyser.getByteTimeDomainData(buffer);

    let maxVol = 0;
    for (let i = 0; i < buffer.length; i++) {
      const deviation = Math.abs(buffer[i] - 128);
      if (deviation > maxVol) maxVol = deviation;
    }

    if (maxVol < 6) {
      if (!silenceTimer) silenceTimer = setTimeout(() => stopRec(), 800);
    } else {
      clearTimeout(silenceTimer);
      silenceTimer = null;
    }

    if (recording) requestAnimationFrame(analyze);
  }

  analyze();
}


/* ===========================================
   🎤 STOP RECORDING
=========================================== */
async function stopRec() {
  if (!recording) return;

  recording = false;

  // ⭐ Switch back to idle pulse
  micBtn.classList.remove("recording");
  micBtn.classList.add("idle");

  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();

  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  clearTimeout(silenceTimer);
  silenceTimer = null;

  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: "audio/webm" });

    const fd = new FormData();
    fd.append("audio", blob, "voice.webm");

    showLoading();
    const res = await fetch("/speech", {
      method: "POST",
      body: fd
    });

    const data = await res.json();
    hideLoading();

    if (data.text) {
      sentenceEl.value = data.text;
      analyzeText();
    }
  };
}


/* ===========================================
   MIC BUTTON CLICK
=========================================== */
micBtn.onclick = () => recording ? stopRec() : startRec();
