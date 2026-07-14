"""
🎯 VectorizeIt — Ücretsiz Açık Kaynak Bitmap → Vektör Dönüştürücü
Vectorizer.AI klonu — Tamamen ücretsiz, GPU gerektirmez!

Motor: VTracer (Rust tabanlı, MIT lisans)
Backend: Flask (Python)
Frontend: Tek HTML dosyası (inline her şey)

Çalıştırmak için:
    pip install flask vtracer Pillow cairosvg
    python app.py

Render.com / Railway / Koyeb'de ücretsiz deploy edilebilir!
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import vtracer
import os
import uuid
import time
from PIL import Image
import io
import json

app = Flask(__name__)

# Ayarlar
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_files(folder, max_age_seconds=3600):
    """1 saatten eski dosyaları sil"""
    now = time.time()
    for f in os.listdir(folder):
        fp = os.path.join(folder, f)
        if os.path.isfile(fp) and now - os.path.getmtime(fp) > max_age_seconds:
            os.remove(fp)


# ==================== ANA SAYFA ====================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>VectorizeIt — Ücretsiz Görsel → Vektör Dönüştürücü</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }

:root {
  --primary: #6C5CE7;
  --primary-dark: #5A4BD1;
  --bg: #0F0E17;
  --surface: #1A1929;
  --surface2: #232136;
  --text: #FFFFFE;
  --text2: #A7A9BE;
  --accent: #FF6B6B;
  --success: #00D2D3;
  --radius: 16px;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}

/* HEADER */
.header {
  padding: 20px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.logo {
  font-size: 22px;
  font-weight: 800;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.badge {
  font-size: 11px;
  background: var(--success);
  color: #000;
  padding: 4px 10px;
  border-radius: 20px;
  font-weight: 700;
}

/* CONTAINER */
.container {
  max-width: 700px;
  margin: 0 auto;
  padding: 20px;
}

/* UPLOAD AREA */
.upload-area {
  border: 2px dashed rgba(108,92,231,0.4);
  border-radius: var(--radius);
  padding: 50px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
  background: var(--surface);
  position: relative;
  margin-bottom: 24px;
}
.upload-area:hover, .upload-area.dragover {
  border-color: var(--primary);
  background: rgba(108,92,231,0.08);
  transform: scale(1.01);
}
.upload-area.has-file {
  border-color: var(--success);
  padding: 20px;
}
.upload-icon {
  font-size: 48px;
  margin-bottom: 16px;
}
.upload-text {
  font-size: 16px;
  color: var(--text2);
  margin-bottom: 8px;
}
.upload-text strong {
  color: var(--primary);
}
.upload-hint {
  font-size: 12px;
  color: var(--text2);
  opacity: 0.6;
}
.upload-area input[type="file"] {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

/* PREVIEW */
.preview-container {
  display: none;
  margin-bottom: 24px;
}
.preview-container.show { display: block; }
.preview-img {
  width: 100%;
  max-height: 300px;
  object-fit: contain;
  border-radius: 12px;
  background: #1a1a2e;
}
.file-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  font-size: 13px;
  color: var(--text2);
}
.file-name { font-weight: 600; color: var(--text); }
.remove-btn {
  background: none;
  border: none;
  color: var(--accent);
  cursor: pointer;
  font-size: 14px;
  padding: 4px 8px;
}

/* SETTINGS */
.settings {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 24px;
  display: none;
}
.settings.show { display: block; }
.settings-title {
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 16px;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 1px;
}
.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.setting-row:last-child { border-bottom: none; }
.setting-label {
  font-size: 14px;
  display: flex;
  flex-direction: column;
}
.setting-label small {
  font-size: 11px;
  color: var(--text2);
  margin-top: 2px;
}
.setting-control select, .setting-control input[type=range] {
  background: var(--surface2);
  border: 1px solid rgba(255,255,255,0.1);
  color: var(--text);
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 13px;
  min-width: 120px;
}
.setting-control input[type=range] {
  -webkit-appearance: none;
  width: 120px;
  height: 6px;
  border-radius: 3px;
  background: var(--surface2);
  outline: none;
  padding: 0;
  border: none;
}
.setting-control input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px; height: 18px;
  border-radius: 50%;
  background: var(--primary);
  cursor: pointer;
}
.range-val {
  font-size: 12px;
  color: var(--primary);
  min-width: 30px;
  text-align: right;
}

/* CONVERT BUTTON */
.convert-btn {
  width: 100%;
  padding: 18px;
  border: none;
  border-radius: var(--radius);
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  color: white;
  font-size: 17px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.3s;
  margin-bottom: 24px;
  display: none;
  position: relative;
  overflow: hidden;
}
.convert-btn.show { display: block; }
.convert-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(108,92,231,0.4); }
.convert-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

/* PROGRESS */
.progress-bar {
  display: none;
  margin-bottom: 24px;
}
.progress-bar.show { display: block; }
.progress-track {
  height: 4px;
  background: var(--surface2);
  border-radius: 4px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary), var(--success));
  border-radius: 4px;
  width: 0%;
  transition: width 0.3s;
}
.progress-text {
  font-size: 13px;
  color: var(--text2);
  margin-top: 8px;
  text-align: center;
}

/* RESULT */
.result-area {
  display: none;
  background: var(--surface);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 24px;
}
.result-area.show { display: block; }
.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.result-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--success);
}
.result-stats {
  font-size: 12px;
  color: var(--text2);
}

/* COMPARISON SLIDER */
.comparison {
  position: relative;
  width: 100%;
  aspect-ratio: 16/10;
  max-height: 350px;
  border-radius: 12px;
  overflow: hidden;
  cursor: col-resize;
  margin-bottom: 16px;
  background: #1a1a2e;
}
.comparison img {
  position: absolute;
  top: 0; left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.comparison .after-wrap {
  position: absolute;
  top: 0; left: 0;
  width: 50%;
  height: 100%;
  overflow: hidden;
}
.comparison .after-wrap img {
  width: 200%;
}
.comparison .slider-line {
  position: absolute;
  top: 0;
  left: 50%;
  width: 3px;
  height: 100%;
  background: var(--primary);
  z-index: 10;
  pointer-events: none;
}
.comparison .slider-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 36px; height: 36px;
  background: var(--primary);
  border-radius: 50%;
  z-index: 11;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  pointer-events: none;
  box-shadow: 0 2px 10px rgba(0,0,0,0.4);
}
.label-badge {
  position: absolute;
  top: 10px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  z-index: 5;
}
.label-before { left: 10px; background: rgba(255,107,107,0.8); }
.label-after { right: 10px; background: rgba(0,210,211,0.8); color: #000; }

/* DOWNLOAD BUTTONS */
.download-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.dl-btn {
  padding: 14px;
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  background: var(--surface2);
  color: var(--text);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
  text-decoration: none;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.dl-btn:hover {
  background: var(--primary);
  border-color: var(--primary);
  transform: translateY(-1px);
}
.dl-btn.primary-dl {
  grid-column: 1 / -1;
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  border-color: var(--primary);
  font-size: 16px;
  padding: 16px;
}

/* FORMAT INFO */
.format-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin: 24px 0;
}
.format-card {
  background: var(--surface);
  border-radius: 12px;
  padding: 16px;
  text-align: center;
}
.format-card .icon { font-size: 28px; margin-bottom: 8px; }
.format-card .name { font-weight: 700; font-size: 14px; }
.format-card .desc { font-size: 11px; color: var(--text2); margin-top: 4px; }

/* HOW IT WORKS */
.how-section {
  margin: 32px 0;
}
.how-title {
  font-size: 20px;
  font-weight: 800;
  margin-bottom: 20px;
  text-align: center;
}
.step-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}
.step-card {
  background: var(--surface);
  border-radius: 12px;
  padding: 20px;
  display: flex;
  align-items: flex-start;
  gap: 16px;
}
.step-num {
  width: 36px; height: 36px;
  min-width: 36px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 16px;
}
.step-content h3 { font-size: 15px; margin-bottom: 4px; }
.step-content p { font-size: 13px; color: var(--text2); }

/* FOOTER */
.footer {
  text-align: center;
  padding: 32px 20px;
  color: var(--text2);
  font-size: 12px;
  border-top: 1px solid rgba(255,255,255,0.04);
  margin-top: 40px;
}
.footer a { color: var(--primary); text-decoration: none; }

/* TOAST */
.toast {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%) translateY(100px);
  background: var(--surface2);
  color: var(--text);
  padding: 14px 24px;
  border-radius: 12px;
  font-size: 14px;
  z-index: 1000;
  transition: transform 0.3s;
  border: 1px solid rgba(255,255,255,0.1);
}
.toast.show { transform: translateX(-50%) translateY(0); }

/* MOBILE */
@media (max-width: 480px) {
  .container { padding: 12px; }
  .upload-area { padding: 36px 16px; }
  .upload-icon { font-size: 36px; }
  .download-grid { grid-template-columns: 1fr; }
  .comparison { aspect-ratio: 4/3; }
}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="logo">⚡ VectorizeIt</div>
  <div class="badge">%100 ÜCRETSİZ</div>
</div>

<div class="container">

  <!-- UPLOAD AREA -->
  <div class="upload-area" id="uploadArea">
    <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.gif,.bmp,.webp">
    <div class="upload-icon">📁</div>
    <div class="upload-text"><strong>Görsel Seç</strong> veya sürükle bırak</div>
    <div class="upload-hint">PNG, JPG, GIF, BMP, WebP • Maks 10MB</div>
  </div>

  <!-- PREVIEW -->
  <div class="preview-container" id="previewContainer">
    <img class="preview-img" id="previewImg">
    <div class="file-info">
      <span class="file-name" id="fileName">—</span>
      <button class="remove-btn" onclick="resetAll()">✕ Kaldır</button>
    </div>
  </div>

  <!-- SETTINGS -->
  <div class="settings" id="settingsPanel">
    <div class="settings-title">⚙️ Dönüşüm Ayarları</div>
    
    <div class="setting-row">
      <div class="setting-label">
        Mod
        <small>Renkli veya siyah-beyaz</small>
      </div>
      <div class="setting-control">
        <select id="colorMode">
          <option value="color">🎨 Renkli</option>
          <option value="binary">⬛ Siyah-Beyaz</option>
        </select>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-label">
        Eğri Tipi
        <small>Çıktı path kalitesi</small>
      </div>
      <div class="setting-control">
        <select id="curveMode">
          <option value="spline">Spline (Bézier)</option>
          <option value="polygon">Polygon</option>
          <option value="none">Piksel</option>
        </select>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-label">
        Renk Hassasiyeti
        <small>Daha yüksek = daha fazla renk</small>
      </div>
      <div class="setting-control" style="display:flex;align-items:center;gap:8px;">
        <input type="range" id="colorPrecision" min="1" max="8" value="6">
        <span class="range-val" id="colorPrecisionVal">6</span>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-label">
        Detay Filtresi
        <small>Küçük lekeleri temizle (px)</small>
      </div>
      <div class="setting-control" style="display:flex;align-items:center;gap:8px;">
        <input type="range" id="filterSpeckle" min="0" max="20" value="4">
        <span class="range-val" id="filterSpeckleVal">4</span>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-label">
        Katman Yapısı
        <small>SVG katman organizasyonu</small>
      </div>
      <div class="setting-control">
        <select id="hierarchical">
          <option value="stacked">Yığılmış (Stacked)</option>
          <option value="cutout">Kesme (Cutout)</option>
        </select>
      </div>
    </div>
  </div>

  <!-- CONVERT BUTTON -->
  <button class="convert-btn" id="convertBtn" onclick="doConvert()">
    ⚡ Vektöre Dönüştür
  </button>

  <!-- PROGRESS -->
  <div class="progress-bar" id="progressBar">
    <div class="progress-track">
      <div class="progress-fill" id="progressFill"></div>
    </div>
    <div class="progress-text" id="progressText">Dönüştürülüyor...</div>
  </div>

  <!-- RESULT -->
  <div class="result-area" id="resultArea">
    <div class="result-header">
      <div class="result-title">✅ Dönüşüm Tamamlandı!</div>
      <div class="result-stats" id="resultStats"></div>
    </div>

    <!-- COMPARISON SLIDER -->
    <div class="comparison" id="compSlider">
      <img id="compOriginal" alt="Orijinal">
      <div class="after-wrap" id="afterWrap">
        <img id="compVector" alt="Vektör">
      </div>
      <div class="slider-line" id="sliderLine"></div>
      <div class="slider-handle" id="sliderHandle">◀▶</div>
      <div class="label-badge label-before">ÖNCE</div>
      <div class="label-badge label-after">SONRA</div>
    </div>

    <!-- DOWNLOAD BUTTONS -->
    <div class="download-grid">
      <a class="dl-btn primary-dl" id="dlSvg">📥 SVG İndir</a>
      <a class="dl-btn" id="dlPng">🖼️ PNG (HD)</a>
      <a class="dl-btn" id="dlPdf">📄 PDF</a>
      <a class="dl-btn" onclick="doConvert()" style="cursor:pointer">🔄 Yeniden</a>
    </div>
  </div>

  <!-- FORMATS -->
  <div class="format-grid">
    <div class="format-card">
      <div class="icon">🖼️</div>
      <div class="name">PNG → SVG</div>
      <div class="desc">Şeffaflık korunur</div>
    </div>
    <div class="format-card">
      <div class="icon">📸</div>
      <div class="name">JPG → SVG</div>
      <div class="desc">Fotoğraf & logo</div>
    </div>
    <div class="format-card">
      <div class="icon">🎨</div>
      <div class="name">WebP → SVG</div>
      <div class="desc">Modern format</div>
    </div>
    <div class="format-card">
      <div class="icon">📐</div>
      <div class="name">BMP → SVG</div>
      <div class="desc">Bitmap dönüşüm</div>
    </div>
  </div>

  <!-- HOW IT WORKS -->
  <div class="how-section">
    <div class="how-title">Nasıl Çalışır?</div>
    <div class="step-grid">
      <div class="step-card">
        <div class="step-num">1</div>
        <div class="step-content">
          <h3>Görsel Yükle</h3>
          <p>PNG, JPG, WebP, GIF veya BMP formatında görseli sürükle-bırak veya seç.</p>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <div class="step-content">
          <h3>AI Dönüştürsün</h3>
          <p>Gelişmiş algoritma şekilleri, renkleri ve kenarları analiz ederek vektöre çevirir.</p>
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <div class="step-content">
          <h3>SVG İndir</h3>
          <p>Sonucu SVG, PNG veya PDF olarak indir. Sınırsız boyuta ölçeklenebilir!</p>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- FOOTER -->
<div class="footer">
  <p>VectorizeIt — Açık kaynak, ücretsiz vektörizasyon motoru</p>
  <p style="margin-top:6px;">Motor: <a href="https://github.com/visioncortex/vtracer" target="_blank">VTracer</a> (MIT Lisans) • GPU gerektirmez</p>
</div>

<!-- TOAST -->
<div class="toast" id="toast"></div>

<script>
// ========== STATE ==========
let selectedFile = null;
let currentJobId = null;

// ========== ELEMENTS ==========
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const settingsPanel = document.getElementById('settingsPanel');
const convertBtn = document.getElementById('convertBtn');
const progressBar = document.getElementById('progressBar');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultArea = document.getElementById('resultArea');

// ========== RANGE SLIDERS ==========
document.getElementById('colorPrecision').oninput = function() {
  document.getElementById('colorPrecisionVal').textContent = this.value;
};
document.getElementById('filterSpeckle').oninput = function() {
  document.getElementById('filterSpeckleVal').textContent = this.value;
};

// ========== DRAG & DROP ==========
uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('dragover');
});
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

// ========== FILE HANDLING ==========
function handleFile(file) {
  if (file.size > 10 * 1024 * 1024) {
    showToast('❌ Dosya 10MB\'dan büyük olamaz!');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewContainer.classList.add('show');
    uploadArea.style.display = 'none';
    settingsPanel.classList.add('show');
    convertBtn.classList.add('show');
    resultArea.classList.remove('show');
  };
  reader.readAsDataURL(file);
  fileName.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
}

function resetAll() {
  selectedFile = null;
  currentJobId = null;
  previewContainer.classList.remove('show');
  settingsPanel.classList.remove('show');
  convertBtn.classList.remove('show');
  progressBar.classList.remove('show');
  resultArea.classList.remove('show');
  uploadArea.style.display = '';
  fileInput.value = '';
}

// ========== CONVERT ==========
async function doConvert() {
  if (!selectedFile) return;
  
  convertBtn.disabled = true;
  convertBtn.textContent = '⏳ İşleniyor...';
  progressBar.classList.add('show');
  resultArea.classList.remove('show');
  
  // Progress animation
  let progress = 0;
  const progInterval = setInterval(() => {
    progress = Math.min(progress + Math.random() * 15, 90);
    progressFill.style.width = progress + '%';
    if (progress < 30) progressText.textContent = '🔍 Görsel analiz ediliyor...';
    else if (progress < 60) progressText.textContent = '✨ Şekiller ve renkler tespit ediliyor...';
    else progressText.textContent = '🎯 Vektör path\'ler oluşturuluyor...';
  }, 300);
  
  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('colormode', document.getElementById('colorMode').value);
  formData.append('mode', document.getElementById('curveMode').value);
  formData.append('color_precision', document.getElementById('colorPrecision').value);
  formData.append('filter_speckle', document.getElementById('filterSpeckle').value);
  formData.append('hierarchical', document.getElementById('hierarchical').value);
  
  try {
    const startTime = Date.now();
    const response = await fetch('/api/vectorize', { method: 'POST', body: formData });
    const data = await response.json();
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    
    clearInterval(progInterval);
    
    if (data.success) {
      progressFill.style.width = '100%';
      progressText.textContent = '✅ Tamamlandı!';
      
      setTimeout(() => {
        progressBar.classList.remove('show');
        showResult(data, elapsed);
      }, 500);
    } else {
      throw new Error(data.error || 'Bilinmeyen hata');
    }
  } catch (err) {
    clearInterval(progInterval);
    progressBar.classList.remove('show');
    showToast('❌ Hata: ' + err.message);
  }
  
  convertBtn.disabled = false;
  convertBtn.textContent = '⚡ Vektöre Dönüştür';
}

// ========== SHOW RESULT ==========
function showResult(data, elapsed) {
  resultArea.classList.add('show');
  currentJobId = data.job_id;
  
  // Stats
  document.getElementById('resultStats').textContent = 
    elapsed + 's • ' + data.svg_size + ' • ' + (data.path_count || '—') + ' path';
  
  // Comparison images
  document.getElementById('compOriginal').src = previewImg.src;
  document.getElementById('compVector').src = '/api/preview/' + data.job_id;
  
  // Download links
  document.getElementById('dlSvg').href = '/api/download/' + data.job_id + '?format=svg';
  document.getElementById('dlSvg').download = selectedFile.name.replace(/\.[^.]+$/, '') + '.svg';
  document.getElementById('dlPng').href = '/api/download/' + data.job_id + '?format=png';
  document.getElementById('dlPng').download = selectedFile.name.replace(/\.[^.]+$/, '') + '_vector.png';
  document.getElementById('dlPdf').href = '/api/download/' + data.job_id + '?format=pdf';
  document.getElementById('dlPdf').download = selectedFile.name.replace(/\.[^.]+$/, '') + '.pdf';
  
  // Scroll to result
  resultArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ========== COMPARISON SLIDER ==========
const compSlider = document.getElementById('compSlider');
let isDragging = false;

function updateSlider(x) {
  const rect = compSlider.getBoundingClientRect();
  let pos = Math.max(0, Math.min(1, (x - rect.left) / rect.width));
  document.getElementById('afterWrap').style.width = (pos * 100) + '%';
  document.getElementById('sliderLine').style.left = (pos * 100) + '%';
  document.getElementById('sliderHandle').style.left = (pos * 100) + '%';
}

compSlider.addEventListener('mousedown', (e) => { isDragging = true; updateSlider(e.clientX); });
compSlider.addEventListener('touchstart', (e) => { isDragging = true; updateSlider(e.touches[0].clientX); }, {passive: true});
document.addEventListener('mousemove', (e) => { if (isDragging) updateSlider(e.clientX); });
document.addEventListener('touchmove', (e) => { if (isDragging) updateSlider(e.touches[0].clientX); }, {passive: true});
document.addEventListener('mouseup', () => isDragging = false);
document.addEventListener('touchend', () => isDragging = false);

// ========== TOAST ==========
function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3000);
}

// ========== PASTE ==========
document.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) handleFile(file);
      break;
    }
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    cleanup_old_files(UPLOAD_FOLDER)
    cleanup_old_files(OUTPUT_FOLDER)

    if "image" not in request.files:
        return jsonify({"success": False, "error": "Dosya yüklenmedi"}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Geçersiz dosya formatı"}), 400

    # Parametreleri al
    colormode = request.form.get("colormode", "color")
    mode = request.form.get("mode", "spline")
    color_precision = int(request.form.get("color_precision", 6))
    filter_speckle = int(request.form.get("filter_speckle", 4))
    hierarchical = request.form.get("hierarchical", "stacked")

    # Dosyayı kaydet
    job_id = str(uuid.uuid4())[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.{ext}")
    output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.svg")
    file.save(input_path)

    try:
        # Boyut kontrolü — çok büyük görselleri küçült
        img = Image.open(input_path)
        w, h = img.size
        max_dim = 2048
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            img.save(input_path)
        img.close()

        # VTracer ile dönüştür
        vtracer.convert_image_to_svg_py(
            input_path,
            output_path,
            colormode=colormode,
            hierarchical=hierarchical,
            mode=mode,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )

        # SVG boyutu
        svg_size = os.path.getsize(output_path)
        if svg_size > 1024 * 1024:
            svg_size_str = f"{svg_size / (1024*1024):.1f} MB"
        else:
            svg_size_str = f"{svg_size / 1024:.1f} KB"

        # Path sayısı (basit hesap)
        with open(output_path, "r") as f:
            svg_content = f.read()
        path_count = svg_content.count("<path")

        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "svg_size": svg_size_str,
                "path_count": path_count,
                "original_size": f"{w}x{h}",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/preview/<job_id>")
def api_preview(job_id):
    svg_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.svg")
    if not os.path.exists(svg_path):
        return "Bulunamadı", 404
    return send_file(svg_path, mimetype="image/svg+xml")


@app.route("/api/download/<job_id>")
def api_download(job_id):
    fmt = request.args.get("format", "svg")
    svg_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.svg")

    if not os.path.exists(svg_path):
        return "Bulunamadı", 404

    if fmt == "svg":
        return send_file(svg_path, as_attachment=True, download_name=f"vectorized_{job_id}.svg")

    elif fmt == "png":
        try:
            import cairosvg
            png_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.png")
            cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=2048)
            return send_file(png_path, as_attachment=True, download_name=f"vectorized_{job_id}.png")
        except Exception:
            # cairosvg yoksa SVG gönder
            return send_file(svg_path, as_attachment=True, download_name=f"vectorized_{job_id}.svg")

    elif fmt == "pdf":
        try:
            import cairosvg
            pdf_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.pdf")
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            return send_file(pdf_path, as_attachment=True, download_name=f"vectorized_{job_id}.pdf")
        except Exception:
            return send_file(svg_path, as_attachment=True, download_name=f"vectorized_{job_id}.svg")

    return "Geçersiz format", 400


# ==================== REST API (Vectorizer.AI uyumlu) ====================
@app.route("/api/v1/vectorize", methods=["POST"])
def api_v1_vectorize():
    """Vectorizer.AI uyumlu API endpoint"""
    if "image" not in request.files:
        return jsonify({"error": {"status": 400, "message": "No image provided"}}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": {"status": 400, "message": "Invalid file format"}}), 400

    job_id = str(uuid.uuid4())[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.{ext}")
    output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}.svg")
    file.save(input_path)

    try:
        vtracer.convert_image_to_svg_py(
            input_path, output_path,
            colormode="color", mode="spline",
            filter_speckle=4, color_precision=6,
            corner_threshold=60, path_precision=3,
        )
        return send_file(output_path, mimetype="image/svg+xml")
    except Exception as e:
        return jsonify({"error": {"status": 500, "message": str(e)}}), 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("⚡ VectorizeIt — Ücretsiz Vektörizasyon")
    print("=" * 50)
    print("🌐 http://localhost:5000")
    print("📡 API: http://localhost:5000/api/v1/vectorize")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
