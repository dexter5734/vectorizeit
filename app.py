"""
VectorizeIt v3 - Professional Bitmap to Vector Converter
Motor: VTracer (Rust/MIT) + OpenCV preprocessing
Framework: Flask
"""

import os
import uuid
import time

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template_string
import vtracer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _ext(filename):
    if "." in filename:
        return filename.rsplit(".", 1)[1].lower()
    return ""


def _allowed(filename):
    return _ext(filename) in ALLOWED_EXT


def _cleanup(folder, max_age=3600):
    """Remove files older than max_age seconds."""
    now = time.time()
    try:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age:
                os.remove(path)
    except OSError:
        pass


def _size_str(num_bytes):
    if num_bytes >= 1_048_576:
        return f"{num_bytes / 1_048_576:.1f} MB"
    return f"{num_bytes / 1024:.0f} KB"


# ---------------------------------------------------------------------------
# IMAGE PREPROCESSING  (the secret sauce for quality)
# ---------------------------------------------------------------------------


def preprocess_image(filepath):
    """
    Prepare a raster image for high-quality vectorisation.

    Pipeline
    --------
    1. Read with alpha support
    2. Up-scale tiny images (< 1000 px) so VTracer gets clean curves
    3. Down-scale huge images (> 2500 px) to keep speed reasonable
    4. Binarise semi-transparent alpha (avoids wispy edges)
    5. Bilateral filter – smooths noise while preserving edges
    6. K-Means colour quantisation – merges similar colours so VTracer
       produces fewer, cleaner paths instead of thousands of noisy ones
    """

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return  # nothing we can do

    h, w = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]

    # ---- 1. Resize --------------------------------------------------------
    longest = max(w, h)
    if longest < 1000:
        scale = 1000.0 / longest
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    elif longest > 2500:
        scale = 2500.0 / longest
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)

    # ---- 2. Alpha clean-up ------------------------------------------------
    has_alpha = (channels == 4)
    alpha_channel = None

    if has_alpha:
        alpha_channel = img[:, :, 3].copy()
        alpha_channel = np.where(alpha_channel > 128, 255, 0).astype(np.uint8)
        img[:, :, 3] = alpha_channel

    # ---- 3. Bilateral filter on colour channels ---------------------------
    if channels >= 3:
        bgr = img[:, :, :3]
        bgr = cv2.bilateralFilter(bgr, d=9, sigmaColor=55, sigmaSpace=55)
        img[:, :, :3] = bgr

    # ---- 4. Colour quantisation (K-Means) ---------------------------------
    if channels >= 3:
        bgr = img[:, :, :3]
        pixels = bgr.reshape(-1, 3).astype(np.float32)

        k = 24
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0,
        )
        _, labels, centres = cv2.kmeans(
            pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        centres = np.uint8(centres)
        quantised = centres[labels.flatten()].reshape(bgr.shape)
        img[:, :, :3] = quantised

    # ---- 5. Write back ----------------------------------------------------
    cv2.imwrite(filepath, img)


# ---------------------------------------------------------------------------
# VECTORISATION PRESETS
# ---------------------------------------------------------------------------

PRESETS = {
    "logo": dict(
        filter_speckle=2,
        color_precision=5,
        corner_threshold=90,
        length_threshold=3.5,
        splice_threshold=45,
        path_precision=5,
        layer_difference=6,
        max_iterations=10,
    ),
    "illustration": dict(
        filter_speckle=3,
        color_precision=6,
        corner_threshold=60,
        length_threshold=4.0,
        splice_threshold=45,
        path_precision=4,
        layer_difference=10,
        max_iterations=10,
    ),
    "photo": dict(
        filter_speckle=6,
        color_precision=7,
        corner_threshold=60,
        length_threshold=4.0,
        splice_threshold=45,
        path_precision=3,
        layer_difference=16,
        max_iterations=10,
    ),
    "sketch": dict(
        filter_speckle=3,
        color_precision=4,
        corner_threshold=80,
        length_threshold=3.5,
        splice_threshold=45,
        path_precision=5,
        layer_difference=8,
        max_iterations=10,
    ),
}

QUALITY_DELTAS = {
    1: (-2, +4, -2),
    2: (-1, +2, -1),
    3: (0, 0, 0),
    4: (+1, -1, +1),
    5: (+2, -2, +2),
}


def _build_params(preset_name, quality_level, colormode):
    base = PRESETS.get(preset_name, PRESETS["logo"]).copy()
    cp_d, fs_d, pp_d = QUALITY_DELTAS.get(quality_level, (0, 0, 0))
    base["color_precision"] = max(1, min(8, base["color_precision"] + cp_d))
    base["filter_speckle"] = max(0, base["filter_speckle"] + fs_d)
    base["path_precision"] = max(1, min(8, base["path_precision"] + pp_d))
    base["colormode"] = colormode
    base["hierarchical"] = "stacked"
    base["mode"] = "spline"
    return base


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    _cleanup(UPLOAD_DIR)
    _cleanup(OUTPUT_DIR)

    if "image" not in request.files:
        return jsonify(success=False, error="Dosya yüklenmedi"), 400

    f = request.files["image"]
    if not f.filename or not _allowed(f.filename):
        return jsonify(success=False, error="Geçersiz dosya formatı"), 400

    raw = f.read()
    if len(raw) > MAX_BYTES:
        return jsonify(success=False, error="Dosya 10 MB'den büyük"), 400

    job_id = uuid.uuid4().hex[:10]
    ext = _ext(f.filename)
    in_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}.svg")

    with open(in_path, "wb") as fp:
        fp.write(raw)

    preset = request.form.get("preset", "logo")
    quality = int(request.form.get("quality", "3"))
    colormode = request.form.get("colormode", "color")

    try:
        preprocess_image(in_path)
        params = _build_params(preset, quality, colormode)

        vtracer.convert_image_to_svg_py(
            in_path,
            out_path,
            colormode=params["colormode"],
            hierarchical=params["hierarchical"],
            mode=params["mode"],
            filter_speckle=params["filter_speckle"],
            color_precision=params["color_precision"],
            layer_difference=params["layer_difference"],
            corner_threshold=params["corner_threshold"],
            length_threshold=params["length_threshold"],
            max_iterations=params["max_iterations"],
            splice_threshold=params["splice_threshold"],
            path_precision=params["path_precision"],
        )

        svg_bytes = os.path.getsize(out_path)
        with open(out_path, "r", encoding="utf-8") as sf:
            path_count = sf.read().count("<path")

        return jsonify(
            success=True,
            job_id=job_id,
            svg_size=_size_str(svg_bytes),
            path_count=path_count,
        )

    except Exception as exc:
        return jsonify(success=False, error=str(exc)), 500


@app.route("/api/preview/<job_id>")
def api_preview(job_id):
    p = os.path.join(OUTPUT_DIR, f"{job_id}.svg")
    if not os.path.exists(p):
        return "Not found", 404
    return send_file(p, mimetype="image/svg+xml")


@app.route("/api/download/<job_id>")
def api_download(job_id):
    fmt = request.args.get("format", "svg")
    svg_path = os.path.join(OUTPUT_DIR, f"{job_id}.svg")
    if not os.path.exists(svg_path):
        return "Not found", 404

    if fmt == "svg":
        return send_file(
            svg_path, as_attachment=True,
            download_name=f"vector_{job_id}.svg",
        )

    if fmt == "png":
        try:
            import cairosvg
            png_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
            cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=2048)
            return send_file(
                png_path, as_attachment=True,
                download_name=f"vector_{job_id}.png",
            )
        except ImportError:
            return send_file(
                svg_path, as_attachment=True,
                download_name=f"vector_{job_id}.svg",
            )

    if fmt == "pdf":
        try:
            import cairosvg
            pdf_path = os.path.join(OUTPUT_DIR, f"{job_id}.pdf")
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            return send_file(
                pdf_path, as_attachment=True,
                download_name=f"vector_{job_id}.pdf",
            )
        except ImportError:
            return send_file(
                svg_path, as_attachment=True,
                download_name=f"vector_{job_id}.svg",
            )

    return "Invalid format", 400


@app.route("/api/v1/vectorize", methods=["POST"])
def api_v1_vectorize():
    """Vectorizer.AI-compatible REST endpoint."""
    if "image" not in request.files:
        return jsonify(error="No image provided"), 400

    f = request.files["image"]
    job_id = uuid.uuid4().hex[:10]
    ext = _ext(f.filename) if f.filename else "png"
    in_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}.svg")
    f.save(in_path)

    try:
        preprocess_image(in_path)
        params = _build_params("logo", 3, "color")
        vtracer.convert_image_to_svg_py(
            in_path,
            out_path,
            colormode=params["colormode"],
            hierarchical=params["hierarchical"],
            mode=params["mode"],
            filter_speckle=params["filter_speckle"],
            color_precision=params["color_precision"],
            layer_difference=params["layer_difference"],
            corner_threshold=params["corner_threshold"],
            length_threshold=params["length_threshold"],
            max_iterations=params["max_iterations"],
            splice_threshold=params["splice_threshold"],
            path_precision=params["path_precision"],
        )
        return send_file(out_path, mimetype="image/svg+xml")
    except Exception as exc:
        return jsonify(error=str(exc)), 500


# ---------------------------------------------------------------------------
# HTML / CSS / JS  — Single-page application
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>VectorizeIt &mdash; Görsel Vektöre Dönüştür</title>
<style>
/* ===== RESET & BASE ===== */
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --c-bg:#050510;
  --c-s1:#0d0d1a;
  --c-s2:#161625;
  --c-s3:#1e1e30;
  --c-border:#ffffff0a;
  --c-text:#eeeef0;
  --c-muted:#7b7b8e;
  --c-primary:#8b5cf6;
  --c-primary-h:#7c3aed;
  --c-glow:rgba(139,92,246,.25);
  --c-accent:#f472b6;
  --c-green:#34d399;
  --c-red:#fb7185;
  --radius:16px;
  --radius-sm:10px;
}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{
  font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:var(--c-bg);color:var(--c-text);
  min-height:100dvh;
  -webkit-tap-highlight-color:transparent;
  overflow-x:hidden;
}
a{color:inherit;text-decoration:none}
img{display:block;max-width:100%}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#ffffff15;border-radius:3px}

/* ===== LAYOUT ===== */
.page{max-width:640px;margin:0 auto;padding:0 16px 40px}

/* ===== HEADER ===== */
.hdr{
  position:sticky;top:0;z-index:50;
  background:var(--c-bg);
  border-bottom:1px solid var(--c-border);
  padding:14px 16px;
  display:flex;align-items:center;justify-content:space-between;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
}
.hdr-logo{
  font-size:19px;font-weight:800;letter-spacing:-.3px;
  background:linear-gradient(135deg,var(--c-primary),var(--c-accent));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.hdr-badge{
  font-size:9px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;
  background:var(--c-green);color:#000;
  padding:4px 10px;border-radius:20px;
}

/* ===== UPLOAD ZONE ===== */
.upload{
  margin-top:20px;
  border:2px dashed #ffffff12;
  border-radius:var(--radius);
  padding:52px 24px;
  text-align:center;
  cursor:pointer;
  background:var(--c-s1);
  position:relative;
  transition:border-color .2s,background .2s,transform .15s;
}
.upload:hover,.upload.over{
  border-color:var(--c-primary);background:rgba(139,92,246,.04);
}
.upload:active{transform:scale(.985)}
.upload input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-icon{
  width:56px;height:56px;margin:0 auto 16px;
  background:var(--c-s3);border-radius:14px;
  display:flex;align-items:center;justify-content:center;font-size:26px;
}
.upload-title{font-size:15px;font-weight:600;margin-bottom:4px}
.upload-title span{color:var(--c-primary)}
.upload-sub{font-size:12px;color:var(--c-muted)}

/* ===== PREVIEW CARD ===== */
.preview{
  display:none;
  background:var(--c-s1);border-radius:var(--radius);
  overflow:hidden;margin-top:16px;
  border:1px solid var(--c-border);
}
.preview.on{display:block}
.preview-img{
  width:100%;max-height:280px;
  object-fit:contain;background:#08080f;
}
.preview-bar{
  display:flex;justify-content:space-between;align-items:center;
  padding:10px 14px;
}
.preview-name{
  font-size:12px;font-weight:600;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  max-width:65%;
}
.preview-size{font-size:11px;color:var(--c-muted)}
.preview-rm{
  background:none;border:none;color:var(--c-red);
  font-size:12px;cursor:pointer;padding:6px 10px;
  border-radius:8px;font-weight:600;
}
.preview-rm:active{background:#ffffff08}

/* ===== SETTINGS PANEL ===== */
.settings{
  display:none;
  background:var(--c-s1);border-radius:var(--radius);
  padding:18px;margin-top:12px;
  border:1px solid var(--c-border);
}
.settings.on{display:block}
.settings-head{
  font-size:11px;font-weight:700;color:var(--c-muted);
  text-transform:uppercase;letter-spacing:1.2px;margin-bottom:14px;
}
.s-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:10px 0;border-bottom:1px solid var(--c-border);
}
.s-row:last-child{border-bottom:none}
.s-label{font-size:13px;line-height:1.3}
.s-label small{display:block;font-size:10px;color:var(--c-muted);margin-top:1px}
.s-ctrl{display:flex;align-items:center;gap:8px}
.s-ctrl select{
  background:var(--c-s3);border:1px solid var(--c-border);
  color:var(--c-text);padding:7px 10px;border-radius:var(--radius-sm);
  font-size:12px;font-family:inherit;
}
.s-ctrl input[type=range]{
  -webkit-appearance:none;appearance:none;
  width:100px;height:4px;border-radius:2px;
  background:var(--c-s3);outline:none;
}
.s-ctrl input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;appearance:none;
  width:16px;height:16px;border-radius:50%;
  background:var(--c-primary);cursor:pointer;
  box-shadow:0 0 8px var(--c-glow);
}
.range-v{
  font-size:11px;font-weight:700;color:var(--c-primary);
  min-width:16px;text-align:right;
}

/* ===== CONVERT BUTTON ===== */
.convert{
  display:none;width:100%;
  margin-top:16px;padding:16px 24px;
  border:none;border-radius:var(--radius);
  background:linear-gradient(135deg,var(--c-primary),var(--c-primary-h));
  color:#fff;font-size:15px;font-weight:700;font-family:inherit;
  cursor:pointer;position:relative;overflow:hidden;
  transition:transform .15s,box-shadow .2s;
}
.convert.on{display:block}
.convert:hover{box-shadow:0 6px 24px var(--c-glow)}
.convert:active{transform:scale(.98)}
.convert:disabled{opacity:.5;cursor:wait;transform:none;box-shadow:none}
.convert .spin{
  display:none;
  width:16px;height:16px;
  border:2px solid #fff4;border-top-color:#fff;
  border-radius:50%;
  animation:spin .5s linear infinite;
  margin-right:8px;vertical-align:middle;
}
.convert.busy .spin{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}

/* ===== PROGRESS ===== */
.progress{display:none;margin-top:16px}
.progress.on{display:block}
.prog-track{height:3px;background:var(--c-s3);border-radius:2px;overflow:hidden}
.prog-bar{
  height:100%;width:0%;
  background:linear-gradient(90deg,var(--c-primary),var(--c-green));
  border-radius:2px;transition:width .35s ease;
}
.prog-msg{font-size:12px;color:var(--c-muted);text-align:center;margin-top:8px}

/* ===== RESULT AREA ===== */
.result{display:none;margin-top:20px}
.result.on{display:block}
.res-header{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:14px;
}
.res-title{font-size:15px;font-weight:700;color:var(--c-green)}
.res-stats{font-size:11px;color:var(--c-muted)}

/* ===== COMPARISON SLIDER ===== */
.comp{
  position:relative;width:100%;
  aspect-ratio:3/2;
  border-radius:14px;overflow:hidden;
  cursor:ew-resize;background:#08080f;
  touch-action:none;
  border:1px solid var(--c-border);
  margin-bottom:14px;
}
.comp img{
  position:absolute;top:0;left:0;
  width:100%;height:100%;
  object-fit:contain;
  pointer-events:none;
  user-select:none;-webkit-user-select:none;
  -webkit-user-drag:none;
}
.comp-clip{
  position:absolute;top:0;left:0;
  width:50%;height:100%;
  overflow:hidden;
}
.comp-clip img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain}
.comp-line{
  position:absolute;top:0;left:50%;
  width:2px;height:100%;
  background:var(--c-primary);
  transform:translateX(-50%);
  z-index:4;
  box-shadow:0 0 10px var(--c-glow);
}
.comp-knob{
  position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);
  width:34px;height:34px;
  background:var(--c-primary);border-radius:50%;
  z-index:5;
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 2px 12px rgba(0,0,0,.5),0 0 16px var(--c-glow);
}
.comp-knob svg{width:16px;height:16px;fill:#fff}
.comp-tag{
  position:absolute;top:10px;
  padding:3px 9px;border-radius:6px;
  font-size:9px;font-weight:800;letter-spacing:.6px;
  text-transform:uppercase;z-index:3;
}
.tag-before{left:10px;background:rgba(244,114,182,.85)}
.tag-after{right:10px;background:rgba(52,211,153,.85);color:#000}

/* ===== DOWNLOAD GRID ===== */
.dl-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px}
.dl-btn{
  display:flex;align-items:center;justify-content:center;gap:6px;
  padding:13px;
  border:1px solid var(--c-border);border-radius:12px;
  background:var(--c-s1);color:var(--c-text);
  font-size:13px;font-weight:600;font-family:inherit;
  cursor:pointer;text-decoration:none;
  transition:background .15s,border-color .15s,transform .1s;
}
.dl-btn:active{transform:scale(.97);background:var(--c-primary);border-color:var(--c-primary)}
.dl-btn.main{
  grid-column:1/-1;
  background:linear-gradient(135deg,var(--c-primary),var(--c-primary-h));
  border-color:transparent;
  font-size:15px;padding:15px;
}
.dl-btn.main:active{opacity:.9}

/* ===== FEATURE CARDS ===== */
.features{
  display:grid;grid-template-columns:1fr 1fr;gap:8px;
  margin-top:32px;
}
.feat{
  background:var(--c-s1);border-radius:14px;
  padding:18px 14px;text-align:center;
  border:1px solid var(--c-border);
  transition:border-color .2s;
}
.feat:active{border-color:var(--c-primary)}
.feat-icon{font-size:26px;margin-bottom:8px}
.feat-name{font-size:13px;font-weight:700}
.feat-desc{font-size:10px;color:var(--c-muted);margin-top:3px;line-height:1.4}

/* ===== HOW IT WORKS ===== */
.how{margin-top:32px}
.how-title{
  font-size:18px;font-weight:800;text-align:center;
  margin-bottom:16px;
}
.step{
  display:flex;gap:14px;
  background:var(--c-s1);border-radius:14px;
  padding:16px;margin-bottom:8px;
  border:1px solid var(--c-border);
}
.step-n{
  width:32px;height:32px;min-width:32px;
  background:linear-gradient(135deg,var(--c-primary),var(--c-accent));
  border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:14px;
}
.step h3{font-size:14px;font-weight:700;margin-bottom:2px}
.step p{font-size:12px;color:var(--c-muted);line-height:1.5}

/* ===== FOOTER ===== */
.foot{
  text-align:center;padding:28px 16px;
  color:var(--c-muted);font-size:11px;
  border-top:1px solid var(--c-border);
  margin-top:40px;line-height:1.6;
}
.foot a{color:var(--c-primary);font-weight:600}

/* ===== TOAST ===== */
.toast{
  position:fixed;bottom:20px;left:50%;
  transform:translateX(-50%) translateY(100px);
  background:var(--c-s2);color:var(--c-text);
  padding:12px 22px;border-radius:12px;
  font-size:13px;font-weight:500;z-index:100;
  transition:transform .3s cubic-bezier(.34,1.56,.64,1);
  border:1px solid var(--c-border);
  box-shadow:0 8px 30px rgba(0,0,0,.4);
  white-space:nowrap;
}
.toast.on{transform:translateX(-50%) translateY(0)}

/* ===== MOBILE ===== */
@media(max-width:420px){
  .page{padding:0 10px 32px}
  .upload{padding:40px 16px}
  .dl-grid{grid-template-columns:1fr}
  .comp{aspect-ratio:1/1}
  .features{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div class="hdr-logo">&#9889; VectorizeIt</div>
  <div class="hdr-badge">&#x2713; Ücretsiz</div>
</div>

<div class="page">

  <!-- UPLOAD -->
  <div class="upload" id="upload">
    <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.gif,.bmp,.webp">
    <div class="upload-icon">&#128193;</div>
    <div class="upload-title"><span>Görsel seç</span> veya sürükle bırak</div>
    <div class="upload-sub">PNG &middot; JPG &middot; WebP &middot; GIF &middot; BMP &mdash; maks 10 MB</div>
  </div>

  <!-- PREVIEW -->
  <div class="preview" id="preview">
    <img class="preview-img" id="prevImg" alt="Preview">
    <div class="preview-bar">
      <div>
        <div class="preview-name" id="fName">-</div>
        <div class="preview-size" id="fSize">-</div>
      </div>
      <button class="preview-rm" id="btnRemove">&#10005; Kaldır</button>
    </div>
  </div>

  <!-- SETTINGS -->
  <div class="settings" id="settings">
    <div class="settings-head">&#9881; Ayarlar</div>

    <div class="s-row">
      <div class="s-label">Görsel Tipi<small>İçeriğe göre optimize eder</small></div>
      <div class="s-ctrl">
        <select id="optPreset">
          <option value="logo">&#127919; Logo / İkon</option>
          <option value="illustration">&#127912; İllüstrasyon</option>
          <option value="photo">&#128248; Fotoğraf</option>
          <option value="sketch">&#9999;&#65039; Çizim</option>
        </select>
      </div>
    </div>

    <div class="s-row">
      <div class="s-label">Kalite<small>Yüksek = daha detaylı</small></div>
      <div class="s-ctrl">
        <input type="range" id="optQuality" min="1" max="5" value="3">
        <span class="range-v" id="qualityVal">3</span>
      </div>
    </div>

    <div class="s-row">
      <div class="s-label">Renk Modu<small>Renkli veya tek renk</small></div>
      <div class="s-ctrl">
        <select id="optColor">
          <option value="color">&#127912; Renkli</option>
          <option value="binary">&#11035; Siyah-Beyaz</option>
        </select>
      </div>
    </div>
  </div>

  <!-- CONVERT -->
  <button class="convert" id="btnConvert">
    <span class="spin"></span>
    <span id="btnText">&#9889; Vektöre Dönüştür</span>
  </button>

  <!-- PROGRESS -->
  <div class="progress" id="progress">
    <div class="prog-track"><div class="prog-bar" id="progBar"></div></div>
    <div class="prog-msg" id="progMsg">Hazırlanıyor...</div>
  </div>

  <!-- RESULT -->
  <div class="result" id="result">
    <div class="res-header">
      <div class="res-title">&#10004; Tamamlandı</div>
      <div class="res-stats" id="resStats">-</div>
    </div>

    <div class="comp" id="comp">
      <img id="compAfter" alt="Vector">
      <div class="comp-clip" id="compClip">
        <img id="compBefore" alt="Original">
      </div>
      <div class="comp-line" id="compLine"></div>
      <div class="comp-knob" id="compKnob">
        <svg viewBox="0 0 24 24"><path d="M8.5 5l-5 7 5 7M15.5 5l5 7-5 7"/></svg>
      </div>
      <div class="comp-tag tag-before">ÖNCE</div>
      <div class="comp-tag tag-after">SONRA</div>
    </div>

    <div class="dl-grid">
      <a class="dl-btn main" id="dlSvg">&#128229; SVG İndir</a>
      <a class="dl-btn" id="dlPng">&#128444; PNG (HD)</a>
      <a class="dl-btn" id="dlPdf">&#128196; PDF</a>
    </div>
  </div>

  <!-- FEATURES -->
  <div class="features">
    <div class="feat">
      <div class="feat-icon">&#128444;</div>
      <div class="feat-name">PNG &#8594; SVG</div>
      <div class="feat-desc">Şeffaflık korunur</div>
    </div>
    <div class="feat">
      <div class="feat-icon">&#128248;</div>
      <div class="feat-name">JPG &#8594; SVG</div>
      <div class="feat-desc">Logo ve fotoğraf</div>
    </div>
    <div class="feat">
      <div class="feat-icon">&#127912;</div>
      <div class="feat-name">WebP &#8594; SVG</div>
      <div class="feat-desc">Modern web formatı</div>
    </div>
    <div class="feat">
      <div class="feat-icon">&#128208;</div>
      <div class="feat-name">Sınırsız</div>
      <div class="feat-desc">Limit yok, ücretsiz</div>
    </div>
  </div>

  <!-- HOW IT WORKS -->
  <div class="how">
    <div class="how-title">Nasıl Çalışır?</div>
    <div class="step">
      <div class="step-n">1</div>
      <div><h3>Yükle</h3><p>PNG, JPG, WebP, GIF veya BMP görseli sürükle bırak ya da seç.</p></div>
    </div>
    <div class="step">
      <div class="step-n">2</div>
      <div><h3>Optimize Et</h3><p>Gürültü temizlenir, kenarlar düzeltilir, renkler sadeleştirilir.</p></div>
    </div>
    <div class="step">
      <div class="step-n">3</div>
      <div><h3>Vektörle</h3><p>Gelişmiş algoritma şekilleri Bézier eğrileriyle vektöre çevirir.</p></div>
    </div>
    <div class="step">
      <div class="step-n">4</div>
      <div><h3>İndir</h3><p>SVG, PNG veya PDF olarak indir. Sonsuza kadar ölçeklenebilir!</p></div>
    </div>
  </div>

</div><!-- /page -->

<!-- FOOTER -->
<div class="foot">
  VectorizeIt &mdash; Açık kaynak, ücretsiz vektörizasyon<br>
  Motor: <a href="https://github.com/visioncortex/vtracer" target="_blank">VTracer</a> (MIT Lisans) &middot; GPU gerektirmez
</div>

<!-- TOAST -->
<div class="toast" id="toast"></div>

<script>
(function(){
"use strict";

/* ===== DOM refs ===== */
var uploadEl   = document.getElementById("upload");
var fileInput  = document.getElementById("fileInput");
var previewEl  = document.getElementById("preview");
var prevImg    = document.getElementById("prevImg");
var fName      = document.getElementById("fName");
var fSize      = document.getElementById("fSize");
var btnRemove  = document.getElementById("btnRemove");
var settingsEl = document.getElementById("settings");
var btnConvert = document.getElementById("btnConvert");
var btnText    = document.getElementById("btnText");
var progressEl = document.getElementById("progress");
var progBar    = document.getElementById("progBar");
var progMsg    = document.getElementById("progMsg");
var resultEl   = document.getElementById("result");
var resStats   = document.getElementById("resStats");
var optQuality = document.getElementById("optQuality");
var qualityVal = document.getElementById("qualityVal");

var selectedFile = null;

/* ===== Quality slider ===== */
optQuality.addEventListener("input", function(){
    qualityVal.textContent = this.value;
});

/* ===== Drag-and-drop ===== */
uploadEl.addEventListener("dragover", function(e){
    e.preventDefault();
    uploadEl.classList.add("over");
});
uploadEl.addEventListener("dragleave", function(){
    uploadEl.classList.remove("over");
});
uploadEl.addEventListener("drop", function(e){
    e.preventDefault();
    uploadEl.classList.remove("over");
    if(e.dataTransfer.files && e.dataTransfer.files[0]){
        pickFile(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener("change", function(){
    if(this.files && this.files[0]) pickFile(this.files[0]);
});

/* ===== Paste ===== */
document.addEventListener("paste", function(e){
    var items = e.clipboardData ? e.clipboardData.items : [];
    for(var i = 0; i < items.length; i++){
        if(items[i].type.indexOf("image") === 0){
            var f = items[i].getAsFile();
            if(f) pickFile(f);
            break;
        }
    }
});

/* ===== Remove ===== */
btnRemove.addEventListener("click", resetAll);

/* ===== Convert ===== */
btnConvert.addEventListener("click", doConvert);

/* ===== Pick file ===== */
function pickFile(file){
    if(file.size > 10 * 1024 * 1024){
        toast("Dosya 10 MB'den büyük olamaz"); return;
    }
    selectedFile = file;
    var reader = new FileReader();
    reader.onload = function(ev){
        prevImg.src = ev.target.result;
        show(previewEl);
        show(settingsEl);
        show(btnConvert);
        hide(uploadEl);
        hide(resultEl);
    };
    reader.readAsDataURL(file);
    fName.textContent = file.name;
    fSize.textContent = formatBytes(file.size);
}

/* ===== Reset ===== */
function resetAll(){
    selectedFile = null;
    hide(previewEl);
    hide(settingsEl);
    hide(btnConvert);
    hide(progressEl);
    hide(resultEl);
    uploadEl.style.display = "";
    fileInput.value = "";
}

/* ===== Convert ===== */
function doConvert(){
    if(!selectedFile) return;

    btnConvert.disabled = true;
    btnConvert.classList.add("busy");
    btnText.textContent = "İşleniyor...";
    show(progressEl);
    hide(resultEl);

    var progress = 0;
    var messages = [
        "Görsel analiz ediliyor...",
        "Gürültü temizleniyor...",
        "Renkler optimize ediliyor...",
        "Kenarlar düzeltiliyor...",
        "Vektör path oluşturuluyor...",
        "Son rötuşlar yapılıyor..."
    ];
    var ticker = setInterval(function(){
        progress = Math.min(progress + Math.random() * 10, 93);
        progBar.style.width = progress + "%";
        var idx = Math.min(Math.floor(progress / 16), messages.length - 1);
        progMsg.textContent = messages[idx];
    }, 350);

    var form = new FormData();
    form.append("image", selectedFile);
    form.append("preset", document.getElementById("optPreset").value);
    form.append("quality", optQuality.value);
    form.append("colormode", document.getElementById("optColor").value);

    var t0 = Date.now();

    fetch("/api/vectorize", {method: "POST", body: form})
    .then(function(r){ return r.json(); })
    .then(function(data){
        clearInterval(ticker);
        var elapsed = ((Date.now() - t0) / 1000).toFixed(1);

        if(data.success){
            progBar.style.width = "100%";
            progMsg.textContent = "Tamamlandı!";
            setTimeout(function(){
                hide(progressEl);
                showResult(data, elapsed);
            }, 400);
        } else {
            throw new Error(data.error || "Bilinmeyen hata");
        }
    })
    .catch(function(err){
        clearInterval(ticker);
        hide(progressEl);
        toast("Hata: " + err.message);
    })
    .finally(function(){
        btnConvert.disabled = false;
        btnConvert.classList.remove("busy");
        btnText.textContent = "\u26A1 Vektöre Dönüştür";
    });
}

/* ===== Show result ===== */
function showResult(data, elapsed){
    show(resultEl);

    resStats.textContent = elapsed + "s \u00B7 " + data.svg_size + " \u00B7 " + data.path_count + " path";

    var ts = Date.now();
    document.getElementById("compBefore").src = prevImg.src;
    document.getElementById("compAfter").src  = "/api/preview/" + data.job_id + "?t=" + ts;

    var baseName = selectedFile.name.replace(/\.[^.]+$/, "");
    var dlSvg = document.getElementById("dlSvg");
    var dlPng = document.getElementById("dlPng");
    var dlPdf = document.getElementById("dlPdf");

    dlSvg.href = "/api/download/" + data.job_id + "?format=svg";
    dlSvg.setAttribute("download", baseName + ".svg");
    dlPng.href = "/api/download/" + data.job_id + "?format=png";
    dlPng.setAttribute("download", baseName + "_hd.png");
    dlPdf.href = "/api/download/" + data.job_id + "?format=pdf";
    dlPdf.setAttribute("download", baseName + ".pdf");

    resultEl.scrollIntoView({behavior: "smooth", block: "start"});
    initSlider();
}

/* ===== Comparison slider ===== */
function initSlider(){
    var comp     = document.getElementById("comp");
    var clip     = document.getElementById("compClip");
    var line     = document.getElementById("compLine");
    var knob     = document.getElementById("compKnob");
    var dragging = false;

    function setPos(clientX){
        var rect = comp.getBoundingClientRect();
        var pct  = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        var px   = (pct * 100) + "%";
        clip.style.width = px;
        line.style.left  = px;
        knob.style.left  = px;
    }

    comp.addEventListener("pointerdown", function(e){
        dragging = true;
        comp.setPointerCapture(e.pointerId);
        setPos(e.clientX);
    });
    comp.addEventListener("pointermove", function(e){
        if(dragging) setPos(e.clientX);
    });
    comp.addEventListener("pointerup", function(e){
        dragging = false;
        comp.releasePointerCapture(e.pointerId);
    });
    comp.addEventListener("pointercancel", function(e){
        dragging = false;
    });

    /* centre on load */
    var rect = comp.getBoundingClientRect();
    setPos(rect.left + rect.width * 0.5);
}

/* ===== Helpers ===== */
function show(el){ el.classList.add("on"); }
function hide(el){ el.classList.remove("on"); el.style.display = ""; }
function formatBytes(b){
    if(b >= 1048576) return (b/1048576).toFixed(1)+" MB";
    return Math.round(b/1024)+" KB";
}
function toast(msg){
    var el = document.getElementById("toast");
    el.textContent = msg;
    el.classList.add("on");
    setTimeout(function(){ el.classList.remove("on"); }, 3500);
}

})();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("")
    print("  VectorizeIt v3")
    print("  http://localhost:5000")
    print("")
    app.run(host="0.0.0.0", port=5000, debug=True)
