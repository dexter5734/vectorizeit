"""
VectorizeIt v4 - Professional Bitmap to Vector Converter
Optimized for free-tier hosting (Render, Railway, etc.)
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

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
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


def _cleanup(folder, max_age=1800):
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
# IMAGE PREPROCESSING — fast & effective
# ---------------------------------------------------------------------------


def preprocess_image(filepath):
    """
    Prepare raster for vectorisation.  Optimised for speed on free-tier CPUs.

    1. Down-scale to max 1500px  (was 2500 — cuts K-Means from 30s to 3s)
    2. Up-scale tiny images to 800px
    3. Binarise alpha
    4. Light bilateral filter
    5. Fast colour quantisation via resize trick instead of slow K-Means
    """

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return

    h, w = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]

    # 1. Resize — keep things manageable
    longest = max(w, h)
    if longest > 1500:
        scale = 1500.0 / longest
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    elif longest < 800:
        scale = 800.0 / longest
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    h, w = img.shape[:2]

    # 2. Alpha clean-up
    has_alpha = (channels == 4)
    if has_alpha:
        alpha = img[:, :, 3]
        img[:, :, 3] = np.where(alpha > 128, 255, 0).astype(np.uint8)

    # 3. Bilateral filter on colour channels — light pass
    if channels >= 3:
        bgr = img[:, :, :3]
        bgr = cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)
        img[:, :, :3] = bgr

    # 4. Fast colour quantisation (resize-trick: much faster than K-Means)
    #    Shrink → blur → expand  snaps similar colours together
    if channels >= 3:
        small_w = max(w // 4, 1)
        small_h = max(h // 4, 1)
        bgr = img[:, :, :3]
        small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small = cv2.medianBlur(small, 5)
        palette = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[:, :, :3] = palette

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
        color_precision=6,
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
        return jsonify(success=False, error="Desteklenmeyen format"), 400

    raw = f.read()
    if len(raw) > MAX_BYTES:
        return jsonify(success=False, error="Dosya 10 MB sınırını aşıyor"), 400

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
            svg_path,
            as_attachment=True,
            download_name=f"vector_{job_id}.svg",
        )

    if fmt == "png":
        try:
            import cairosvg
            png_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
            cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=2048)
            return send_file(
                png_path,
                as_attachment=True,
                download_name=f"vector_{job_id}.png",
            )
        except Exception:
            return send_file(
                svg_path,
                as_attachment=True,
                download_name=f"vector_{job_id}.svg",
            )

    if fmt == "pdf":
        try:
            import cairosvg
            pdf_path = os.path.join(OUTPUT_DIR, f"{job_id}.pdf")
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=f"vector_{job_id}.pdf",
            )
        except Exception:
            return send_file(
                svg_path,
                as_attachment=True,
                download_name=f"vector_{job_id}.svg",
            )

    return "Invalid format", 400


@app.route("/api/v1/vectorize", methods=["POST"])
def api_v1_vectorize():
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
# PAGE HTML
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>VectorizeIt</title>
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
--bg:#06060e;--s1:#0c0c18;--s2:#141424;--s3:#1c1c30;
--bd:#ffffff08;--tx:#f0f0f2;--tx2:#76768a;
--pr:#8b5cf6;--pr2:#7c3aed;--glow:rgba(139,92,246,.2);
--pink:#ec4899;--green:#34d399;--red:#f87171;
--r:14px;--rs:10px;
}
html{scroll-behavior:smooth}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--tx);min-height:100dvh;-webkit-tap-highlight-color:transparent;overflow-x:hidden}
img{display:block;max-width:100%}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:#fff1;border-radius:3px}

.page{max-width:620px;margin:0 auto;padding:0 16px 48px}

/* --- Header --- */
.hd{position:sticky;top:0;z-index:50;background:var(--bg);border-bottom:1px solid var(--bd);padding:14px 16px;display:flex;align-items:center;justify-content:space-between;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px)}
.hd-logo{font-size:18px;font-weight:800;background:linear-gradient(135deg,var(--pr),var(--pink));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hd-tag{font-size:9px;font-weight:800;letter-spacing:.7px;text-transform:uppercase;background:var(--green);color:#000;padding:3px 10px;border-radius:20px}

/* --- Upload --- */
.up{margin-top:20px;border:2px dashed #ffffff10;border-radius:var(--r);padding:48px 20px;text-align:center;cursor:pointer;background:var(--s1);position:relative;transition:all .2s}
.up:hover,.up.over{border-color:var(--pr);background:rgba(139,92,246,.03)}
.up:active{transform:scale(.99)}
.up input{position:absolute;inset:0;opacity:0;cursor:pointer}
.up-ic{width:52px;height:52px;margin:0 auto 14px;background:var(--s3);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px}
.up-t{font-size:15px;font-weight:600;margin-bottom:3px}
.up-t span{color:var(--pr)}
.up-s{font-size:11px;color:var(--tx2)}

/* --- Preview --- */
.pv{display:none;background:var(--s1);border-radius:var(--r);overflow:hidden;margin-top:14px;border:1px solid var(--bd)}
.pv.on{display:block}
.pv img{width:100%;max-height:260px;object-fit:contain;background:#07070d}
.pv-bar{display:flex;justify-content:space-between;align-items:center;padding:10px 14px}
.pv-name{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:60%}
.pv-sz{font-size:10px;color:var(--tx2);margin-left:6px}
.pv-rm{background:none;border:none;color:var(--red);font-size:12px;cursor:pointer;padding:6px 10px;border-radius:8px;font-weight:600}

/* --- Settings --- */
.cfg{display:none;background:var(--s1);border-radius:var(--r);padding:16px;margin-top:10px;border:1px solid var(--bd)}
.cfg.on{display:block}
.cfg-h{font-size:10px;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
.cfg-r{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--bd)}
.cfg-r:last-child{border:none}
.cfg-l{font-size:13px}.cfg-l small{display:block;font-size:10px;color:var(--tx2);margin-top:1px}
.cfg-c{display:flex;align-items:center;gap:6px}
.cfg-c select{background:var(--s3);border:1px solid var(--bd);color:var(--tx);padding:6px 8px;border-radius:var(--rs);font-size:11px;font-family:inherit}
.cfg-c input[type=range]{-webkit-appearance:none;width:90px;height:4px;border-radius:2px;background:var(--s3);outline:none}
.cfg-c input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--pr);cursor:pointer;box-shadow:0 0 6px var(--glow)}
.rv{font-size:11px;font-weight:700;color:var(--pr);min-width:14px;text-align:right}

/* --- Button --- */
.btn{display:none;width:100%;margin-top:14px;padding:15px;border:none;border-radius:var(--r);background:linear-gradient(135deg,var(--pr),var(--pr2));color:#fff;font-size:15px;font-weight:700;font-family:inherit;cursor:pointer;transition:all .15s;position:relative}
.btn.on{display:block}
.btn:active{transform:scale(.98)}
.btn:disabled{opacity:.45;cursor:wait;transform:none}
.btn .sp{display:none;width:15px;height:15px;border:2px solid #fff4;border-top-color:#fff;border-radius:50%;animation:spin .5s linear infinite;margin-right:8px;vertical-align:middle}
.btn.ld .sp{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}

/* --- Progress --- */
.prg{display:none;margin-top:14px}
.prg.on{display:block}
.prg-t{height:3px;background:var(--s3);border-radius:2px;overflow:hidden}
.prg-b{height:100%;width:0%;background:linear-gradient(90deg,var(--pr),var(--green));border-radius:2px;transition:width .3s}
.prg-m{font-size:11px;color:var(--tx2);text-align:center;margin-top:6px}

/* --- Result --- */
.res{display:none;margin-top:18px}
.res.on{display:block}
.res-hd{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.res-ok{font-size:14px;font-weight:700;color:var(--green)}
.res-st{font-size:11px;color:var(--tx2)}

/* --- Comparison slider --- */
.cmp{position:relative;width:100%;aspect-ratio:3/2;border-radius:12px;overflow:hidden;cursor:ew-resize;background:#07070d;touch-action:none;border:1px solid var(--bd);margin-bottom:12px}
.cmp img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;pointer-events:none;user-select:none;-webkit-user-select:none;-webkit-user-drag:none}
.cmp-cl{position:absolute;top:0;left:0;width:50%;height:100%;overflow:hidden}
.cmp-cl img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain}
.cmp-ln{position:absolute;top:0;left:50%;width:2px;height:100%;background:var(--pr);transform:translateX(-50%);z-index:4;box-shadow:0 0 8px var(--glow)}
.cmp-kb{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:32px;height:32px;background:var(--pr);border-radius:50%;z-index:5;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 10px #0008,0 0 12px var(--glow)}
.cmp-kb svg{width:14px;height:14px;fill:#fff;stroke:#fff;stroke-width:.5}
.cmp-tg{position:absolute;top:8px;padding:3px 8px;border-radius:5px;font-size:9px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;z-index:3}
.tg-b{left:8px;background:rgba(244,114,182,.8)}
.tg-a{right:8px;background:rgba(52,211,153,.8);color:#000}

/* --- Downloads --- */
.dl-g{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.dl{display:flex;align-items:center;justify-content:center;gap:6px;padding:12px;border:1px solid var(--bd);border-radius:11px;background:var(--s1);color:var(--tx);font-size:13px;font-weight:600;font-family:inherit;cursor:pointer;text-decoration:none;transition:all .15s}
.dl:active{background:var(--pr);border-color:var(--pr);transform:scale(.97)}
.dl.dm{grid-column:1/-1;background:linear-gradient(135deg,var(--pr),var(--pr2));border-color:transparent;font-size:14px;padding:14px}

/* --- Features --- */
.ft{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:28px}
.fc{background:var(--s1);border-radius:12px;padding:16px 12px;text-align:center;border:1px solid var(--bd)}
.fc-i{font-size:24px;margin-bottom:6px}
.fc-n{font-size:12px;font-weight:700}
.fc-d{font-size:10px;color:var(--tx2);margin-top:2px}

/* --- Steps --- */
.hw{margin-top:28px}
.hw-t{font-size:17px;font-weight:800;text-align:center;margin-bottom:14px}
.st{display:flex;gap:12px;background:var(--s1);border-radius:12px;padding:14px;margin-bottom:7px;border:1px solid var(--bd)}
.st-n{width:30px;height:30px;min-width:30px;background:linear-gradient(135deg,var(--pr),var(--pink));border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px}
.st h3{font-size:13px;font-weight:700;margin-bottom:1px}
.st p{font-size:11px;color:var(--tx2);line-height:1.5}

/* --- Footer --- */
.fo{text-align:center;padding:24px 16px;color:var(--tx2);font-size:10px;border-top:1px solid var(--bd);margin-top:36px;line-height:1.7}
.fo a{color:var(--pr);font-weight:600;text-decoration:none}

/* --- Toast --- */
.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%) translateY(90px);background:var(--s2);color:var(--tx);padding:11px 20px;border-radius:10px;font-size:13px;z-index:99;transition:transform .3s cubic-bezier(.34,1.56,.64,1);border:1px solid var(--bd);box-shadow:0 6px 24px #0006;white-space:nowrap}
.toast.on{transform:translateX(-50%) translateY(0)}

@media(max-width:420px){
.page{padding:0 10px 36px}
.up{padding:36px 14px}
.dl-g{grid-template-columns:1fr}
.cmp{aspect-ratio:1/1}
}
</style>
</head>
<body>

<div class="hd">
<div class="hd-logo">&#9889; VectorizeIt</div>
<div class="hd-tag">&#10003; &#220;cretsiz</div>
</div>

<div class="page">

<div class="up" id="U">
<input type="file" id="F" accept=".png,.jpg,.jpeg,.gif,.bmp,.webp">
<div class="up-ic">&#128193;</div>
<div class="up-t"><span>G&#246;rsel se&#231;</span> veya s&#252;r&#252;kle b&#305;rak</div>
<div class="up-s">PNG &#183; JPG &#183; WebP &#183; GIF &#183; BMP &#8212; maks 10 MB</div>
</div>

<div class="pv" id="P">
<img id="PI" alt="">
<div class="pv-bar">
<div style="display:flex;align-items:baseline;min-width:0">
<span class="pv-name" id="PN">-</span>
<span class="pv-sz" id="PS">-</span>
</div>
<button class="pv-rm" id="PR">&#10005; Kald&#305;r</button>
</div>
</div>

<div class="cfg" id="C">
<div class="cfg-h">&#9881; Ayarlar</div>
<div class="cfg-r">
<div class="cfg-l">G&#246;rsel Tipi<small>&#304;&#231;eri&#287;e g&#246;re optimize eder</small></div>
<div class="cfg-c"><select id="oP">
<option value="logo">&#127919; Logo</option>
<option value="illustration">&#127912; &#304;ll&#252;strasyon</option>
<option value="photo">&#128248; Foto&#287;raf</option>
<option value="sketch">&#9999;&#65039; &#199;izim</option>
</select></div>
</div>
<div class="cfg-r">
<div class="cfg-l">Kalite<small>Y&#252;ksek = detayl&#305;</small></div>
<div class="cfg-c">
<input type="range" id="oQ" min="1" max="5" value="3">
<span class="rv" id="QV">3</span>
</div>
</div>
<div class="cfg-r">
<div class="cfg-l">Renk<small>Renkli veya tek renk</small></div>
<div class="cfg-c"><select id="oC">
<option value="color">&#127912; Renkli</option>
<option value="binary">&#11035; S-B</option>
</select></div>
</div>
</div>

<button class="btn" id="B"><span class="sp"></span><span id="BT">&#9889; Vekt&#246;re D&#246;n&#252;&#351;t&#252;r</span></button>

<div class="prg" id="G">
<div class="prg-t"><div class="prg-b" id="GB"></div></div>
<div class="prg-m" id="GM">Haz&#305;rlan&#305;yor...</div>
</div>

<div class="res" id="R">
<div class="res-hd">
<div class="res-ok">&#10004; Tamamland&#305;</div>
<div class="res-st" id="RS">-</div>
</div>
<div class="cmp" id="CM">
<img id="CA" alt="">
<div class="cmp-cl" id="CL"><img id="CB" alt=""></div>
<div class="cmp-ln" id="LN"></div>
<div class="cmp-kb" id="KB"><svg viewBox="0 0 24 24"><path d="M8 5L3 12l5 7M16 5l5 7-5 7" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
<div class="cmp-tg tg-b">&#214;NCE</div>
<div class="cmp-tg tg-a">SONRA</div>
</div>
<div class="dl-g">
<a class="dl dm" id="dS">&#128229; SVG &#304;ndir</a>
<a class="dl" id="dP">&#128444; PNG</a>
<a class="dl" id="dF">&#128196; PDF</a>
</div>
</div>

<div class="ft">
<div class="fc"><div class="fc-i">&#128444;</div><div class="fc-n">PNG &#8594; SVG</div><div class="fc-d">&#350;effafl&#305;k korunur</div></div>
<div class="fc"><div class="fc-i">&#128248;</div><div class="fc-n">JPG &#8594; SVG</div><div class="fc-d">Logo ve foto&#287;raf</div></div>
<div class="fc"><div class="fc-i">&#127912;</div><div class="fc-n">WebP &#8594; SVG</div><div class="fc-d">Modern format</div></div>
<div class="fc"><div class="fc-i">&#128208;</div><div class="fc-n">S&#305;n&#305;rs&#305;z</div><div class="fc-d">Limit yok</div></div>
</div>

<div class="hw">
<div class="hw-t">Nas&#305;l &#199;al&#305;&#351;&#305;r?</div>
<div class="st"><div class="st-n">1</div><div><h3>Y&#252;kle</h3><p>G&#246;rselini se&#231; veya s&#252;r&#252;kle b&#305;rak.</p></div></div>
<div class="st"><div class="st-n">2</div><div><h3>Optimize Et</h3><p>G&#252;r&#252;lt&#252; temizlenir, renkler sadele&#351;tirilir.</p></div></div>
<div class="st"><div class="st-n">3</div><div><h3>Vekt&#246;rle</h3><p>&#350;ekiller B&#233;zier e&#287;rileriyle vekt&#246;re &#231;evrilir.</p></div></div>
<div class="st"><div class="st-n">4</div><div><h3>&#304;ndir</h3><p>SVG, PNG veya PDF olarak indir.</p></div></div>
</div>

</div>

<div class="fo">
VectorizeIt &#8212; A&#231;&#305;k kaynak vekt&#246;rizasyon<br>
Motor: <a href="https://github.com/visioncortex/vtracer" target="_blank">VTracer</a> (MIT) &#183; GPU gerektirmez
</div>

<div class="toast" id="T"></div>

<script>
(function(){
"use strict";

var U=document.getElementById("U"),
    F=document.getElementById("F"),
    P=document.getElementById("P"),
    PI=document.getElementById("PI"),
    PN=document.getElementById("PN"),
    PS=document.getElementById("PS"),
    PR=document.getElementById("PR"),
    C=document.getElementById("C"),
    B=document.getElementById("B"),
    BT=document.getElementById("BT"),
    G=document.getElementById("G"),
    GB=document.getElementById("GB"),
    GM=document.getElementById("GM"),
    R=document.getElementById("R"),
    RS=document.getElementById("RS"),
    oQ=document.getElementById("oQ"),
    QV=document.getElementById("QV"),
    file=null;

oQ.oninput=function(){QV.textContent=this.value};

U.addEventListener("dragover",function(e){e.preventDefault();U.classList.add("over")});
U.addEventListener("dragleave",function(){U.classList.remove("over")});
U.addEventListener("drop",function(e){e.preventDefault();U.classList.remove("over");if(e.dataTransfer.files[0])pick(e.dataTransfer.files[0])});
F.addEventListener("change",function(){if(this.files[0])pick(this.files[0])});

document.addEventListener("paste",function(e){
var items=e.clipboardData?e.clipboardData.items:[];
for(var i=0;i<items.length;i++){if(items[i].type.indexOf("image")===0){var f=items[i].getAsFile();if(f)pick(f);break}}
});

PR.addEventListener("click",reset);
B.addEventListener("click",convert);

function pick(f){
if(f.size>10485760){toast("Dosya 10 MB'den b\u00fcy\u00fck");return}
file=f;
var r=new FileReader();
r.onload=function(e){
PI.src=e.target.result;
show(P);show(C);show(B);U.style.display="none";hide(R);
};
r.readAsDataURL(f);
PN.textContent=f.name;
PS.textContent=fb(f.size);
}

function reset(){
file=null;hide(P);hide(C);hide(B);hide(G);hide(R);
U.style.display="";F.value="";
}

function convert(){
if(!file)return;
B.disabled=true;B.classList.add("ld");BT.textContent="\u0130\u015fleniyor...";
show(G);hide(R);

var p=0;
var msgs=["G\u00f6rsel analiz ediliyor...","G\u00fcr\u00fclt\u00fc temizleniyor...","Renkler optimize ediliyor...","Vekt\u00f6r path olu\u015fturuluyor...","Tamamlan\u0131yor..."];
var iv=setInterval(function(){
p=Math.min(p+Math.random()*8,93);
GB.style.width=p+"%";
GM.textContent=msgs[Math.min(Math.floor(p/19),msgs.length-1)];
},500);

var fd=new FormData();
fd.append("image",file);
fd.append("preset",document.getElementById("oP").value);
fd.append("quality",oQ.value);
fd.append("colormode",document.getElementById("oC").value);

var t0=Date.now();

fetch("/api/vectorize",{method:"POST",body:fd})
.then(function(r){
if(!r.ok) throw new Error("Sunucu hatas\u0131: "+r.status);
return r.json();
})
.then(function(d){
clearInterval(iv);
if(d.success){
GB.style.width="100%";GM.textContent="Tamamland\u0131!";
setTimeout(function(){hide(G);showRes(d,((Date.now()-t0)/1000).toFixed(1))},350);
}else{throw new Error(d.error||"Bilinmeyen hata")}
})
.catch(function(e){clearInterval(iv);hide(G);toast(e.message)})
.finally(function(){B.disabled=false;B.classList.remove("ld");BT.textContent="\u26A1 Vekt\u00f6re D\u00f6n\u00fc\u015ft\u00fcr"});
}

function showRes(d,sec){
show(R);
RS.textContent=sec+"s \u00b7 "+d.svg_size+" \u00b7 "+d.path_count+" path";

document.getElementById("CB").src=PI.src;
document.getElementById("CA").src="/api/preview/"+d.job_id+"?t="+Date.now();

var bn=file.name.replace(/\.[^.]+$/,"");
var dS=document.getElementById("dS"),dP=document.getElementById("dP"),dF=document.getElementById("dF");
dS.href="/api/download/"+d.job_id+"?format=svg";dS.setAttribute("download",bn+".svg");
dP.href="/api/download/"+d.job_id+"?format=png";dP.setAttribute("download",bn+"_hd.png");
dF.href="/api/download/"+d.job_id+"?format=pdf";dF.setAttribute("download",bn+".pdf");

R.scrollIntoView({behavior:"smooth",block:"start"});
initComp();
}

function initComp(){
var cm=document.getElementById("CM"),cl=document.getElementById("CL"),ln=document.getElementById("LN"),kb=document.getElementById("KB");
var drag=false;
function mv(x){var r=cm.getBoundingClientRect();var p=Math.max(0,Math.min(1,(x-r.left)/r.width));var s=p*100+"%";cl.style.width=s;ln.style.left=s;kb.style.left=s}
cm.addEventListener("pointerdown",function(e){drag=true;cm.setPointerCapture(e.pointerId);mv(e.clientX)});
cm.addEventListener("pointermove",function(e){if(drag)mv(e.clientX)});
cm.addEventListener("pointerup",function(e){drag=false;cm.releasePointerCapture(e.pointerId)});
cm.addEventListener("pointercancel",function(){drag=false});
var r=cm.getBoundingClientRect();mv(r.left+r.width*.5);
}

function show(el){el.classList.add("on")}
function hide(el){el.classList.remove("on");el.style.display=""}
function fb(b){return b>=1048576?(b/1048576).toFixed(1)+" MB":Math.round(b/1024)+" KB"}
function toast(m){var t=document.getElementById("T");t.textContent=m;t.classList.add("on");setTimeout(function(){t.classList.remove("on")},3500)}

})();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  VectorizeIt v4\n  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
