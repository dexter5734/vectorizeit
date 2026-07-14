"""
⚡ VectorizeIt v2 — Ücretsiz Bitmap → Vektör
Kaliteli ön-işleme + VTracer motoru
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import vtracer
import cv2
import numpy as np
from PIL import Image
import os, uuid, time, io

app = Flask(__name__)
UPLOAD = "uploads"
OUTPUT = "outputs"
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)
ALLOWED = {"png","jpg","jpeg","gif","bmp","webp"}

def allowed(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED

def cleanup(folder, age=3600):
    now = time.time()
    for f in os.listdir(folder):
        fp = os.path.join(folder, f)
        if os.path.isfile(fp) and now - os.path.getmtime(fp) > age:
            os.remove(fp)

def preprocess(path):
    """Görseli vektörizasyona hazırla — kaliteyi dramatik artırır"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return

    h, w = img.shape[:2]

    # 1) Küçük görselleri upscale et (en az 1000px)
    min_dim = 1000
    if max(w, h) < min_dim:
        scale = min_dim / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Çok büyük görselleri küçült
    elif max(w, h) > 2500:
        scale = 2500 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 2) Alpha kanalı varsa düzelt
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        # Alpha'yı binary yap (yarı saydam sorun yaratır)
        alpha = np.where(alpha > 128, 255, 0).astype(np.uint8)
        img[:, :, 3] = alpha

    # 3) Bilateral filter — kenarları koruyarak gürültüyü temizle
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.bilateralFilter(img, 9, 60, 60)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        bgr = cv2.bilateralFilter(bgr, 9, 60, 60)
        img[:, :, :3] = bgr

    # 4) Renk kuantizasyonu — benzer renkleri birleştir
    if len(img.shape) == 3:
        ch = img.shape[2]
        has_alpha = ch == 4
        if has_alpha:
            bgr = img[:, :, :3]
            alpha_ch = img[:, :, 3:]
        else:
            bgr = img

        pixels = bgr.reshape(-1, 3).astype(np.float32)
        K = 24  # max renk sayısı
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(bgr.shape)

        if has_alpha:
            img = np.dstack([quantized, alpha_ch])
        else:
            img = quantized

    cv2.imwrite(path, img)


# ==================== HTML ====================
HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>VectorizeIt — Görsel → Vektör</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --p:#7C3AED;--p2:#6D28D9;--bg:#09090B;--s1:#18181B;--s2:#27272A;
  --t:#FAFAFA;--t2:#A1A1AA;--acc:#F43F5E;--ok:#10B981;--r:14px;
}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--t);min-height:100dvh;-webkit-tap-highlight-color:transparent}

/* HEADER */
header{padding:16px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #ffffff08}
.logo{font-size:20px;font-weight:800;background:linear-gradient(135deg,#7C3AED,#F43F5E);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.tag{font-size:10px;background:#10B981;color:#000;padding:3px 10px;border-radius:20px;font-weight:700;letter-spacing:.5px}

.wrap{max-width:600px;margin:0 auto;padding:16px}

/* UPLOAD */
.drop{border:2px dashed #ffffff15;border-radius:var(--r);padding:48px 20px;text-align:center;cursor:pointer;background:var(--s1);position:relative;transition:all .2s}
.drop:active,.drop.over{border-color:var(--p);background:#7C3AED10;transform:scale(.99)}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer}
.drop-icon{font-size:44px;margin-bottom:12px}
.drop-t{font-size:15px;color:var(--t2)}.drop-t b{color:var(--p)}
.drop-h{font-size:11px;color:var(--t2);opacity:.5;margin-top:6px}

/* FILE PREVIEW */
.file-box{display:none;background:var(--s1);border-radius:var(--r);overflow:hidden;margin-bottom:16px}
.file-box.on{display:block}
.file-box img{width:100%;max-height:260px;object-fit:contain;background:#0a0a0a;display:block}
.file-meta{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;font-size:13px}
.file-name{color:var(--t);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:70%}
.file-rm{background:none;border:none;color:var(--acc);font-size:13px;cursor:pointer;padding:6px}

/* SETTINGS */
.cfg{display:none;background:var(--s1);border-radius:var(--r);padding:16px;margin-bottom:16px}
.cfg.on{display:block}
.cfg-title{font-size:11px;font-weight:700;color:var(--t2);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
.cfg-row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #ffffff06}
.cfg-row:last-child{border:none}
.cfg-l{font-size:13px}.cfg-l small{display:block;font-size:10px;color:var(--t2);margin-top:1px}
.cfg-r{display:flex;align-items:center;gap:6px}
.cfg-r select{background:var(--s2);border:1px solid #ffffff10;color:var(--t);padding:7px 10px;border-radius:8px;font-size:12px}
.cfg-r input[type=range]{-webkit-appearance:none;width:100px;height:5px;border-radius:3px;background:var(--s2);outline:none}
.cfg-r input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--p);cursor:pointer}
.rv{font-size:11px;color:var(--p);min-width:20px;text-align:right}

/* BUTTON */
.go{display:none;width:100%;padding:16px;border:none;border-radius:var(--r);background:linear-gradient(135deg,var(--p),var(--p2));color:#fff;font-size:16px;font-weight:700;cursor:pointer;margin-bottom:16px;transition:all .2s;position:relative;overflow:hidden}
.go.on{display:block}
.go:active{transform:scale(.98)}
.go:disabled{opacity:.5;cursor:wait}
.go .spinner{display:none;width:18px;height:18px;border:2px solid #fff3;border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;margin-right:8px;vertical-align:middle}
.go.loading .spinner{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}

/* PROGRESS */
.prog{display:none;margin-bottom:16px}
.prog.on{display:block}
.prog-bar{height:3px;background:var(--s2);border-radius:3px;overflow:hidden}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--p),var(--ok));width:0%;transition:width .4s ease}
.prog-txt{font-size:12px;color:var(--t2);text-align:center;margin-top:6px}

/* RESULT */
.result{display:none;margin-bottom:16px}
.result.on{display:block}
.res-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.res-ok{font-size:15px;font-weight:700;color:var(--ok)}
.res-stats{font-size:11px;color:var(--t2)}

/* COMPARISON */
.comp{position:relative;width:100%;aspect-ratio:3/2;border-radius:12px;overflow:hidden;cursor:ew-resize;background:#0a0a0a;touch-action:none;margin-bottom:14px}
.comp img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;pointer-events:none;user-select:none;-webkit-user-select:none}
.comp .clip{position:absolute;top:0;left:0;width:50%;height:100%;overflow:hidden}
.comp .clip img{width:100%;min-width:100%}
.comp .line{position:absolute;top:0;left:50%;width:2px;height:100%;background:var(--p);z-index:3;transform:translateX(-50%)}
.comp .knob{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:32px;height:32px;background:var(--p);border-radius:50%;z-index:4;display:flex;align-items:center;justify-content:center;font-size:11px;box-shadow:0 2px 12px #0008;color:#fff;font-weight:700}
.comp .tag-l,.comp .tag-r{position:absolute;top:8px;padding:3px 8px;border-radius:6px;font-size:10px;font-weight:700;z-index:2}
.comp .tag-l{left:8px;background:#F43F5Ecc}
.comp .tag-r{right:8px;background:#10B981cc;color:#000}

/* DOWNLOAD */
.dl-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.dl{display:flex;align-items:center;justify-content:center;gap:6px;padding:13px;border:1px solid #ffffff10;border-radius:12px;background:var(--s1);color:var(--t);font-size:13px;font-weight:600;cursor:pointer;text-decoration:none;transition:all .15s}
.dl:active{background:var(--p);border-color:var(--p)}
.dl.main{grid-column:1/-1;background:linear-gradient(135deg,var(--p),var(--p2));border-color:var(--p);font-size:15px;padding:15px}

/* INFO CARDS */
.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:24px 0}
.card{background:var(--s1);border-radius:12px;padding:14px;text-align:center}
.card .ic{font-size:24px;margin-bottom:6px}
.card .nm{font-weight:700;font-size:13px}
.card .ds{font-size:10px;color:var(--t2);margin-top:3px}

/* STEPS */
.steps{margin:24px 0}
.steps-t{font-size:18px;font-weight:800;text-align:center;margin-bottom:16px}
.step{display:flex;gap:12px;background:var(--s1);border-radius:12px;padding:14px;margin-bottom:8px}
.step-n{width:30px;height:30px;min-width:30px;background:linear-gradient(135deg,var(--p),var(--acc));border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px}
.step h3{font-size:14px;margin-bottom:2px}
.step p{font-size:12px;color:var(--t2)}

footer{text-align:center;padding:24px 16px;color:var(--t2);font-size:11px;border-top:1px solid #ffffff06;margin-top:32px}
footer a{color:var(--p);text-decoration:none}

.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%) translateY(80px);background:var(--s2);color:var(--t);padding:12px 20px;border-radius:10px;font-size:13px;z-index:99;transition:transform .3s;border:1px solid #ffffff10}
.toast.on{transform:translateX(-50%) translateY(0)}

@media(max-width:420px){
  .wrap{padding:10px}
  .drop{padding:36px 14px}
  .dl-grid{grid-template-columns:1fr}
  .comp{aspect-ratio:1/1}
}
</style>
</head>
<body>

<header>
  <div class="logo">⚡ VectorizeIt</div>
  <div class="tag">ÜCRETSİZ</div>
</header>

<div class="wrap">

<!-- UPLOAD -->
<div class="drop" id="drop">
  <input type="file" id="fi" accept=".png,.jpg,.jpeg,.gif,.bmp,.webp">
  <div class="drop-icon">📁</div>
  <div class="drop-t"><b>Görsel seç</b> veya sürükle bırak</div>
  <div class="drop-h">PNG · JPG · WebP · GIF · BMP — maks 10 MB</div>
</div>

<!-- FILE -->
<div class="file-box" id="fb">
  <img id="prev">
  <div class="file-meta">
    <span class="file-name" id="fn">—</span>
    <button class="file-rm" onclick="reset()">✕ Kaldır</button>
  </div>
</div>

<!-- SETTINGS -->
<div class="cfg" id="cfg">
  <div class="cfg-title">⚙️ Ayarlar</div>
  <div class="cfg-row">
    <div class="cfg-l">Tip<small>Görsel türüne göre seç</small></div>
    <div class="cfg-r"><select id="preset">
      <option value="logo">🎯 Logo / İkon</option>
      <option value="illustration">🎨 İllüstrasyon</option>
      <option value="photo">📸 Fotoğraf</option>
      <option value="sketch">✏️ Çizim / Sketch</option>
    </select></div>
  </div>
  <div class="cfg-row">
    <div class="cfg-l">Kalite<small>Yüksek = daha detaylı</small></div>
    <div class="cfg-r">
      <input type="range" id="quality" min="1" max="5" value="3">
      <span class="rv" id="qv">3</span>
    </div>
  </div>
  <div class="cfg-row">
    <div class="cfg-l">Mod<small>Renkli veya tek renk</small></div>
    <div class="cfg-r"><select id="cmode">
      <option value="color">🎨 Renkli</option>
      <option value="binary">⬛ Siyah-Beyaz</option>
    </select></div>
  </div>
</div>

<!-- BUTTON -->
<button class="go" id="go" onclick="convert()">
  <span class="spinner"></span>
  <span class="go-txt">⚡ Vektöre Dönüştür</span>
</button>

<!-- PROGRESS -->
<div class="prog" id="prog">
  <div class="prog-bar"><div class="prog-fill" id="pf"></div></div>
  <div class="prog-txt" id="pt">Hazırlanıyor...</div>
</div>

<!-- RESULT -->
<div class="result" id="res">
  <div class="res-head">
    <div class="res-ok">✅ Tamamlandı</div>
    <div class="res-stats" id="stats"></div>
  </div>

  <div class="comp" id="comp">
    <img id="imgB">
    <div class="clip" id="clip"><img id="imgA"></div>
    <div class="line" id="cline"></div>
    <div class="knob" id="knob">⇔</div>
    <div class="tag-l">ÖNCE</div>
    <div class="tag-r">SONRA</div>
  </div>

  <div class="dl-grid">
    <a class="dl main" id="dSvg">📥 SVG İndir</a>
    <a class="dl" id="dPng">🖼️ PNG</a>
    <a class="dl" id="dPdf">📄 PDF</a>
  </div>
</div>

<!-- CARDS -->
<div class="cards">
  <div class="card"><div class="ic">🖼️</div><div class="nm">PNG → SVG</div><div class="ds">Şeffaflık korunur</div></div>
  <div class="card"><div class="ic">📸</div><div class="nm">JPG → SVG</div><div class="ds">Fotoğraf & logo</div></div>
  <div class="card"><div class="ic">🎨</div><div class="nm">WebP → SVG</div><div class="ds">Modern format</div></div>
  <div class="card"><div class="ic">📐</div><div class="nm">BMP → SVG</div><div class="ds">Bitmap dönüşüm</div></div>
</div>

<!-- STEPS -->
<div class="steps">
  <div class="steps-t">Nasıl Çalışır?</div>
  <div class="step"><div class="step-n">1</div><div><h3>Yükle</h3><p>PNG, JPG veya WebP görselini seç</p></div></div>
  <div class="step"><div class="step-n">2</div><div><h3>İşle</h3><p>AI kenarları, renkleri ve şekilleri analiz eder</p></div></div>
  <div class="step"><div class="step-n">3</div><div><h3>İndir</h3><p>SVG, PNG veya PDF olarak indir</p></div></div>
</div>

</div>

<footer>
  VectorizeIt — Açık kaynak vektörizasyon<br>
  Motor: <a href="https://github.com/visioncortex/vtracer" target="_blank">VTracer</a> (MIT) • GPU gerektirmez
</footer>

<div class="toast" id="toast"></div>

<script>
let file=null, jid=null;
const $=id=>document.getElementById(id);

// Range
$('quality').oninput=function(){$('qv').textContent=this.value};

// Drag & drop
const drop=$('drop');
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');if(e.dataTransfer.files[0])pick(e.dataTransfer.files[0])};
$('fi').onchange=e=>{if(e.target.files[0])pick(e.target.files[0])};

// Paste
document.onpaste=e=>{
  for(const i of(e.clipboardData?.items||[]))
    if(i.type.startsWith('image/')){const f=i.getAsFile();if(f)pick(f);break}
};

function pick(f){
  if(f.size>10485760)return msg('❌ Maks 10 MB');
  file=f;
  const r=new FileReader();
  r.onload=e=>{
    $('prev').src=e.target.result;
    $('fb').classList.add('on');
    drop.style.display='none';
    $('cfg').classList.add('on');
    $('go').classList.add('on');
    $('res').classList.remove('on');
  };
  r.readAsDataURL(f);
  $('fn').textContent=f.name+' ('+Math.round(f.size/1024)+' KB)';
}

function reset(){
  file=null;jid=null;
  $('fb').classList.remove('on');
  $('cfg').classList.remove('on');
  $('go').classList.remove('on');
  $('prog').classList.remove('on');
  $('res').classList.remove('on');
  drop.style.display='';
  $('fi').value='';
}

async function convert(){
  if(!file)return;
  const btn=$('go');
  btn.disabled=true;btn.classList.add('loading');
  $('go-txt')&&($('go-txt').textContent='İşleniyor...');
  $('prog').classList.add('on');
  $('res').classList.remove('on');

  let p=0;
  const msgs=['🔍 Görsel analiz ediliyor...','🎨 Renkler optimize ediliyor...','✨ Kenarlar düzeltiliyor...','📐 Vektör path oluşturuluyor...','🎯 Son düzenlemeler...'];
  const iv=setInterval(()=>{
    p=Math.min(p+Math.random()*12,92);
    $('pf').style.width=p+'%';
    $('pt').textContent=msgs[Math.min(Math.floor(p/20),msgs.length-1)];
  },400);

  const fd=new FormData();
  fd.append('image',file);
  fd.append('preset',$('preset').value);
  fd.append('quality',$('quality').value);
  fd.append('colormode',$('cmode').value);

  try{
    const t0=Date.now();
    const r=await fetch('/api/vectorize',{method:'POST',body:fd});
    const d=await r.json();
    const sec=((Date.now()-t0)/1000).toFixed(1);
    clearInterval(iv);
    if(d.success){
      $('pf').style.width='100%';
      $('pt').textContent='✅ Tamamlandı!';
      setTimeout(()=>{
        $('prog').classList.remove('on');
        showResult(d,sec);
      },400);
    }else throw new Error(d.error||'Hata');
  }catch(e){
    clearInterval(iv);
    $('prog').classList.remove('on');
    msg('❌ '+e.message);
  }
  btn.disabled=false;btn.classList.remove('loading');
  const gt=btn.querySelector('.go-txt');if(gt)gt.textContent='⚡ Vektöre Dönüştür';
}

function showResult(d,sec){
  $('res').classList.add('on');
  jid=d.job_id;
  $('stats').textContent=sec+'s · '+d.svg_size+' · '+d.path_count+' path';
  $('imgB').src='/api/preview/'+d.job_id+'?t='+Date.now();
  $('imgA').src=$('prev').src;
  const base=file.name.replace(/\.[^.]+$/,'');
  $('dSvg').href='/api/download/'+d.job_id+'?format=svg';$('dSvg').download=base+'.svg';
  $('dPng').href='/api/download/'+d.job_id+'?format=png';$('dPng').download=base+'_hd.png';
  $('dPdf').href='/api/download/'+d.job_id+'?format=pdf';$('dPdf').download=base+'.pdf';
  $('res').scrollIntoView({behavior:'smooth',block:'start'});
  initSlider();
}

/* ===== COMPARISON SLIDER ===== */
function initSlider(){
  const c=$('comp');
  let active=false;
  function mv(x){
    const r=c.getBoundingClientRect();
    let p=Math.max(0,Math.min(1,(x-r.left)/r.width));
    $('clip').style.width=(p*100)+'%';
    $('cline').style.left=(p*100)+'%';
    $('knob').style.left=(p*100)+'%';
  }
  // Mouse
  c.onmousedown=e=>{active=true;mv(e.clientX);e.preventDefault()};
  document.onmousemove=e=>{if(active)mv(e.clientX)};
  document.onmouseup=()=>active=false;
  // Touch
  c.ontouchstart=e=>{active=true;mv(e.touches[0].clientX)};
  document.ontouchmove=e=>{if(active){mv(e.touches[0].clientX);e.preventDefault()}};
  document.ontouchend=()=>active=false;
  // Başlangıç pozisyonu
  mv(c.getBoundingClientRect().left+c.getBoundingClientRect().width*0.5);
}

function msg(t){
  const el=$('toast');el.textContent=t;el.classList.add('on');
  setTimeout(()=>el.classList.remove('on'),3000);
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ===== PRESETS =====
PRESETS = {
    "logo": {
        "filter_speckle": 2,
        "color_precision": 5,
        "corner_threshold": 90,
        "length_threshold": 3.5,
        "splice_threshold": 45,
        "path_precision": 5,
        "layer_difference": 6,
        "max_iterations": 10,
    },
    "illustration": {
        "filter_speckle": 3,
        "color_precision": 6,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "splice_threshold": 45,
        "path_precision": 4,
        "layer_difference": 10,
        "max_iterations": 10,
    },
    "photo": {
        "filter_speckle": 6,
        "color_precision": 7,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "splice_threshold": 45,
        "path_precision": 3,
        "layer_difference": 16,
        "max_iterations": 10,
    },
    "sketch": {
        "filter_speckle": 3,
        "color_precision": 4,
        "corner_threshold": 80,
        "length_threshold": 3.5,
        "splice_threshold": 45,
        "path_precision": 5,
        "layer_difference": 8,
        "max_iterations": 10,
    },
}

QUALITY_MAP = {
    "1": {"color_precision_add": -2, "filter_speckle_add": 4, "path_precision_add": -2},
    "2": {"color_precision_add": -1, "filter_speckle_add": 2, "path_precision_add": -1},
    "3": {"color_precision_add": 0,  "filter_speckle_add": 0, "path_precision_add": 0},
    "4": {"color_precision_add": 1,  "filter_speckle_add": -1, "path_precision_add": 1},
    "5": {"color_precision_add": 2,  "filter_speckle_add": -2, "path_precision_add": 2},
}


@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    cleanup(UPLOAD)
    cleanup(OUTPUT)

    if "image" not in request.files:
        return jsonify({"success": False, "error": "Dosya yüklenmedi"}), 400
    f = request.files["image"]
    if not allowed(f.filename):
        return jsonify({"success": False, "error": "Geçersiz format"}), 400

    preset = request.form.get("preset", "logo")
    quality = request.form.get("quality", "3")
    colormode = request.form.get("colormode", "color")

    jid = uuid.uuid4().hex[:10]
    ext = f.filename.rsplit(".", 1)[1].lower()
    inp = os.path.join(UPLOAD, f"{jid}.{ext}")
    out = os.path.join(OUTPUT, f"{jid}.svg")
    f.save(inp)

    try:
        # ÖN İŞLEME — kaliteyi dramatik artırır
        preprocess(inp)

        # Preset + kalite ayarları birleştir
        p = PRESETS.get(preset, PRESETS["logo"]).copy()
        q = QUALITY_MAP.get(quality, QUALITY_MAP["3"])
        p["color_precision"] = max(1, min(8, p["color_precision"] + q["color_precision_add"]))
        p["filter_speckle"] = max(0, p["filter_speckle"] + q["filter_speckle_add"])
        p["path_precision"] = max(1, min(8, p["path_precision"] + q["path_precision_add"]))

        # VTracer
        vtracer.convert_image_to_svg_py(
            inp, out,
            colormode=colormode,
            hierarchical="stacked",
            mode="spline",
            filter_speckle=p["filter_speckle"],
            color_precision=p["color_precision"],
            layer_difference=p["layer_difference"],
            corner_threshold=p["corner_threshold"],
            length_threshold=p["length_threshold"],
            max_iterations=p["max_iterations"],
            splice_threshold=p["splice_threshold"],
            path_precision=p["path_precision"],
        )

        sz = os.path.getsize(out)
        szs = f"{sz/(1024*1024):.1f} MB" if sz > 1048576 else f"{sz/1024:.0f} KB"
        with open(out) as svf:
            cnt = svf.read().count("<path")

        return jsonify({"success": True, "job_id": jid, "svg_size": szs, "path_count": cnt})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/preview/<jid>")
def preview(jid):
    p = os.path.join(OUTPUT, f"{jid}.svg")
    return send_file(p, mimetype="image/svg+xml") if os.path.exists(p) else ("", 404)


@app.route("/api/download/<jid>")
def download(jid):
    fmt = request.args.get("format", "svg")
    svg = os.path.join(OUTPUT, f"{jid}.svg")
    if not os.path.exists(svg):
        return "", 404
    if fmt == "svg":
        return send_file(svg, as_attachment=True, download_name=f"vector_{jid}.svg")
    elif fmt == "png":
        try:
            import cairosvg
            png = os.path.join(OUTPUT, f"{jid}.png")
            cairosvg.svg2png(url=svg, write_to=png, output_width=2048)
            return send_file(png, as_attachment=True, download_name=f"vector_{jid}.png")
        except:
            return send_file(svg, as_attachment=True, download_name=f"vector_{jid}.svg")
    elif fmt == "pdf":
        try:
            import cairosvg
            pdf = os.path.join(OUTPUT, f"{jid}.pdf")
            cairosvg.svg2pdf(url=svg, write_to=pdf)
            return send_file(pdf, as_attachment=True, download_name=f"vector_{jid}.pdf")
        except:
            return send_file(svg, as_attachment=True, download_name=f"vector_{jid}.svg")
    return "", 400


# Vectorizer.AI uyumlu API
@app.route("/api/v1/vectorize", methods=["POST"])
def api_v1():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    f = request.files["image"]
    jid = uuid.uuid4().hex[:10]
    ext = f.filename.rsplit(".", 1)[1].lower() if "." in f.filename else "png"
    inp = os.path.join(UPLOAD, f"{jid}.{ext}")
    out = os.path.join(OUTPUT, f"{jid}.svg")
    f.save(inp)
    try:
        preprocess(inp)
        vtracer.convert_image_to_svg_py(inp, out, colormode="color", mode="spline",
            filter_speckle=2, color_precision=6, corner_threshold=60, path_precision=4)
        return send_file(out, mimetype="image/svg+xml")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n⚡ VectorizeIt v2")
    print("🌐 http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
