 // Theme
const themeToggle = document.getElementById('themeToggle');
const themeLabel = document.getElementById('themeLabel');
function setTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  themeLabel.textContent = t[0].toUpperCase() + t.slice(1);
  localStorage.setItem('theme', t);
  themeToggle.checked = (t === 'dark');
}
setTheme(localStorage.getItem('theme') || 'light');
themeToggle.addEventListener('change', () => setTheme(themeToggle.checked ? 'dark' : 'light'));

// Canvas (main)
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
canvas.width = 280; canvas.height = 280;
function resetCanvas() {
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#000000'; ctx.lineWidth = 20; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
}
resetCanvas();

let drawing=false, lastX=0, lastY=0;
function getPos(e){
  const rect = canvas.getBoundingClientRect();
  const isTouch = e.touches && e.touches.length;
  const clientX = isTouch ? e.touches[0].clientX : e.clientX;
  const clientY = isTouch ? e.touches[0].clientY : e.clientY;
  return { x: clientX - rect.left, y: clientY - rect.top };
}
function startDraw(e){ e.preventDefault(); drawing=true; const p=getPos(e); lastX=p.x; lastY=p.y; }
function moveDraw(e){ if(!drawing) return; e.preventDefault(); const p=getPos(e); ctx.beginPath(); ctx.moveTo(lastX,lastY); ctx.lineTo(p.x,p.y); ctx.stroke(); lastX=p.x; lastY=p.y; }
function endDraw(e){ if(!drawing) return; e.preventDefault(); drawing=false; ctx.beginPath(); }

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', moveDraw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);
canvas.addEventListener('touchstart', startDraw, {passive:false});
canvas.addEventListener('touchmove', moveDraw, {passive:false});
canvas.addEventListener('touchend', endDraw);

// Buttons (main)
document.getElementById('clearBtn').addEventListener('click', () => { resetCanvas(); document.getElementById('drawResult').innerHTML=''; });
document.getElementById('downloadBtn').addEventListener('click', () => {
  const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download='digit.png'; a.click();
});

// Predict canvas (non-calculator)
document.getElementById('predictBtn').addEventListener('click', async () => {
  const btn = document.getElementById('predictBtn');
  btn.disabled=true; btn.textContent='Predicting...';
  try {
    const threshold = parseFloat(document.getElementById('threshold').value || '0.7');
    const smoothing = document.getElementById('smoothing') ? document.getElementById('smoothing').checked : true;
    const dataURL = canvas.toDataURL('image/png');
    const res = await fetch('/predict_draw', {
      method:'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ image: dataURL, threshold, smoothing })
    });
    const data = await res.json();
    const out = document.getElementById('drawResult');
    if (data.error) {
      out.innerHTML = `<div class="error">Error: ${data.error}</div>`;
    } else {
      const pct = (data.confidence*100).toFixed(1);
      const shown = data.is_sure ? data.raw_digit : 'Not Sure';
      out.innerHTML = `<div class="pred"><div class="big">${shown}</div><div>Confidence: ${pct}%</div><details><summary>Class probs</summary><pre>${JSON.stringify(data.probs,null,2)}</pre></details></div>`;
    }
  } catch (e) {
    document.getElementById('drawResult').innerHTML = `<div class="error">Error: ${e}</div>`;
  } finally {
    btn.disabled=false; btn.textContent='Predict';
  }
});

// Save to gallery (localStorage)
const GALLERY_KEY = 'digit_gallery_v1';
function loadGallery(){ try { return JSON.parse(localStorage.getItem(GALLERY_KEY) || '[]'); } catch(e){ return []; } }
function saveGallery(arr){ localStorage.setItem(GALLERY_KEY, JSON.stringify(arr)); }

function renderGallery(){
  const gallery = document.getElementById('gallery');
  gallery.innerHTML = '';
  const items = loadGallery();
  if (!items.length) {
    gallery.innerHTML = `<div class="predline" style="color:var(--muted)">No saved drawings yet.</div>`;
    return;
  }

  items.forEach((dataURL, idx) => {
    const div = document.createElement('div');
    div.className = 'grid-item';
    div.dataset.idx = idx;

    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    const img = document.createElement('img');
    img.id = `gimg-${idx}`;
    img.src = dataURL;
    img.alt = `Saved ${idx+1}`;
    img.style.cursor = 'pointer';
    img.addEventListener('click', () => openModalByIndex(idx));

    const deleteOverlay = document.createElement('button');
    deleteOverlay.className = 'delete-overlay';
    deleteOverlay.title = 'Delete';
    deleteOverlay.innerText = '✕';
    deleteOverlay.dataset.idx = idx;
    deleteOverlay.addEventListener('click', (ev) => {
      ev.stopPropagation();
      deleteGalleryItem(parseInt(ev.currentTarget.dataset.idx, 10));
    });

    thumb.appendChild(img);
    thumb.appendChild(deleteOverlay);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = `<div class="fn">Saved ${idx+1}</div>
                      <div class="predline"><button data-idx="${idx}" class="openBtn">Open</button></div>`;

    div.appendChild(thumb);
    div.appendChild(meta);
    gallery.appendChild(div);
  });
}

document.getElementById('saveGalleryBtn').addEventListener('click', () => {
  const dataURL = canvas.toDataURL('image/png');
  const items = loadGallery();
  items.unshift(dataURL);
  if (items.length > 50) items.pop();
  saveGallery(items);
  renderGallery();
});
document.getElementById('clearGallery').addEventListener('click', () => { localStorage.removeItem(GALLERY_KEY); renderGallery(); });

// Modal preview
const modal = document.getElementById('modal');
const modalImg = document.getElementById('modalImg');
const modalClose = document.getElementById('modalClose');
const modalDownload = document.getElementById('modalDownload');
const modalDelete = document.getElementById('modalDelete');

let modalCurrentIndex = null;

function openModalByIndex(idx){
  const items = loadGallery();
  if (idx < 0 || idx >= items.length) return;
  modalCurrentIndex = idx;
  modalImg.src = items[idx];
  modal.classList.remove('hidden');
  modalDelete.style.display = '';
  modalDownload.onclick = () => { const a=document.createElement('a'); a.href = modalImg.src; a.download=`saved-${idx+1}.png`; a.click(); };
}

function openModalWithSrc(src, filename=null){
  const items = loadGallery();
  const idx = items.indexOf(src);
  if (idx >= 0) {
    openModalByIndex(idx);
    return;
  }
  modalCurrentIndex = null;
  modalImg.src = src;
  modal.classList.remove('hidden');
  modalDelete.style.display = 'none';
  modalDownload.onclick = () => { const a=document.createElement('a'); a.href = src; a.download = filename || 'image.png'; a.click(); };
}

modalClose.addEventListener('click', ()=> { modal.classList.add('hidden'); modalCurrentIndex = null; });
modal.addEventListener('click', (e) => { if (e.target === modal) { modal.classList.add('hidden'); modalCurrentIndex = null; } });

modalDelete.addEventListener('click', () => {
  if (modalCurrentIndex === null) return;
  deleteGalleryItem(modalCurrentIndex);
  modal.classList.add('hidden');
  modalCurrentIndex = null;
});

function deleteGalleryItem(index) {
  const items = loadGallery();
  if (index < 0 || index >= items.length) return;
  items.splice(index, 1);
  saveGallery(items);
  renderGallery();
}

document.body.addEventListener('click', (e) => {
  if (e.target.classList.contains('openBtn')) {
    const idx = parseInt(e.target.dataset.idx, 10);
    openModalByIndex(idx);
  }
});

const __tempObjectURLs = [];

// Upload & predict files
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const grid = document.getElementById('grid'); grid.innerHTML='';
  const files = Array.from(document.getElementById('files').files);
  if (!files.length) return;
  const threshold = parseFloat(document.getElementById('thresholdUpload').value || '0.7');
  const smoothing = document.getElementById('smoothingUpload') ? document.getElementById('smoothingUpload').checked : true;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  fd.append('threshold', threshold);
  fd.append('smoothing', smoothing ? 'true' : 'false');

  const res = await fetch('/predict_files', { method:'POST', body: fd });
  const data = await res.json();
  if (data.error) { grid.innerHTML = `<div class="error">Error: ${data.error}</div>`; return; }

  data.results.forEach((item, idx) => {
    const file = files[idx];
    const pct = (item.confidence*100).toFixed(1);
    const label = item.is_sure ? item.raw_digit : 'Not Sure';

    const card = document.createElement('div');
    card.className = 'grid-item';

    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    const img = document.createElement('img');
    img.id = `upimg-${idx}`;
    img.alt = item.filename || `Uploaded ${idx+1}`;
    thumb.appendChild(img);

    const deleteOverlay = document.createElement('button');
    deleteOverlay.className = 'delete-overlay';
    deleteOverlay.title = 'Delete';
    deleteOverlay.innerText = '✕';
    deleteOverlay.addEventListener('click', (ev) => {
      ev.stopPropagation();
      card.remove();
      try { URL.revokeObjectURL(url); } catch(_) {}
    });
    thumb.appendChild(deleteOverlay);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = `<div class="fn" title="${item.filename}">${item.filename}</div>
                      <div class="predline"><strong>${label}</strong> — ${pct}%</div>
                      <details><summary>Probs</summary><pre>${JSON.stringify(item.probs,null,2)}</pre></details>`;

    card.appendChild(thumb);
    card.appendChild(meta);
    grid.appendChild(card);

    const url = URL.createObjectURL(file);
    __tempObjectURLs.push(url);
    img.src = url;
    img.style.cursor = 'pointer';
    img.addEventListener('click', () => openModalWithSrc(url, file.name));
  });
});

window.addEventListener('beforeunload', () => {
  __tempObjectURLs.forEach(u => {
    try { URL.revokeObjectURL(u); } catch(_) {}
  });
});

renderGallery();

// ----------------- Calculator -----------------
const calcCanvas = document.getElementById('calcCanvas');
const calcCtx = calcCanvas.getContext('2d');
calcCanvas.width = 280; calcCanvas.height = 280;
function resetCalcCanvas() {
  calcCtx.fillStyle = '#ffffff';
  calcCtx.fillRect(0,0,calcCanvas.width,calcCanvas.height);
  calcCtx.strokeStyle = '#000000';
  calcCtx.lineWidth = 20;
  calcCtx.lineCap = 'round';
  calcCtx.lineJoin = 'round';
}
resetCalcCanvas();

let calcDrawing = false, calcLastX=0, calcLastY=0;
function calcGetPos(e){ const r = calcCanvas.getBoundingClientRect(); const x=(e.touches?e.touches[0].clientX:e.clientX)-r.left; const y=(e.touches?e.touches[0].clientY:e.clientY)-r.top; return {x,y}; }
function calcStart(e){ e.preventDefault(); calcDrawing=true; const p=calcGetPos(e); calcLastX=p.x; calcLastY=p.y; }
function calcMove(e){ if(!calcDrawing) return; e.preventDefault(); const p=calcGetPos(e); calcCtx.beginPath(); calcCtx.moveTo(calcLastX,calcLastY); calcCtx.lineTo(p.x,p.y); calcCtx.stroke(); calcLastX=p.x; calcLastY=p.y; }
function calcEnd(e){ if(!calcDrawing) return; e.preventDefault(); calcDrawing=false; calcCtx.beginPath(); }

calcCanvas.addEventListener('mousedown', calcStart);
calcCanvas.addEventListener('mousemove', calcMove);
calcCanvas.addEventListener('mouseup', calcEnd);
calcCanvas.addEventListener('mouseleave', calcEnd);
calcCanvas.addEventListener('touchstart', calcStart, {passive:false});
calcCanvas.addEventListener('touchmove', calcMove, {passive:false});
calcCanvas.addEventListener('touchend', calcEnd);

document.getElementById('calcClearBtn').addEventListener('click', () => { resetCalcCanvas(); });

async function refreshExpressionDisplay(){
  const res = await fetch('/get_expression');
  const data = await res.json();
  if (data.expression !== undefined) {
    document.getElementById('calcExpr').value = data.expression;
  }
}
refreshExpressionDisplay();

document.getElementById('calcAddBtn').addEventListener('click', async () => {
  const dataURL = calcCanvas.toDataURL('image/png');
  const threshold = parseFloat(document.getElementById('threshold').value || '0.7');
  const smoothing = document.getElementById('smoothing') ? document.getElementById('smoothing').checked : true;
  
  const res = await fetch('/predict_and_append', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ image: dataURL, threshold, smoothing })
  });
  const data = await res.json();
  
  if (data.error) {
    alert("Prediction error: " + data.error);
    return;
  }

  if (data.is_sure) {
    document.getElementById('calcExpr').value = data.expression || "";
  } else {
    alert("Uncertain digit prediction. Please redraw.");
  }
  resetCalcCanvas();
});

document.querySelectorAll('.op-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const op = btn.dataset.op;
    const res = await fetch('/append_symbol', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ symbol: op })
    });
    const data = await res.json();
    if (data.error) {
      alert("Could not append operator: " + data.error);
      return;
    }
    document.getElementById('calcExpr').value = data.expression || "";
  });
});

document.getElementById('calcBackspace').addEventListener('click', async () => {
  const g = await fetch('/get_expression'); const dg = await g.json();
  if (dg.error) { alert(dg.error); return; }
  let expr = dg.expression || "";
  if (!expr) return;
  expr = expr.slice(0, -1);
  await fetch('/clear_expression', { method:'POST' });
  if (expr.length) {
    await fetch('/append_symbol', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ symbol: expr })
    });
  }
  await refreshExpressionDisplay();
});

document.getElementById('calcClearExpr').addEventListener('click', async () => {
  const res = await fetch('/clear_expression', { method:'POST' });
  const data = await res.json();
  if (data.error) { alert("Error clearing expression: " + data.error); return; }
  document.getElementById('calcExpr').value = data.expression;
});

document.getElementById('calcEvalBtn').addEventListener('click', async () => {
  const res = await fetch('/evaluate_expression', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({})
  });
  const data = await res.json();
  const out = document.getElementById('calcResult');
  if (data.error) {
    out.innerHTML = `<div class="error">${data.error}</div>`;
  } else {
    out.innerHTML = `<div class="pred"><div class="big">${data.result}</div><div>from ${data.expression}</div></div>`;
  }
  await refreshExpressionDisplay();
});
