#pragma once

// Self-contained web dashboard for the ALPR server, embedded as a raw string so
// the binary needs no external assets. Vanilla JS: upload an image, POST it to
// /recognize, draw the returned plate boxes on a canvas, and poll /events.
namespace vision {

inline const char* kDashboardHtml = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ALPR Dashboard</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1115; --card:#1a1d24; --fg:#e6e8ec;
          --muted:#9aa0aa; --accent:#4ade80; --line:#2a2e37; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex;
           align-items:center; gap:12px; }
  header h1 { font-size:18px; margin:0; font-weight:650; }
  header .dot { width:10px; height:10px; border-radius:50%; background:#f87171; }
  header .dot.ok { background:var(--accent); }
  main { max-width:960px; margin:0 auto; padding:24px; display:grid; gap:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:20px; }
  .drop { border:2px dashed var(--line); border-radius:10px; padding:36px;
          text-align:center; color:var(--muted); cursor:pointer; transition:.15s; }
  .drop.hover { border-color:var(--accent); color:var(--fg); }
  canvas { max-width:100%; border-radius:8px; margin-top:16px; display:none; }
  .plates { margin-top:14px; display:flex; flex-wrap:wrap; gap:10px; }
  .plate { font:600 20px ui-monospace,SFMono-Regular,Menlo,monospace;
           letter-spacing:1px; background:#000; color:var(--accent);
           border:1px solid var(--line); padding:8px 14px; border-radius:8px; }
  .plate small { display:block; font:400 11px system-ui; color:var(--muted);
                 letter-spacing:0; margin-top:2px; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  td.mono { font-family:ui-monospace,Menlo,monospace; }
  .muted { color:var(--muted); }
  h2 { font-size:14px; text-transform:uppercase; letter-spacing:.5px;
       color:var(--muted); margin:0 0 12px; }
</style>
</head>
<body>
<header>
  <span class="dot" id="dot"></span>
  <h1>ALPR Dashboard</h1>
  <span class="muted" id="status" style="margin-left:auto"></span>
</header>
<main>
  <div class="card">
    <h2>Recognise a plate</h2>
    <div class="drop" id="drop">Drop an image here, or click to choose a file</div>
    <input type="file" id="file" accept="image/*" hidden>
    <canvas id="canvas"></canvas>
    <div class="plates" id="plates"></div>
  </div>
  <div class="card">
    <h2>Recent events</h2>
    <table>
      <thead><tr><th>Time</th><th>Plate</th><th>OCR</th><th>Detect</th></tr></thead>
      <tbody id="events"><tr><td colspan="4" class="muted">No events yet.</td></tr></tbody>
    </table>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
const drop = $('#drop'), file = $('#file'), canvas = $('#canvas'), ctx = canvas.getContext('2d');

async function health() {
  try {
    const r = await fetch('/health'); const j = await r.json();
    $('#dot').classList.add('ok');
    $('#status').textContent = `model ${j.detector.split('/').pop()} · OCR ${j.ocr}`;
  } catch { $('#dot').classList.remove('ok'); $('#status').textContent = 'offline'; }
}

function drawResult(img, plates) {
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  canvas.style.display = 'block';
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = Math.max(2, img.naturalWidth / 300);
  ctx.strokeStyle = '#4ade80'; ctx.fillStyle = '#4ade80';
  ctx.font = `${Math.max(16, img.naturalWidth/40)}px ui-monospace, monospace`;
  for (const p of plates) {
    const [x,y,w,h] = p.box;
    ctx.strokeRect(x, y, w, h);
    if (p.text) ctx.fillText(p.text, x, Math.max(y-6, 14));
  }
  const box = $('#plates'); box.innerHTML = '';
  if (!plates.length) box.innerHTML = '<span class="muted">No plates detected.</span>';
  for (const p of plates) {
    const el = document.createElement('div'); el.className = 'plate';
    el.innerHTML = (p.text || '— unreadable —') +
      `<small>OCR ${(p.ocr_confidence*100).toFixed(0)}% · det ${(p.det_confidence*100).toFixed(0)}%</small>`;
    box.appendChild(el);
  }
}

async function recognise(fileObj) {
  $('#status').textContent = 'recognising…';
  const buf = await fileObj.arrayBuffer();
  const r = await fetch('/recognize', { method:'POST',
    headers:{'Content-Type': fileObj.type || 'application/octet-stream'}, body: buf });
  const j = await r.json();
  const img = new Image();
  img.onload = () => { drawResult(img, j.plates || []); loadEvents(); };
  img.src = URL.createObjectURL(fileObj);
  $('#status').textContent = `${j.count} plate(s) in ${j.ms} ms`;
}

async function loadEvents() {
  try {
    const r = await fetch('/events?limit=15'); const j = await r.json();
    const tb = $('#events');
    if (!j.events || !j.events.length) { tb.innerHTML = '<tr><td colspan="4" class="muted">No events yet.</td></tr>'; return; }
    tb.innerHTML = j.events.slice().reverse().map(e =>
      `<tr><td class="muted">${e.time}</td><td class="mono">${e.text}</td>`+
      `<td>${(e.ocr_confidence*100).toFixed(0)}%</td>`+
      `<td>${(e.det_confidence*100).toFixed(0)}%</td></tr>`).join('');
  } catch {}
}

drop.onclick = () => file.click();
file.onchange = () => file.files[0] && recognise(file.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hover'); };
drop.ondragleave = () => drop.classList.remove('hover');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('hover');
  e.dataTransfer.files[0] && recognise(e.dataTransfer.files[0]); };

health(); loadEvents(); setInterval(loadEvents, 5000); setInterval(health, 10000);
</script>
</body>
</html>)HTML";

}  // namespace vision
