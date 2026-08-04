"""A small dashboard, served as a single self-contained page.

No CDN, no build step, no charting library. The page renders its own SVG from
the JSON the API already returns. That is a deliberate constraint: a pod inside
a cluster may have no egress, and a dashboard that silently breaks without
internet access is worse than no dashboard. It also keeps the image free of a
node toolchain.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAN Capacity Forecast</title>
<style>
  :root {
    --ink: #16202b; --muted: #6b7885; --line: #dde3e9; --bg: #f6f8fa;
    --actual: #16202b; --naive: #c0392b; --model: #2471a3;
    --alert: #e67e22; --risk: #b03a2e;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { background: #fff; border-bottom: 1px solid var(--line); padding: 18px 24px; }
  h1 { margin: 0; font-size: 18px; letter-spacing: -0.01em; }
  header p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
  main { padding: 20px 24px; max-width: 1180px; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 8px;
          padding: 16px 18px; margin-bottom: 18px; }
  .card h2 { margin: 0 0 12px; font-size: 14px; text-transform: uppercase;
             letter-spacing: 0.06em; color: var(--muted); }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  select, button { font: inherit; padding: 6px 10px; border: 1px solid var(--line);
                   border-radius: 6px; background: #fff; color: var(--ink); }
  button { cursor: pointer; }
  button:hover { background: var(--bg); }
  .stats { display: flex; gap: 26px; flex-wrap: wrap; margin-top: 14px; }
  .stat b { display: block; font-size: 20px; font-weight: 600; }
  .stat span { color: var(--muted); font-size: 12px; }
  .legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted);
            margin-top: 8px; flex-wrap: wrap; }
  .swatch { display: inline-block; width: 14px; height: 3px; vertical-align: middle;
            margin-right: 5px; border-radius: 2px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         vertical-align: middle; margin-right: 5px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.05em; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 10px;
          font-size: 11px; font-weight: 600; }
  .high { background: #fdedeb; color: #b03a2e; }
  .medium { background: #fef5e7; color: #b9770e; }
  .low { background: #eaf2f8; color: #21618c; }
  .muted { color: var(--muted); }
  svg { width: 100%; height: auto; display: block; }
</style>
</head>
<body>
<header>
  <h1>RAN Capacity Forecast</h1>
  <p>Cell-level PRB utilisation &mdash; day-ahead forecast, baseline comparison and anomaly episodes.
     Synthetic data; see README.</p>
</header>
<main>
  <div class="card">
    <h2>Forecast</h2>
    <div class="controls">
      <label for="cell">Cell</label>
      <select id="cell"></select>
      <label for="hist">History</label>
      <select id="hist">
        <option value="72">3 days</option>
        <option value="168" selected>7 days</option>
        <option value="336">14 days</option>
      </select>
      <button id="refresh">Refresh</button>
    </div>
    <div class="stats" id="stats"></div>
    <div id="chart"></div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--actual)"></i>Actual</span>
      <span><i class="swatch" style="background:var(--naive)"></i>Seasonal naive</span>
      <span><i class="swatch" style="background:var(--model)"></i>LightGBM</span>
      <span><i class="dot" style="background:var(--alert)"></i>Forecast window</span>
      <span><i class="swatch" style="background:var(--risk)"></i>Capacity threshold (85%)</span>
    </div>
  </div>

  <div class="card">
    <h2>Capacity risk &mdash; next 24 hours</h2>
    <table id="risk"><tbody><tr><td class="muted">Loading&hellip;</td></tr></tbody></table>
  </div>

  <div class="card">
    <h2>Anomaly episodes</h2>
    <table id="anomalies"><tbody><tr><td class="muted">Loading&hellip;</td></tr></tbody></table>
  </div>
</main>

<script>
const W = 1100, H = 320, PAD = { t: 14, r: 14, b: 26, l: 42 };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + ' -> ' + res.status);
  return res.json();
}

function path(points, xs, ys) {
  let d = '', pen = false;
  points.forEach((p, i) => {
    if (p == null || Number.isNaN(p)) { pen = false; return; }
    d += (pen ? 'L' : 'M') + xs(i).toFixed(1) + ' ' + ys(p).toFixed(1) + ' ';
    pen = true;
  });
  return d.trim();
}

function drawChart(body) {
  const pts = body.points;
  if (!pts.length) { $('chart').innerHTML = '<p class="muted">No data.</p>'; return; }

  const innerW = W - PAD.l - PAD.r, innerH = H - PAD.t - PAD.b;
  const xs = (i) => PAD.l + (pts.length === 1 ? innerW / 2 : (i / (pts.length - 1)) * innerW);
  const ys = (v) => PAD.t + innerH - (Math.max(0, Math.min(100, v)) / 100) * innerH;

  const firstFuture = pts.findIndex(p => p.is_forecast);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Forecast chart">`;

  for (let v = 0; v <= 100; v += 25) {
    svg += `<line x1="${PAD.l}" y1="${ys(v)}" x2="${W - PAD.r}" y2="${ys(v)}"
            stroke="var(--line)" stroke-width="1"/>`;
    svg += `<text x="${PAD.l - 8}" y="${ys(v) + 4}" text-anchor="end"
            font-size="11" fill="var(--muted)">${v}</text>`;
  }
  svg += `<line x1="${PAD.l}" y1="${ys(85)}" x2="${W - PAD.r}" y2="${ys(85)}"
          stroke="var(--risk)" stroke-width="1" stroke-dasharray="5 4" opacity="0.75"/>`;

  if (firstFuture > 0) {
    svg += `<rect x="${xs(firstFuture)}" y="${PAD.t}"
            width="${W - PAD.r - xs(firstFuture)}" height="${innerH}"
            fill="var(--alert)" opacity="0.07"/>`;
    svg += `<line x1="${xs(firstFuture)}" y1="${PAD.t}" x2="${xs(firstFuture)}"
            y2="${PAD.t + innerH}" stroke="var(--alert)" stroke-width="1.5"/>`;
  }

  const series = [
    ['yhat_baseline', 'var(--naive)', 1.2, '4 3'],
    ['actual',        'var(--actual)', 1.9, null],
    ['yhat',          'var(--model)',  1.6, null],
  ];
  for (const [key, colour, width, dash] of series) {
    const d = path(pts.map(p => p[key]), xs, ys);
    if (!d) continue;
    svg += `<path d="${d}" fill="none" stroke="${colour}" stroke-width="${width}"
            ${dash ? `stroke-dasharray="${dash}"` : ''} stroke-linejoin="round"/>`;
  }

  const step = Math.max(1, Math.floor(pts.length / 8));
  pts.forEach((p, i) => {
    if (i % step) return;
    const label = new Date(p.timestamp).toISOString().slice(5, 13).replace('T', ' ');
    svg += `<text x="${xs(i)}" y="${H - 8}" text-anchor="middle"
            font-size="10" fill="var(--muted)">${label}</text>`;
  });

  $('chart').innerHTML = svg + '</svg>';
}

function drawStats(body) {
  const cells = [
    ['Peak forecast', body.peak_forecast.toFixed(1) + '%'],
    ['Hours at risk', body.hours_at_risk],
    ['Archetype', body.archetype ?? '—'],
    ['Site', body.site_id ?? '—'],
  ];
  $('stats').innerHTML = cells
    .map(([k, v]) => `<div class="stat"><b>${esc(v)}</b><span>${k}</span></div>`)
    .join('');
}

async function loadForecast() {
  const cell = $('cell').value, hist = $('hist').value;
  if (!cell) return;
  const body = await getJSON(`/forecast/${cell}?horizon=24&include_history=${hist}`);
  drawStats(body);
  drawChart(body);
}

async function loadRisk() {
  const rows = await getJSON('/capacity-risk?limit=10');
  const head = `<thead><tr><th>Cell</th><th>Archetype</th>
    <th class="num">Peak</th><th class="num">Hours at risk</th><th>Recommendation</th></tr></thead>`;
  const body = rows.length
    ? rows.map(r => `<tr><td>${esc(r.cell_id)}</td><td>${esc(r.archetype ?? '—')}</td>
        <td class="num">${r.peak_forecast.toFixed(1)}%</td>
        <td class="num">${r.hours_at_risk}</td>
        <td class="muted">${esc(r.recommendation)}</td></tr>`).join('')
    : '<tr><td class="muted">No cells forecast to breach the threshold.</td></tr>';
  $('risk').innerHTML = head + '<tbody>' + body + '</tbody>';
}

async function loadAnomalies() {
  const data = await getJSON('/anomalies?limit=12');
  const head = `<thead><tr><th>Cell</th><th>Window</th><th class="num">Hours</th>
    <th class="num">Peak z</th><th>Direction</th><th>Severity</th></tr></thead>`;
  const body = data.episodes.length
    ? data.episodes.map(e => `<tr><td>${esc(e.cell_id)}</td>
        <td class="muted">${e.start.slice(5, 16).replace('T', ' ')} → ${e.end.slice(11, 16)}</td>
        <td class="num">${e.hours}</td>
        <td class="num">${e.peak_z.toFixed(1)}</td>
        <td class="muted">${esc(e.direction.replace('_', ' '))}</td>
        <td><span class="pill ${esc(e.severity)}">${esc(e.severity)}</span></td></tr>`).join('')
    : '<tr><td class="muted">No anomalies detected.</td></tr>';
  $('anomalies').innerHTML = head + '<tbody>' + body + '</tbody>';
}

async function init() {
  try {
    const data = await getJSON('/cells');
    $('cell').innerHTML = data.cells
      .map(c => `<option value="${esc(c.cell_id)}">${esc(c.cell_id)} (${esc(c.archetype)})</option>`)
      .join('');
    await Promise.all([loadForecast(), loadRisk(), loadAnomalies()]);
  } catch (err) {
    document.querySelector('main').innerHTML =
      `<div class="card"><h2>Not ready</h2><p class="muted">${esc(err.message)}
       — run <code>make train</code> to build the artifacts.</p></div>`;
  }
}

$('cell').addEventListener('change', loadForecast);
$('hist').addEventListener('change', loadForecast);
$('refresh').addEventListener('click', () => Promise.all([loadForecast(), loadRisk(), loadAnomalies()]));
init();
</script>
</body>
</html>
"""
