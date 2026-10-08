/* plotspec.js — render a library-agnostic plot spec (see runner._build_plot_spec)
 * as a stack of zoomable, cursor-synced uPlot panels.
 *
 *   PlotSpec.render(containerEl, spec)  ->  { destroy() }
 *
 * Spec shape (per panel):
 *   { name, ylabel, ylabel_right?, ymin_span?, type: "lines"|"gantt"|"xy",
 *     hlines: [{y, color, style, label?}], shading: [[t0,t1], ...],
 *     traces: [{label, color, style, axis: "left"|"right", y: [...]}],
 *     lanes:  [...]   // type == "gantt" only
 *   }
 * type == "xy" (horizontal vs vertical, drawn after the time panels):
 *   { xlabel, ylabel_axis, min_span?,
 *     traces: [{label, color, x: [...], y: [...], x_offset?, role?: "target"}] }
 *   x/y are sampled on spec.t; setTime() moves a current-frame dot + fading trail.
 *
 * Zoom: drag-select on any panel zooms the shared time axis on every panel;
 * double-click resets. Hover shows a synced crosshair + per-series values.
 *
 * Requires uPlot (window.uPlot) loaded first.
 */
(function () {
  'use strict';

  const SYNC_KEY = 'plotspec-x';

  // matplotlib linestyle -> uPlot dash array (CSS px)
  function dash(style) {
    switch (style) {
      case '--': return [6, 4];
      case '-.': return [8, 4, 2, 4];
      case ':':  return [2, 4];
      default:   return [];          // solid
    }
  }

  function injectStyles() {
    if (document.getElementById('plotspec-styles')) return;
    const css = `
      .ps-panel { margin-bottom: 6px; }
      .ps-panel .u-legend { font-size: 10px; padding: 2px 0 4px; }
      .ps-panel .u-legend .u-marker { width: 10px; height: 10px; }
      .ps-ylabel { font-size: 10px; color: #64748b; letter-spacing: .03em;
                   margin: 2px 0 0 6px; text-transform: none; }
      .ps-hint { font-size: 10px; color: #94a3b8; margin: 0 0 8px 6px; }
      .ps-playhead { position: absolute; top: 0; bottom: 0; width: 0;
                     border-left: 1.5px solid #ef4444; pointer-events: none;
                     display: none; z-index: 5; }
      /* title + legend share one row to keep the plot area tall */
      .ps-head { display: flex; align-items: baseline; justify-content: space-between;
                 gap: 12px; margin: 2px 6px 0; flex-wrap: wrap; }
      .ps-head .ps-ylabel { margin: 0; }
      .ps-legend { display: flex; flex-wrap: wrap; gap: 2px 12px; align-items: center; }
      .ps-leg-item { display: inline-flex; align-items: center; gap: 5px;
                     font-size: 10px; color: #475569; white-space: nowrap; }
      .ps-swatch { width: 14px; height: 2px; border-radius: 1px;
                   display: inline-block; flex-shrink: 0; }
      /* X–Y panel: two stacked canvases (static trace below, per-frame marker above) */
      .ps-xy { position: relative; height: 300px; }
      .ps-xy canvas { position: absolute; left: 0; top: 0; }
      /* gantt */
      .ps-gantt-lane { display: flex; align-items: center; height: 24px; margin: 2px 0; }
      .ps-gantt-label { box-sizing: border-box; width: 80px; flex-shrink: 0; font-size: 10px;
                        color: #64748b; text-align: right; padding-right: 8px; }
      /* x-axis label under the stack (shared time axis). Left margin = gutter so
         it centres over the plot area, which starts after the 80px y-gutter. */
      .ps-xlabel { text-align: center; font-size: 10px; color: #64748b;
                   margin: 1px 0 4px 80px; }
      .ps-gantt-track { position: relative; flex: 1; height: 20px;
                        background: #eef0f4; border-radius: 3px; overflow: hidden; }
      .ps-gantt-seg { position: absolute; top: 0; height: 100%; display: flex;
                      align-items: center; justify-content: center; font-size: 9px;
                      color: #1c2230; white-space: nowrap; overflow: hidden; }
      .ps-gantt-seg.off { background-image: repeating-linear-gradient(
                            45deg, #dfe3ea 0, #dfe3ea 4px, #eef0f4 4px, #eef0f4 8px);
                          color: #94a3b8; }
    `;
    const el = document.createElement('style');
    el.id = 'plotspec-styles';
    el.textContent = css;
    document.head.appendChild(el);
  }

  // Plugin: dark-period shading + horizontal reference lines, drawn behind series.
  function annotationsPlugin(panel) {
    return {
      hooks: {
        drawClear: (u) => {
          const ctx = u.ctx;
          // shaded dark spans (x in data units)
          (panel.shading || []).forEach(([t0, t1]) => {
            const x0 = u.valToPos(t0, 'x', true);
            const x1 = u.valToPos(t1, 'x', true);
            ctx.save();
            ctx.fillStyle = 'rgba(120,120,140,0.10)';
            ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
            ctx.restore();
          });
        },
        draw: (u) => {
          const ctx = u.ctx;
          (panel.hlines || []).forEach((h) => {
            if (u.scales.y.min == null) return;
            const y = u.valToPos(h.y, 'y', true);
            ctx.save();
            ctx.strokeStyle = h.color || '#888';
            ctx.lineWidth = Math.max(1, Math.round(devicePixelRatio));
            ctx.setLineDash(dash(h.style).map((d) => d * devicePixelRatio));
            ctx.beginPath();
            ctx.moveTo(u.bbox.left, y);
            ctx.lineTo(u.bbox.left + u.bbox.width, y);
            ctx.stroke();
            ctx.restore();
          });
        },
      },
    };
  }

  // Header: y-axis label (left) + compact legend (right) on one row, so the
  // canvas keeps its full height instead of losing it to a legend below.
  function makeHead(panel) {
    const head = document.createElement('div');
    head.className = 'ps-head';
    const label = document.createElement('span');
    label.className = 'ps-ylabel';
    label.textContent = panel.ylabel;
    head.appendChild(label);

    const legend = document.createElement('span');
    legend.className = 'ps-legend';
    (panel.traces || []).forEach((tr) => {
      const item = document.createElement('span');
      item.className = 'ps-leg-item';
      const sw = document.createElement('span');
      sw.className = 'ps-swatch';
      const d = dash(tr.style);
      if (d.length) {
        // Dashed/dotted swatch matching the line style (horizontal dashes).
        const on = d[0], off = d[1] || d[0];
        sw.style.background =
          `repeating-linear-gradient(90deg, ${tr.color} 0 ${on}px, transparent ${on}px ${on + off}px)`;
      } else {
        sw.style.background = tr.color;
      }
      item.appendChild(sw);
      item.appendChild(document.createTextNode(
        tr.label + (tr.axis === 'right' ? ' (R)' : '')));
      legend.appendChild(item);
    });
    head.appendChild(legend);
    return head;
  }

  function makeLinePanel(panel, t, sharedX, registerChart) {
    const wrap = document.createElement('div');
    wrap.className = 'ps-panel';

    const hasRight = (panel.traces || []).some((tr) => tr.axis === 'right');

    const series = [{ label: 'time (s)' }];
    const data = [t];
    (panel.traces || []).forEach((tr) => {
      series.push({
        label: tr.label,
        stroke: tr.color,
        width: 1.3,
        dash: dash(tr.style),
        scale: tr.axis === 'right' ? 'y2' : 'y',
        spanGaps: false,
        points: { show: false },
      });
      // `offset` is the binocular zero-reference, applied here at render time so
      // the spec's y data stays raw/veridical (and a raw↔calibrated toggle is easy).
      data.push(tr.offset
        ? tr.y.map((v) => (v == null ? null : v + tr.offset))
        : tr.y);
    });

    const yRange = (u, dMin, dMax) => {
      let [min, max] = uPlot.rangeNum(dMin, dMax, 0.1, true);
      if (panel.ymin_span && (max - min) < panel.ymin_span) {
        const mid = (min + max) / 2;
        min = mid - panel.ymin_span / 2;
        max = mid + panel.ymin_span / 2;
      }
      return [min, max];
    };

    const axes = [
      { stroke: '#475569', grid: { stroke: '#e5e7eb', width: 1 },
        ticks: { stroke: '#d3d7e0' }, font: '10px sans-serif' },
      { scale: 'y', stroke: '#475569', grid: { stroke: '#e5e7eb', width: 1 },
        ticks: { stroke: '#d3d7e0' }, font: '10px sans-serif', size: 80 },   // gutter matches gantt label width (alignment)
    ];
    const scales = { x: { time: false }, y: { range: yRange } };
    if (hasRight) {
      scales.y2 = { range: (u, a, b) => uPlot.rangeNum(a, b, 0.1, true) };
      axes.push({ scale: 'y2', side: 1, stroke: '#64748b', grid: { show: false },
                  ticks: { stroke: '#d3d7e0' }, font: '10px sans-serif', size: 46 });
    }

    let syncingLocal = false;
    const opts = {
      width: wrap.clientWidth || 600,
      height: 180,
      pxAlign: false,                  // don't snap thin diagonals to the pixel grid → smoother lines
      scales,
      series,
      axes,
      legend: { show: false },         // custom inline legend shares the title row
      cursor: {
        sync: { key: SYNC_KEY },
        drag: { x: true, y: false },
        // NB: do NOT set points.show = true here — uPlot wraps a boolean via
        // fnOrSelf and then calls addClass(true, …), throwing on .classList.
      },
      plugins: [annotationsPlugin(panel)],
      hooks: {
        setScale: [
          (u, key) => {
            if (key !== 'x' || syncingLocal) return;
            syncingLocal = true;
            registerChart.syncX(u, u.scales.x.min, u.scales.x.max);
            syncingLocal = false;
            registerChart.reposition();          // keep playhead glued on zoom
          },
        ],
      },
    };

    wrap.appendChild(makeHead(panel));

    const u = new uPlot(opts, data, wrap);
    registerChart.add(u);

    // External time playhead (synced to 3D playback): a positioned line over the
    // plot area, moved via the render handle's setTime() — no canvas redraw.
    const ph = document.createElement('div');
    ph.className = 'ps-playhead';
    u.over.appendChild(ph);
    registerChart.addPlayhead(u, ph);

    return wrap;
  }

  function makeGanttPanel(panel, tmin, tmax, registry) {
    const wrap = document.createElement('div');
    wrap.className = 'ps-panel';
    const label = document.createElement('div');
    label.className = 'ps-ylabel';
    label.textContent = panel.ylabel || 'Visual context';
    wrap.appendChild(label);

    const span = (tmax - tmin) || 1;
    (panel.lanes || []).forEach((lane) => {
      const row = document.createElement('div');
      row.className = 'ps-gantt-lane';
      const lab = document.createElement('div');
      lab.className = 'ps-gantt-label';
      lab.textContent = lane.label;
      const track = document.createElement('div');
      track.className = 'ps-gantt-track';
      registry.addGanttTrack(track, lab);   // align its width to the plot area after layout
      (lane.segments || []).forEach(([s0, s1, state]) => {
        const seg = document.createElement('div');
        seg.className = 'ps-gantt-seg' + (state ? '' : ' off');
        seg.style.left = (100 * (s0 - tmin) / span) + '%';
        seg.style.width = (100 * (s1 - s0) / span) + '%';
        if (state) seg.style.background = lane.color_on;
        const w = (s1 - s0) / span;
        if (w > 0.04) seg.textContent = state ? lane.on_label : lane.off_label;
        track.appendChild(seg);
      });
      // Time cursor for this lane. The gantt uses a fixed full-time scale (it
      // never zooms), so position is a simple fraction of [tmin, tmax].
      const ph = document.createElement('div');
      ph.className = 'ps-playhead';
      track.appendChild(ph);
      registry.addGanttPlayhead(ph, tmin, tmax);
      row.appendChild(lab);
      row.appendChild(track);
      wrap.appendChild(row);
    });
    return wrap;
  }

  // ── X–Y panel (horizontal vs vertical position; not a time series) ──────────
  const XY_HEIGHT  = 300;
  const XY_MIN_WIDTH = 320;
  const XY_TRAIL_S = 0.3;    // trail drawn behind the current frame (s)
  const XY_MARGIN  = { left: 80, right: 16, top: 8, bottom: 34 };   // left = shared y-gutter

  // 1-2-5 tick step giving roughly `n` ticks across `span`.
  function niceStep(span, n) {
    const raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return p * (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10);
  }
  const fmtTick = (v) => String(Math.abs(v) < 1e-9 ? 0 : +v.toFixed(6));

  // Last sample index with t[i] <= tq (-1 if tq precedes the trace).
  function idxAt(t, tq) {
    if (!t.length || tq < t[0]) return -1;
    let lo = 0, hi = t.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (t[mid] <= tq) lo = mid; else hi = mid - 1; }
    return lo;
  }

  function makeXYPanel(panel, t, registry) {
    const wrap = document.createElement('div');
    wrap.className = 'ps-panel';
    wrap.appendChild(makeHead(panel));
    const box = document.createElement('div');
    box.className = 'ps-xy';
    const base = document.createElement('canvas');   // axes + whole trace (redrawn on resize)
    const over = document.createElement('canvas');   // trail + current frame (redrawn per frame)
    box.append(base, over);
    wrap.appendChild(box);

    // `x_offset` is the binocular zero-reference (same as the horizontal panel's
    // `offset`), applied at render time so the spec data stays raw.
    const traces = (panel.traces || []).map((s) => {
      const off = s.x_offset || 0;
      return { ...s, x: off ? s.x.map((v) => (v == null ? null : v + off)) : s.x };
    });
    // Targets first so the eyes draw on top of them.
    traces.sort((a, b) => (b.role === 'target') - (a.role === 'target'));

    // Data bounds, padded and floored at min_span so fixational noise isn't blown
    // up to fill the panel. 1 deg is the same length on both axes.
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    traces.forEach((s) => s.x.forEach((x, i) => {
      const y = s.y[i];
      if (x == null || y == null) return;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }));
    if (!isFinite(x0)) { x0 = y0 = -1; x1 = y1 = 1; }
    const minSpan = panel.min_span || 5;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const spanX = Math.max(x1 - x0, minSpan) * 1.15, spanY = Math.max(y1 - y0, minSpan) * 1.15;

    const M = XY_MARGIN;
    let g = null;   // geometry in CSS px: k = px per deg, (ox, oy) = pixel of (0, 0)
    const X = (x) => g.ox + x * g.k;
    const Y = (y) => g.oy - y * g.k;
    const ok = (s, i) => i >= 0 && s.x[i] != null && s.y[i] != null;
    const clipPlot = (ctx) => { ctx.beginPath(); ctx.rect(g.L, g.T, g.R - g.L, g.B - g.T); ctx.clip(); };

    function layout() {
      // Width follows the data's aspect (equal deg/px), so an H-test is ~square
      // instead of a full-width strip of empty grid; capped at the container.
      const full = (wrap.parentElement && wrap.parentElement.clientWidth) || 600;
      const H = XY_HEIGHT, T = M.top, B = H - M.bottom;
      const W = Math.round(Math.max(XY_MIN_WIDTH,
                  Math.min(full, M.left + M.right + spanX * (B - T) / spanY)));
      wrap.style.maxWidth = W + 'px';
      const L = M.left, R = W - M.right;
      const k = Math.min((R - L) / spanX, (B - T) / spanY);
      g = { W, H, L, R, T, B, k,
            ox: (L + R) / 2 - cx * k, oy: (T + B) / 2 + cy * k };
      const dpr = window.devicePixelRatio || 1;
      for (const c of [base, over]) {
        c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
        c.style.width = W + 'px'; c.style.height = H + 'px';
        c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    }

    function drawBase() {
      const ctx = base.getContext('2d');
      ctx.clearRect(0, 0, g.W, g.H);
      const xmin = (g.L - g.ox) / g.k, xmax = (g.R - g.ox) / g.k;
      const ymin = (g.oy - g.B) / g.k, ymax = (g.oy - g.T) / g.k;
      const step = niceStep(Math.min(xmax - xmin, ymax - ymin), 5);   // same step on both axes
      const hline = (x0p, y0p, x1p, y1p) => { ctx.beginPath(); ctx.moveTo(x0p, y0p); ctx.lineTo(x1p, y1p); ctx.stroke(); };

      // Grid + tick labels
      ctx.font = '10px sans-serif'; ctx.fillStyle = '#475569';
      ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 1;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let n = Math.ceil(xmin / step); n * step <= xmax; n++) {
        const px = X(n * step); hline(px, g.T, px, g.B); ctx.fillText(fmtTick(n * step), px, g.B + 4);
      }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let n = Math.ceil(ymin / step); n * step <= ymax; n++) {
        const py = Y(n * step); hline(g.L, py, g.R, py); ctx.fillText(fmtTick(n * step), g.L - 6, py);
      }
      // Zero lines (dashed, like the time panels' hlines)
      ctx.save(); ctx.strokeStyle = '#aaaaaa'; ctx.setLineDash([6, 4]);
      if (xmin < 0 && xmax > 0) hline(X(0), g.T, X(0), g.B);
      if (ymin < 0 && ymax > 0) hline(g.L, Y(0), g.R, Y(0));
      ctx.restore();
      // Axis labels
      ctx.fillStyle = '#64748b'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(panel.xlabel || '', (g.L + g.R) / 2, g.H - 2);
      ctx.save(); ctx.translate(12, (g.T + g.B) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle'; ctx.fillText(panel.ylabel_axis || '', 0, 0); ctx.restore();

      // Whole trace: one dot per sample, so dot spacing shows speed.
      ctx.save(); clipPlot(ctx);
      for (const s of traces) {
        const tgt = s.role === 'target';
        ctx.fillStyle = s.color; ctx.globalAlpha = tgt ? 0.3 : 0.5;
        const r = tgt ? 1.3 : 1.6;
        for (let i = 0; i < s.x.length; i++) {
          if (!ok(s, i)) continue;
          ctx.beginPath(); ctx.arc(X(s.x[i]), Y(s.y[i]), r, 0, 2 * Math.PI); ctx.fill();
        }
      }
      ctx.restore();
    }

    function drawOver(tNow) {
      const ctx = over.getContext('2d');
      ctx.clearRect(0, 0, g.W, g.H);
      if (tNow == null) return;
      const i1 = idxAt(t, tNow);
      if (i1 < 0) return;
      const i0 = Math.max(0, idxAt(t, tNow - XY_TRAIL_S));
      ctx.save(); clipPlot(ctx); ctx.lineCap = 'round';
      const eyes = traces.filter((s) => s.role !== 'target');
      const circle = (s, r) => { ctx.beginPath(); ctx.arc(X(s.x[i1]), Y(s.y[i1]), r, 0, 2 * Math.PI); };
      // Layered so no trail or halo ever covers another eye's current dot:
      // trails → target rings → white halos → eye dots.
      for (const s of eyes) {
        // Trail: segments thicken + darken toward the current frame.
        ctx.strokeStyle = s.color;
        for (let i = i0 + 1; i <= i1; i++) {
          if (!ok(s, i - 1) || !ok(s, i)) continue;
          const f = (i - i0) / Math.max(1, i1 - i0);
          ctx.globalAlpha = 0.15 + 0.85 * f; ctx.lineWidth = 1 + 2.5 * f;
          ctx.beginPath(); ctx.moveTo(X(s.x[i - 1]), Y(s.y[i - 1]));
          ctx.lineTo(X(s.x[i]), Y(s.y[i])); ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      // Current target: open ring, wide enough to surround an eye dot sitting on it.
      for (const s of traces) {
        if (s.role !== 'target' || !ok(s, i1)) continue;
        ctx.strokeStyle = s.color; ctx.lineWidth = 2; circle(s, 11); ctx.stroke();
      }
      // Current eye position: white halo + filled dot + dark outline, so it stands
      // out against its own same-colored trace and trail.
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      for (const s of eyes) if (ok(s, i1)) { circle(s, 10); ctx.fill(); }
      ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1.5;
      for (const s of eyes) if (ok(s, i1)) { ctx.fillStyle = s.color; circle(s, 7); ctx.fill(); ctx.stroke(); }
      ctx.restore();
    }

    registry.addXY({ resize: () => { layout(); drawBase(); }, draw: drawOver });
    return wrap;
  }

  function render(container, spec) {
    injectStyles();
    container.innerHTML = '';

    const charts = [];
    const playheads = [];        // uPlot panels: [{ u, ph }]
    const ganttPlayheads = [];   // gantt lanes: [{ ph, tmin, tmax }] (fixed scale)
    const ganttTracks = [];      // gantt tracks: [{ track, label }] — aligned to the plot area
    const xyPanels = [];         // X–Y panels: [{ resize(), draw(t) }]
    let playT = null;            // current playhead time (s), or null = hidden
    let syncingGlobal = false;
    const registry = {
      add: (u) => charts.push(u),
      addPlayhead: (u, ph) => playheads.push({ u, ph }),
      addXY: (p) => xyPanels.push(p),
      addGanttPlayhead: (ph, a, b) => ganttPlayheads.push({ ph, tmin: a, tmax: b }),
      addGanttTrack: (track, label) => ganttTracks.push({ track, label }),
      // Size each gantt track to EXACTLY the uPlot plot area (same left + right
      // gutters), so its fixed-scale cursor lines up with the zoomable panels'.
      // Otherwise the track runs wider (esp. when a panel has a right y-axis) and
      // the cursor drifts ahead — most visibly toward the end of the trace.
      alignGantt: () => {
        const u = charts[0];
        if (!u || !u.over || !u.over.clientWidth) return;   // wait until the chart is laid out
        // The gantt label is a FIXED 80px = the panels' left y-gutter (axis size:80), so the
        // track already starts at the plot-area left — do NOT resize the label (that truncated
        // "Scene"/"Target"). Only match the RIGHT gutter (e.g. a right y-axis) so the track
        // spans exactly the plot area and the fixed-scale cursor stays in sync.
        const rightCss = Math.max(0, u.root.clientWidth - (u.over.offsetLeft + u.over.clientWidth));
        for (const { track } of ganttTracks) {
          track.style.marginRight = rightCss + 'px';
        }
      },
      reposition: () => {
        for (const { u, ph } of playheads) {
          if (playT == null) { ph.style.display = 'none'; continue; }
          const x = u.valToPos(playT, 'x');
          const w = u.over.clientWidth;
          // 1px tolerance + clamp so a cursor sitting exactly on a panned edge
          // is still drawn (never flickers off) rather than hidden.
          if (x < -1 || x > w + 1) { ph.style.display = 'none'; }
          else {
            ph.style.left = Math.max(0, Math.min(w, x)) + 'px';
            ph.style.display = 'block';
          }
        }
        // Gantt lanes: fixed full-time scale, so a simple clamped fraction.
        for (const { ph, tmin, tmax } of ganttPlayheads) {
          if (playT == null) { ph.style.display = 'none'; continue; }
          const frac = (playT - tmin) / ((tmax - tmin) || 1);
          ph.style.left = (Math.max(0, Math.min(1, frac)) * 100) + '%';
          ph.style.display = 'block';
        }
        // X–Y panels: current-frame marker + trail (time zoom doesn't affect them).
        for (const p of xyPanels) p.draw(playT);
      },
      setPlayT: (t) => {
        playT = t;
        // When zoomed in time, pan the shared window so the cursor stays visible.
        const c0 = charts[0];
        if (c0 && c0.scales.x.min != null) {
          let min = c0.scales.x.min, max = c0.scales.x.max;
          const width = max - min;
          const zoomed = width < (tmax - tmin) - 1e-9;
          if (zoomed && (t < min || t > max)) {
            if (t < min) { min = t; max = t + width; }
            else         { min = t - width; max = t; }
            if (min < tmin) { min = tmin; max = tmin + width; }
            if (max > tmax) { max = tmax; min = tmax - width; }
            c0.setScale('x', { min, max });   // hook syncs siblings + repositions
            return;
          }
        }
        registry.reposition();
      },
      syncX: (src, min, max) => {
        if (syncingGlobal) return;
        syncingGlobal = true;
        for (const c of charts) {
          if (c === src) continue;
          if (c.scales.x.min !== min || c.scales.x.max !== max) {
            c.setScale('x', { min, max });
          }
        }
        syncingGlobal = false;
      },
    };

    const t = spec.t || [];
    const tmin = t.length ? t[0] : 0;
    const tmax = t.length ? t[t.length - 1] : 1;

    const drawPanel = (panel) => {
      try {
        const el = (panel.type === 'gantt') ? makeGanttPanel(panel, tmin, tmax, registry)
                 : (panel.type === 'xy')    ? makeXYPanel(panel, t, registry)
                 : makeLinePanel(panel, t, [tmin, tmax], registry);
        container.appendChild(el);
      } catch (e) {
        console.error('plotspec: failed to render panel', panel && panel.name, e);
        const err = document.createElement('div');
        err.className = 'ps-ylabel';
        err.textContent = `(panel "${panel && panel.name}" could not be drawn)`;
        container.appendChild(err);
      }
    };
    const isXY = (p) => p.type === 'xy';
    (spec.panels || []).filter((p) => !isXY(p)).forEach(drawPanel);

    // Shared x-axis label under the bottom time panel (X–Y panels carry their own axes).
    const xlabel = document.createElement('div');
    xlabel.className = 'ps-xlabel';
    xlabel.textContent = 'time (s)';
    container.appendChild(xlabel);

    (spec.panels || []).filter(isXY).forEach(drawPanel);

    const hint = document.createElement('div');
    hint.className = 'ps-hint';
    hint.textContent = 'drag to zoom · double-click to reset · hover for values';
    container.appendChild(hint);

    // Responsive width
    const onResize = () => {
      charts.forEach((u) => u.setSize({ width: u.root.parentElement.clientWidth, height: 180 }));
      xyPanels.forEach((p) => p.resize());
      registry.alignGantt();     // re-fit gantt tracks to the (possibly resized) plot area
      registry.reposition();
    };
    window.addEventListener('resize', onResize);
    onResize();

    return {
      // Move the synced time cursor across all panels (t in seconds, null hides).
      setTime: (t) => registry.setPlayT(t),
      destroy() {
        window.removeEventListener('resize', onResize);
        charts.forEach((u) => u.destroy());
        container.innerHTML = '';
      },
    };
  }

  window.PlotSpec = { render };
})();
