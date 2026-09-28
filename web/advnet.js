/* AdvNet: a tiny MNIST CNN with forward pass and gradients with respect to the input, in plain JavaScript (CPU only).
   Used by /lab/adversarial/ for FGSM / PGD attacks, saliency and the decision map (in a worker).

   Architecture (matches Net in advlab/common.py of github.com/niansia/adversarial-lab):
     conv1 1→16 5×5 pad 2 → ReLU → maxpool 2   (28→14)
     conv2 16→32 3×3 pad 1 → ReLU → maxpool 2  (14→7)
     fc1 1568→128 → ReLU → fc2 128→10
   Input: 28×28 floats in [0, 1], row-major. Works in browsers, workers and Node (for the gradient check). */
(function (root) {
  'use strict';
  const SHAPES = [['conv1.weight', [16, 1, 5, 5]], ['conv1.bias', [16]], ['conv2.weight', [32, 16, 3, 3]], ['conv2.bias', [32]],
    ['fc1.weight', [128, 1568]], ['fc1.bias', [128]], ['fc2.weight', [10, 128]], ['fc2.bias', [10]]];

  // IEEE half → float
  function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function load(buffer, dtype) {
    const n = SHAPES.reduce((a, [, s]) => a + s.reduce((x, y) => x * y, 1), 0);
    let flat;
    if (dtype === 'float16') { const u = new Uint16Array(buffer); if (u.length !== n) throw new Error(`expected ${n} weights, got ${u.length}`); flat = new Float32Array(n); for (let i = 0; i < n; i++) flat[i] = halfToFloat(u[i]); }
    else { flat = new Float32Array(buffer); if (flat.length !== n) throw new Error(`expected ${n} weights, got ${flat.length}`); }
    const W = {}; let o = 0;
    for (const [name, shape] of SHAPES) { const len = shape.reduce((x, y) => x * y, 1); W[name] = flat.subarray(o, o + len); o += len; }
    return W;
  }

  /* ---------- layers (single image, CHW) ---------- */
  // im2col: unfold every K×K window into a column once, then each output channel is one long, contiguous
  // multiply-add per weight (the direct loop spent most of its time on 14-pixel inner loops).
  const colCache = {};
  function conv(x, C, H, Wd, w, b, O, K, pad) {
    const P = H * Wd, KK = C * K * K, key = `${KK}x${P}`;
    const col = colCache[key] || (colCache[key] = new Float32Array(KK * P));
    for (let c = 0; c < C; c++) for (let ky = 0; ky < K; ky++) for (let kx = 0; kx < K; kx++) {
      const row = ((c * K + ky) * K + kx) * P, dy = ky - pad, dx = kx - pad, xc = c * P;
      for (let r = 0; r < H; r++) {
        const sr = r + dy, base = row + r * Wd;
        if (sr < 0 || sr >= H) { col.fill(0, base, base + Wd); continue; }
        const src = xc + sr * Wd + dx;
        for (let q = 0; q < Wd; q++) { const sq = q + dx; col[base + q] = sq >= 0 && sq < Wd ? x[src + q] : 0; }
      }
    }
    const y = new Float32Array(O * P);
    for (let o = 0; o < O; o++) {
      const yo = o * P, wo = o * KK;
      y.fill(b[o], yo, yo + P);
      for (let k = 0; k < KK; k++) {
        const wv = w[wo + k]; if (wv === 0) continue;
        const ro = k * P;
        for (let p = 0; p < P; p++) y[yo + p] += wv * col[ro + p];
      }
    }
    return y;
  }
  // gradient of conv output w.r.t. its input
  function convBack(g, C, H, Wd, w, O, K, pad) {
    const dx = new Float32Array(C * H * Wd);
    for (let o = 0; o < O; o++) {
      const go = o * H * Wd;
      for (let c = 0; c < C; c++) {
        const xc = c * H * Wd, wk = (o * C + c) * K * K;
        for (let ky = 0; ky < K; ky++) for (let kx = 0; kx < K; kx++) {
          const wv = w[wk + ky * K + kx]; if (wv === 0) continue;
          const dy = ky - pad, ddx = kx - pad;
          const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -ddx), x1 = Math.min(Wd, Wd - ddx);
          for (let r = y0; r < y1; r++) {
            const gr = go + r * Wd, xr = xc + (r + dy) * Wd + ddx;
            for (let q = x0; q < x1; q++) dx[xr + q] += wv * g[gr + q];
          }
        }
      }
    }
    return dx;
  }
  function relu(x) { const y = new Float32Array(x.length); for (let i = 0; i < x.length; i++) y[i] = x[i] > 0 ? x[i] : 0; return y; }
  function pool(x, C, H, Wd) {
    const h = H >> 1, w = Wd >> 1, y = new Float32Array(C * h * w), idx = new Int32Array(C * h * w);
    for (let c = 0; c < C; c++) for (let r = 0; r < h; r++) for (let q = 0; q < w; q++) {
      let best = -Infinity, bi = 0;
      for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) { const i = c * H * Wd + (2 * r + a) * Wd + 2 * q + b; if (x[i] > best) { best = x[i]; bi = i; } }
      const o = c * h * w + r * w + q; y[o] = best; idx[o] = bi;
    }
    return {y, idx};
  }
  function linear(x, w, b, O) {
    const I = x.length, y = new Float32Array(O);
    for (let o = 0; o < O; o++) { let s = b[o]; const wo = o * I; for (let i = 0; i < I; i++) s += w[wo + i] * x[i]; y[o] = s; }
    return y;
  }

  function forward(W, x) {
    const a1 = conv(x, 1, 28, 28, W['conv1.weight'], W['conv1.bias'], 16, 5, 2), r1 = relu(a1), p1 = pool(r1, 16, 28, 28);
    const a2 = conv(p1.y, 16, 14, 14, W['conv2.weight'], W['conv2.bias'], 32, 3, 1), r2 = relu(a2), p2 = pool(r2, 32, 14, 14);
    const h = linear(p2.y, W['fc1.weight'], W['fc1.bias'], 128), hr = relu(h);
    const logits = linear(hr, W['fc2.weight'], W['fc2.bias'], 10);
    return {logits, cache: {a1, p1, a2, p2, h, hr}};
  }
  function softmax(z) {
    const m = Math.max(...z), e = Array.from(z, v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }
  // d(loss)/d(input) given d(loss)/d(logits)
  function backward(W, cache, dlogits) {
    const {a1, p1, a2, p2, h} = cache;
    const dhr = new Float32Array(128), w2 = W['fc2.weight'];
    for (let o = 0; o < 10; o++) { const g = dlogits[o]; if (!g) continue; for (let i = 0; i < 128; i++) dhr[i] += w2[o * 128 + i] * g; }
    for (let i = 0; i < 128; i++) if (h[i] <= 0) dhr[i] = 0;
    const dp2 = new Float32Array(1568), w1 = W['fc1.weight'];
    for (let o = 0; o < 128; o++) { const g = dhr[o]; if (!g) continue; const wo = o * 1568; for (let i = 0; i < 1568; i++) dp2[i] += w1[wo + i] * g; }
    const dr2 = new Float32Array(32 * 14 * 14);
    for (let i = 0; i < dp2.length; i++) dr2[p2.idx[i]] += dp2[i];
    for (let i = 0; i < dr2.length; i++) if (a2[i] <= 0) dr2[i] = 0;
    const dp1 = convBack(dr2, 16, 14, 14, W['conv2.weight'], 32, 3, 1);
    const dr1 = new Float32Array(16 * 28 * 28);
    for (let i = 0; i < dp1.length; i++) dr1[p1.idx[i]] += dp1[i];
    for (let i = 0; i < dr1.length; i++) if (a1[i] <= 0) dr1[i] = 0;
    return convBack(dr1, 1, 28, 28, W['conv1.weight'], 16, 5, 2);
  }
  // Cross-entropy of `label`, its gradient w.r.t. the input, and the probabilities.
  function lossGrad(W, x, label) {
    const {logits, cache} = forward(W, x), p = softmax(logits);
    const d = p.slice(); d[label] -= 1;
    return {loss: -Math.log(Math.max(p[label], 1e-12)), grad: backward(W, cache, d), probs: p, logits};
  }
  function predict(W, x) { return softmax(forward(W, x).logits); }

  /* ---------- attacks (L∞, pixel range [0, 1]) ---------- */
  const clip01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
  function project(xAdv, x0, eps) { for (let i = 0; i < 784; i++) xAdv[i] = clip01(Math.min(x0[i] + eps, Math.max(x0[i] - eps, xAdv[i]))); return xAdv; }
  // One PGD step. Untargeted: climb the loss of the true label. Targeted: descend the loss of the target.
  function pgdStep(W, xAdv, x0, {eps, alpha, label, target}) {
    const targeted = target !== undefined && target !== null && target >= 0;
    const {grad, probs, loss} = lossGrad(W, xAdv, targeted ? target : label);
    const s = targeted ? -1 : 1;
    const next = new Float32Array(784);
    for (let i = 0; i < 784; i++) next[i] = xAdv[i] + s * alpha * Math.sign(grad[i]);
    return {x: project(next, x0, eps), probs, loss, grad};
  }
  function randomStart(x0, eps, rand = Math.random) { const x = new Float32Array(784); for (let i = 0; i < 784; i++) x[i] = x0[i] + (rand() * 2 - 1) * eps; return project(x, x0, eps); }
  function fgsm(W, x0, eps, label) { return pgdStep(W, Float32Array.from(x0), x0, {eps, alpha: eps, label}).x; }

  const API = {SHAPES, load, forward, backward, lossGrad, predict, softmax, pgdStep, randomStart, fgsm, project};
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.AdvNet = API;
})(typeof self !== 'undefined' ? self : this);
