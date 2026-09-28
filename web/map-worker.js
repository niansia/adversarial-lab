/* Decision map worker: classifies a grid of points on the plane x0 + a·u + b·v (a, b ∈ [-span, span]) for one model.
   Coarse-to-fine levels are posted as they finish. The page cancels a job by terminating the worker, so the loop never
   has to yield (timers in background tabs are throttled and made the old yielding version several times slower).
   One thread, CPU only. */
importScripts('advnet.js');
const models = {};

async function weights(name, url) {
  if (!models[name]) models[name] = AdvNet.load(await (await fetch(url)).arrayBuffer(), 'float16');
  return models[name];
}

self.onmessage = async ({data}) => {
  const {id, model, url, x0, u, v, span, levels} = data;
  const W = await weights(model, url);
  const x = new Float32Array(784);
  for (const n of levels) {
    const cls = new Uint8Array(n * n), conf = new Float32Array(n * n);
    for (let r = 0; r < n; r++) {
      const b = span - (2 * span * (r + .5)) / n;               // top row = +span
      for (let c = 0; c < n; c++) {
        const a = -span + (2 * span * (c + .5)) / n;
        for (let i = 0; i < 784; i++) { const t = x0[i] + a * u[i] + b * v[i]; x[i] = t < 0 ? 0 : t > 1 ? 1 : t; }
        const p = AdvNet.predict(W, x);
        let k = 0; for (let j = 1; j < 10; j++) if (p[j] > p[k]) k = j;
        cls[r * n + c] = k; conf[r * n + c] = p[k];
      }
    }
    self.postMessage({id, model, n, cls, conf}, [cls.buffer, conf.buffer]);
  }
  self.postMessage({id, model, done: true});
};
