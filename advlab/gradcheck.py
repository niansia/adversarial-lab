"""Check that web/advnet.js computes the same logits and input gradients as PyTorch.

Uses a randomly initialised model and three test images, so it takes about a second of CPU time.
    python advlab/gradcheck.py
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, Net, be_gentle, export, load_mnist

be_gentle(1)
torch.manual_seed(1)
net = Net().eval()
x, y = load_mnist("test")
cases = []
for i in (0, 1, 2):
    xi = x[i:i + 1].clone().requires_grad_(True)
    logits = net(xi)
    loss = F.cross_entropy(logits, y[i:i + 1])
    (g,) = torch.autograd.grad(loss, xi)
    cases.append({"x": xi.detach().flatten().tolist(), "label": int(y[i]), "logits": logits.detach().flatten().tolist(), "grad": g.flatten().tolist()})

with tempfile.TemporaryDirectory() as tmp:
    wpath = Path(tmp) / "w.bin"
    # float32 export for the check, so only the JS arithmetic is compared
    sd = net.state_dict()
    from common import SHAPES
    np.concatenate([sd[n].numpy().astype(np.float32).ravel() for n, _ in SHAPES]).tofile(wpath)
    (Path(tmp) / "cases.json").write_text(json.dumps(cases))
    js = r"""
const fs=require('fs'), A=require(process.argv[1]), dir=process.argv[2];
const buf=fs.readFileSync(dir+'/w.bin'); const W=A.load(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength),'float32');
const cases=JSON.parse(fs.readFileSync(dir+'/cases.json'));
let worstL=0, worstG=0, scale=0;
for (const c of cases){ const x=Float32Array.from(c.x); const r=A.lossGrad(W,x,c.label);
  r.logits.forEach((v,i)=>{worstL=Math.max(worstL,Math.abs(v-c.logits[i]));});
  r.grad.forEach((v,i)=>{worstG=Math.max(worstG,Math.abs(v-c.grad[i])); scale=Math.max(scale,Math.abs(c.grad[i]));}); }
console.log(JSON.stringify({maxLogitError:worstL, maxGradError:worstG, gradScale:scale}));
"""
    out = subprocess.run(["node", "-e", js, str(ROOT / "web" / "advnet.js"), tmp], capture_output=True, text=True, check=True).stdout
res = json.loads(out)
print(res)
ok = res["maxLogitError"] < 1e-4 and res["maxGradError"] < 1e-4 * max(1.0, res["gradScale"])
print("OK" if ok else "MISMATCH")
raise SystemExit(0 if ok else 1)
