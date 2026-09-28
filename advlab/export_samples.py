"""Pick test digits for the page: three per class from the MNIST test set, stored as base64 uint8 (784 bytes each).

    python advlab/export_samples.py
"""
from __future__ import annotations

import base64
import json

from common import OUT, load_mnist

x, y = load_mnist("test")
picked = []
for digit in range(10):
    idx = [i for i in range(len(y)) if int(y[i]) == digit][:3]
    for i in idx:
        raw = (x[i, 0] * 255).round().byte().numpy().tobytes()
        picked.append({"i": i, "label": digit, "px": base64.b64encode(raw).decode()})
picked.sort(key=lambda s: (s["i"] % 7, s["label"]))  # interleave classes so the strip does not read 0,0,0,1,1,1…
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "samples.json").write_text(json.dumps({"source": "MNIST test set (LeCun, Cortes & Burges; CC BY-SA 3.0)", "samples": picked}), encoding="utf-8")
print(len(picked), "samples")
