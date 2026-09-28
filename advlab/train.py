"""Train and evaluate the two Adversarial Lab models on the CPU, then write the files the page loads.

    python advlab/train.py                 # defaults: 2 threads, idle priority
    python advlab/train.py --threads 4     # faster when the machine is free

Outputs (web/):
    standard.bin      float16 weights, normal training
    robust.bin        float16 weights, PGD adversarial training (Madry et al., 2018) at ε = 0.3
    robustness.json   clean / FGSM / PGD accuracy vs ε on the MNIST test set, plus training settings
    samples.json      a few test digits for the page (see export_samples.py)

The GPU is never used. Expect roughly 20–40 minutes with 2 threads (the adversarial model dominates).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date

import torch
import torch.nn.functional as F

from common import OUT, Net, be_gentle, export, load_mnist, pgd

EPS_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


def train(model, x, y, *, epochs, adv, eps, alpha, steps, seed, log):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(x) + 127) // 128))
    for ep in range(epochs):
        order, t0 = torch.randperm(len(x), generator=g), time.time()
        # ε ramps up over the first epoch of adversarial training, which keeps PGD-AT from stalling early.
        for b, i in enumerate(range(0, len(x), 128)):
            idx = order[i:i + 128]
            xb, yb = x[idx], y[idx]
            if adv:
                e = eps * min(1.0, (ep * len(x) + i) / len(x))
                model.eval()
                xb = pgd(model, xb, yb, e, alpha * e / eps, steps)
                model.train()
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if b % 100 == 0:
                log(f"  epoch {ep + 1}/{epochs}  batch {b}  loss {loss.item():.3f}  {time.time() - t0:.0f}s")
    model.eval()


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(1) == y).float().mean().item()


def evaluate(model, x, y, *, source=None, steps=40):
    """Accuracy of `model` on examples crafted against `source` (default: itself)."""
    src = source or model
    out = {"fgsm": [], "pgd": []}
    for eps in EPS_GRID:
        xs_f, xs_p = [], []
        for i in range(0, len(x), 500):
            xb, yb = x[i:i + 500], y[i:i + 500]
            xs_f.append(pgd(src, xb, yb, eps, eps, 1, random_start=False))
            xs_p.append(pgd(src, xb, yb, eps, 2.5 * eps / steps, steps))
        out["fgsm"].append(round(accuracy(model, torch.cat(xs_f), y), 4))
        out["pgd"].append(round(accuracy(model, torch.cat(xs_p), y), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--std-epochs", type=int, default=3)
    ap.add_argument("--adv-epochs", type=int, default=6)
    ap.add_argument("--adv-steps", type=int, default=7)
    ap.add_argument("--eval-n", type=int, default=2000, help="test images used for the robustness curves")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    be_gentle(a.threads)
    torch.manual_seed(a.seed)
    log = lambda s: print(s, flush=True)
    xtr, ytr = load_mnist("train")
    xte, yte = load_mnist("test")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    log("standard model")
    std = Net(); train(std, xtr, ytr, epochs=a.std_epochs, adv=False, eps=0, alpha=0, steps=0, seed=a.seed, log=log)
    export(std, OUT / "standard.bin")
    log(f"  clean test accuracy {accuracy(std, xte, yte):.4f}")

    log("robust model (PGD adversarial training, eps 0.3)")
    rob = Net(); train(rob, xtr, ytr, epochs=a.adv_epochs, adv=True, eps=0.3, alpha=0.1, steps=a.adv_steps, seed=a.seed + 1, log=log)
    export(rob, OUT / "robust.bin")
    log(f"  clean test accuracy {accuracy(rob, xte, yte):.4f}")

    log(f"evaluating on the first {a.eval_n} test images")
    xe, ye = xte[:a.eval_n], yte[:a.eval_n]
    report = {
        "date": date.today().isoformat(),
        "eps": EPS_GRID,
        "clean": {"standard": round(accuracy(std, xte, yte), 4), "robust": round(accuracy(rob, xte, yte), 4), "n": len(xte)},
        "standard": evaluate(std, xe, ye),
        "robust": evaluate(rob, xe, ye),
        "transfer": evaluate(rob, xe, ye, source=std),  # crafted on the standard model, tested on the robust one
        "eval": {"n": a.eval_n, "pgd_steps": 40, "pgd_alpha": "2.5·ε/40", "random_start": True, "restarts": 1, "norm": "L∞", "range": "[0, 1]"},
        "train": {"standard": {"epochs": a.std_epochs}, "robust": {"epochs": a.adv_epochs, "eps": 0.3, "steps": a.adv_steps, "alpha": 0.1, "ramp": "ε linear over epoch 1"},
                  "optimizer": "Adam + OneCycle (max lr 2e-3)", "batch": 128, "seed": a.seed, "device": "cpu", "threads": a.threads},
        "minutes": round((time.time() - started) / 60, 1),
    }
    (OUT / "robustness.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(json.dumps({k: report[k] for k in ("clean", "standard", "robust", "transfer", "minutes")}))


if __name__ == "__main__":
    main()
