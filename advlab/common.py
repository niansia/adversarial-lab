"""Shared pieces for Adversarial Lab: model, MNIST loading, attacks, CPU-friendly setup.

Everything here is meant to run on the CPU with few threads at low priority, so it can share the machine with other
jobs. Nothing touches the GPU (CUDA is hidden before torch is imported).
"""
from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never use the GPU

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web"
DATA = Path(os.environ.get("MNIST_DIR", r"D:\data\mnist"))

# Order and shapes must match web/advnet.js
SHAPES = [("conv1.weight", (16, 1, 5, 5)), ("conv1.bias", (16,)), ("conv2.weight", (32, 16, 3, 3)), ("conv2.bias", (32,)),
          ("fc1.weight", (128, 1568)), ("fc1.bias", (128,)), ("fc2.weight", (10, 128)), ("fc2.bias", (10,))]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def be_gentle(threads: int) -> None:
    """Few threads and the lowest scheduling priority, so other work on this machine always goes first."""
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if sys.platform == "win32":
        import ctypes
        IDLE_PRIORITY_CLASS = 0x40
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p  # the pseudo-handle is -1; the default int return type truncates it
        k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), IDLE_PRIORITY_CLASS):
            print("warning: could not lower the process priority", flush=True)
    else:
        os.nice(19)


def load_mnist(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = "train" if split == "train" else "t10k"
    with gzip.open(DATA / f"{prefix}-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with gzip.open(DATA / f"{prefix}-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return torch.from_numpy(x.astype(np.float32) / 255.0), torch.from_numpy(y.astype(np.int64))


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, alpha: float, steps: int, random_start: bool = True) -> torch.Tensor:
    """Untargeted L∞ PGD in pixel space [0, 1]. steps=1, alpha=eps, random_start=False is FGSM."""
    if eps == 0:
        return x
    delta = (torch.rand_like(x) * 2 - 1) * eps if random_start else torch.zeros_like(x)
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        (g,) = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def export(model: nn.Module, path: Path) -> None:
    """float16 weights in the order of SHAPES."""
    sd = model.state_dict()
    parts = []
    for name, shape in SHAPES:
        t = sd[name].detach().cpu().float()
        assert tuple(t.shape) == shape, (name, t.shape)
        parts.append(t.numpy().astype(np.float16).ravel())
    path.write_bytes(np.concatenate(parts).tobytes())
