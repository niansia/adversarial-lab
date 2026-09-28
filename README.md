# Adversarial Lab

**English** · [繁體中文](README.zh-TW.md)

**Fool a neural network with noise you can barely see, then watch an adversarially trained one hold.**

Adversarial Lab is an in-browser playground for L∞ adversarial attacks on a handwritten-digit classifier. Draw a digit or pick one from the MNIST test set, attack it with FGSM, PGD or targeted PGD, and watch the prediction flip step by step. The same attack is then tried on a model trained with PGD adversarial training, and a decision map shows why one model breaks and the other does not. Everything, including the gradients the attacks follow, is computed on the visitor's CPU; nothing is uploaded.

**Live demo: [niansia.github.io/lab/adversarial](https://niansia.github.io/lab/adversarial/)** (English, 繁體中文, 简体中文)

![A PGD attack turns a 7 into a 3 for the standard model; the decision maps of both models](docs/preview.jpg)

*A PGD attack at ε = 0.2 makes the standard model read this 7 as a 3 (99.5%); the adversarially trained model still reads the same image as a 7 (99.9%). Bottom: a 2-D slice of input space around the digit (→ gradient-sign direction, ↑ random direction, dashed square = ε-box). The standard model's boundary sits inside the ε-box along the gradient direction; the robust model's lies outside it.*

## Results

Two small CNNs (≈207 k parameters each), trained on the MNIST training set on a CPU. Accuracy under attack on the first 2,000 MNIST test images; clean accuracy on all 10,000.

| Model | Clean | PGD ε = 0.1 | ε = 0.2 | ε = 0.3 | ε = 0.35 | ε = 0.4 |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 98.95% | 53.3% | 0.2% | 0.0% | 0.0% | 0.0% |
| Robust (PGD-AT, ε = 0.3) | 97.92% | 95.1% | 92.2% | **85.3%** | 13.2% | 0.0% |
| Transfer: examples from the standard model → robust model | | 96.4% | 95.0% | 92.8% | 62.4% | 10.5% |

<details><summary>FGSM and the full ε grid</summary>

| ε | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard · FGSM | 98.8 | 93.4 | 75.8 | 49.0 | 23.7 | 8.6 | 3.1 | 1.2 | 0.7 |
| Standard · PGD-40 | 98.8 | 90.0 | 53.3 | 8.3 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| Robust · FGSM | 97.5 | 96.3 | 95.5 | 94.6 | 93.8 | 92.9 | 91.9 | 74.2 | 38.5 |
| Robust · PGD-40 | 97.5 | 96.2 | 95.1 | 94.0 | 92.2 | 89.4 | 85.3 | 13.2 | 0.0 |
| Transfer · PGD-40 | 97.5 | 96.8 | 96.4 | 95.5 | 95.0 | 93.7 | 92.8 | 62.4 | 10.5 |

The raw numbers and every setting are in [`web/robustness.json`](web/robustness.json).
</details>

- **Evaluation**: L∞ threat model in pixel space [0, 1]. PGD-40 with step 2.5·ε/40 and one random start; FGSM is a single step of size ε.
- **The cost of robustness**: about one point of clean accuracy (98.95% → 97.92%) and about 8× the compute per epoch (16× in total here, with twice the epochs).
- **Robustness stops at the training budget**: the robust model holds 85% at ε = 0.3, the ε it was trained for, and collapses beyond it (13% at 0.35).
- **Checks against gradient masking** (Athalye et al., 2018): at every ε the multi-step white-box attack is stronger than the one-step attack (PGD ≤ FGSM) and stronger than the transfer attack, and accuracy reaches 0% once ε is large enough. None of these rule out a stronger attack, so **the robust-accuracy numbers are upper bounds**, not certificates. AutoAttack was not run.

## What the demo shows

- **x + δ = x′**: the input, the perturbation δ (blue = darker, amber = brighter, scaled to ε) and the adversarial image, with the prediction and a *FOOLED* / *HELD* verdict.
- **Attacks**: FGSM, PGD (1–100 steps) and targeted PGD ("make it read as 3"), with an ε slider from 0 to 0.4, step-by-step animation, class probabilities, a confidence trajectory and the step at which the decision flipped.
- **Gradient view**: the loss gradient with respect to the pixels. For the standard model it looks like static; for the robust model it follows the strokes of the digit.
- **Transfer**: every adversarial image is also shown to the other model.
- **Decision maps**: a 2-D slice through input space around the input, x + a·u + b·v, where u is the sign of the loss gradient (the FGSM direction) and v is a random ±1 direction. Each cell is coloured by the predicted class and its confidence. Adversarial directions are special: the standard model changes its mind a short step along u but hardly at all along v.
- **Robustness curves**: the offline evaluation above, with the current ε marked.

## How it works

**Model** (`advlab/common.py`): conv 5×5 (16) → ReLU → max-pool → conv 3×3 (32) → ReLU → max-pool → FC 128 → FC 10, on raw pixels in [0, 1] (no normalisation, so ε is directly "change per pixel").

**Training** (`advlab/train.py`): Adam with a one-cycle schedule (max lr 2·10⁻³), batch 128. The standard model trains for 3 epochs. The robust model uses PGD adversarial training (Madry et al., 2018) for 6 epochs: every batch is first attacked with 7-step PGD at ε = 0.3 (step 0.1), with ε ramped up linearly over the first epoch, and the model is trained on the attacked batch. Both models together, including the evaluation above, took 15 minutes on 4 CPU threads.

**In the browser** (`web/advnet.js`): the forward pass *and* the gradient of the loss with respect to the input are written in plain JavaScript (im2col convolutions, max-pool index routing), so FGSM and PGD run exactly as in PyTorch. `advlab/gradcheck.py` compares logits and input gradients against PyTorch (max difference ≈ 10⁻⁸). The weights ship as float16 (405 KB per model). One PGD-40 attack takes about 0.3 s; the two decision maps (24 × 24, coarse to fine) run in two Web Workers in about 5 s and can be cancelled at any time.

## Run it yourself

```bash
pip install -r requirements.txt
```

Serve the page locally (any static server works; the page needs HTTP for its workers):

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/web/>.

Retrain and re-evaluate. The scripts read MNIST from `MNIST_DIR` (default `D:\data\mnist`), the four `*-ubyte.gz` files from the MNIST distribution:

```bash
python advlab/train.py --threads 4
```

Training never touches the GPU (CUDA is hidden), uses a few threads and runs at idle priority, so it can share the machine with other jobs. It rewrites `web/standard.bin`, `web/robust.bin` and `web/robustness.json`.

Check the JavaScript engine against PyTorch:

```bash
python advlab/gradcheck.py
```

`advlab/export_samples.py` writes the sample digits (`web/samples.json`), and `advlab/og_shot.py` renders `docs/preview.jpg` from the running page with Playwright.

## Layout

```
advlab/     model, training, evaluation, gradient check, sample export, preview render (Python, CPU only)
web/        the demo page, the JavaScript engine, the decision-map worker, weights and results
docs/       preview image
```

The live copy on niansia.github.io is the same page with absolute asset paths.

## Limits

- MNIST is the easiest setting for adversarial robustness; the same methods give much weaker guarantees on natural images.
- The models are small and trained briefly on a CPU; the numbers describe these models, not the state of the art.
- PGD with one restart is a strong but not exhaustive attack. The robust accuracies are upper bounds.
- Drawn digits are centred and scaled like MNIST, but very unusual handwriting can still be misread before any attack.

## References

- I. Goodfellow, J. Shlens, C. Szegedy. *Explaining and Harnessing Adversarial Examples.* ICLR 2015 (FGSM).
- A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018 (PGD, adversarial training).
- A. Athalye, N. Carlini, D. Wagner. *Obfuscated Gradients Give a False Sense of Security.* ICML 2018.
- D. Tsipras, S. Santurkar, L. Engstrom, A. Turner, A. Madry. *Robustness May Be at Odds with Accuracy.* ICLR 2019 (human-aligned gradients of robust models).
- Y. LeCun, C. Cortes, C. J. C. Burges. *The MNIST database of handwritten digits* (CC BY-SA 3.0).

## License

Code under the [MIT License](LICENSE). `web/samples.json` contains 30 digits from the MNIST test set, which is available under CC BY-SA 3.0.
