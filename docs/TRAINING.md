# Training

How BRIN — the Bird's-eye Risk Inference Network — was trained.

## Why from scratch

BRIN's input is a 6-channel bird's-eye velocity raster where channels encode
signed velocity in metres per second. No pretrained network exists for this
modality: ImageNet weights are built for 3-channel RGB photographs and carry no
information about a velocity field. Transfer learning is conceptually inapplicable,
so training from scratch is the correct choice — not a shortcut.

## Data

- Source: the procedural simulator (see `ARCHITECTURE.md`, stage 6).
- ~400,000 labelled timesteps, each a (6, 64, 64) raster.
- Class balance: SAFE ~64% / IMMINENT ~26% / CAUTION ~10%.
- Split: train / validation held out at the scenario level (no timestep leakage
  between a scenario's frames across the split).

The full uncompressed raster dataset is ~15.7 GB, so training runs on Kaggle GPU;
the dataset is never loaded on the development laptop.

## Model

Residual CNN, ~491k parameters:

```
stem  (6 → 32)   64×64
res1  (32)
down1 (32 → 64)  32×32
res2  (64)
down2 (64 → 128) 16×16
res3  (128)
global average pool → 128-vector
head  (128 → 64 → 3)
```

Each downsampling halves spatial size and doubles channels, so by the final block
each cell's receptive field covers enough floor for two distant agents to interact.

## The instability problem and the fix

The first run at learning rate 1e-3 was unstable: per-class recall lurched between
epochs (one epoch all-SAFE, the next all-IMMINENT). The optimiser was overshooting.

Fixes:
- Learning rate lowered to **3e-4** (smaller, more stable steps).
- **Cosine annealing** scheduler so late epochs fine-tune with tiny steps.
- **Class-weighted cross-entropy** (inverse-frequency) so rare-class errors cost more.

Training loop is the standard `zero_grad → forward → loss → backward → step`, with
`scheduler.step()` once per epoch.

## Results

| Model | Accuracy | IMMINENT recall |
|---|---|---|
| Naive baseline (always SAFE) | 63.1% | 0% |
| **BRIN** | **92.7%** | **93%** |

Per-class recall: SAFE 0.92 / CAUTION 0.91 / IMMINENT 0.93.

The metric that matters for a safety system is **IMMINENT recall** — the fraction of
real collisions caught. Accuracy alone is misleading on imbalanced data: the naive
baseline scores a plausible-looking 63% while catching zero collisions.

## Why the baseline comparison matters

If BRIN were merely re-deriving the TTC formula it was trained on, it would only
match a physics baseline. Instead it beats a trivial baseline by 93 points on
collision recall — evidence that it learned conflict structure from motion patterns,
not a memorised shortcut. Grad-CAM (stage 9) corroborates this: attention
concentrates on the collision zone between the two agents.

## Reproducing

Training code is in `notebooks/04_brin_training.ipynb`. The trained weights
(`models/brin/brin_final.pt`, ~2 MB) are gitignored but regenerable from the
simulator + rasteriser + this notebook.