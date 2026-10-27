"""
tstr.py
=======
Train-on-synthetic, test-on-real (TSTR) helpers shared by the TSTR notebooks.

The classifier, its training loop and the per-clip scoring are exactly those of
notebooks/tstr_pilot_esc50.ipynb (where the recipe was chosen and frozen), moved
here so the full study and later studies use the identical code instead of a copy.

    ProbeCNN            the small spectrogram CNN (probe architecture)
    middle_crop         one fixed crop per clip: the middle valid crop
    batch_fixed         sampler over a fixed set of spectrograms
    batch_random_crops  sampler drawing a fresh random valid crop per draw (Real-full)
    train_clf           one training run with a fixed recipe
    clip_probs          per-clip class probabilities, averaged over each clip's crops
    accuracy            per-clip accuracy from probabilities
    grouped_bootstrap   percentile interval of a statistic, resampling groups
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import common as C


class ProbeCNN(nn.Module):
    """Four blocks of two 3x3 conv + BN + ReLU, 2x2 max-pool; average over time, keep frequency."""

    def __init__(self, n_classes: int, n_bands: int = 128, width=(32, 64, 128, 256), p_drop: float = 0.3):
        super().__init__()
        layers, c_in = [], 1
        for c in width:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(width[-1] * (n_bands // 2 ** len(width)), n_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean(dim=-1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def middle_crop(split: dict, idx: torch.Tensor, cfg: C.Config) -> torch.Tensor:
    """One fixed crop per selected clip: the middle of its valid crops (same energy rule as training)."""
    starts, valid = C.valid_crop_starts(split["waveforms"][idx], cfg)
    out = []
    for j, i in enumerate(idx.tolist()):
        v = torch.nonzero(valid[j]).squeeze(1)
        s = int(starts[v[len(v) // 2]])
        out.append(split["waveforms"][i, :, s:s + cfg.clip_samples])
    return torch.stack(out)


def batch_fixed(X: torch.Tensor, y: torch.Tensor, device) -> Callable:
    """Sampler over a fixed, class-balanced set: uniformly random items, with replacement."""
    X, y = X.to(device), y.to(device)

    def get(g: torch.Generator, n: int):
        i = torch.randint(len(y), (n,), generator=g).to(device)
        return X[i], y[i]
    return get


def batch_random_crops(split: dict, cfg: C.Config, frontend, device) -> Callable:
    """Sampler for Real-full: a uniformly random clip, then a fresh random valid crop of it."""
    wave = split["waveforms"].to(device)
    y = split["labels"].to(device)
    starts, valid = C.valid_crop_starts(split["waveforms"], cfg)
    starts, valid = starts.to(device), valid.float().to(device)
    ar = torch.arange(cfg.clip_samples, device=device)[None]

    def get(g: torch.Generator, n: int):
        i = torch.randint(len(y), (n,), generator=g).to(device)
        s = starts[torch.multinomial(valid[i], 1).squeeze(1)]
        w = torch.gather(wave[i, 0], 1, s[:, None] + ar).unsqueeze(1)
        return frontend(w), y[i]
    return get


_fmask = torchaudio.transforms.FrequencyMasking(16, iid_masks=True)
_tmask = torchaudio.transforms.TimeMasking(24, iid_masks=True)


def train_clf(get_batch: Callable, n_classes: int, lr: float, steps: int, specaug: bool, seed: int, device,
              batch: int = 64, soft_targets: Callable | None = None) -> ProbeCNN:
    """One training run: AdamW (wd 1e-4), one-cycle LR, fixed step count, final weights.

    SpecAugment (when on): one frequency mask (<= 16 bands) and one time mask (<= 24 frames)
    per clip, filled with -1 (the front end's floor, i.e. silence).
    soft_targets, if given, maps (x, y) -> loss for distillation-style training; otherwise
    cross-entropy on the hard labels.
    """
    torch.manual_seed(seed)
    model = ProbeCNN(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    use_amp = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        x, y = get_batch(g, batch)
        if specaug:
            x = _tmask(_fmask(x, mask_value=-1.0), mask_value=-1.0)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = soft_targets(logits, x, y) if soft_targets is not None else F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return model.eval()


@torch.no_grad()
def clip_probs(model: nn.Module, X: torch.Tensor, clip_idx: torch.Tensor, n_classes: int, batch: int = 256) -> np.ndarray:
    """Per-clip class probabilities: softmax averaged over each clip's crops. (n_clips, n_classes)."""
    p = torch.cat([model(X[i:i + batch]).float().softmax(-1).cpu() for i in range(0, len(X), batch)])
    n = int(clip_idx.max()) + 1
    out = torch.zeros(n, n_classes).index_add_(0, clip_idx, p)
    return (out / torch.bincount(clip_idx, minlength=n)[:, None]).numpy()


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean())


def grouped_bootstrap(stat: Callable[[np.ndarray], float], groups: Sequence, n_boot: int = 2000,
                      seed: int = 0, q=(2.5, 97.5)) -> tuple:
    """Percentile interval of stat(idx) where idx resamples whole groups with replacement."""
    ug, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals = [stat(np.concatenate([members[j] for j in rng.integers(len(ug), size=len(ug))])) for _ in range(n_boot)]
    return tuple(float(v) for v in np.percentile(vals, q))
