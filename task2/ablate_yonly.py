"""Ablation required by the manual 2.4 #4: the same Autoformer with NO optional file.

The model sees only the last LATE hours of standardised log1p y and predicts the next 168.  It uses
the same Series Decomp + Auto-Correlation blocks and the same rolling-origin combination as
train.py; only the input channel set and the read-out differ (a flatten read-out over the context
instead of a per-step one, because the input no longer has one row per predicted step).

usage: python ablate_yonly.py NAME SEEDS FOLDS EPOCHS STRIDE [d=.. layers=.. dk=.. decomp=.. mixer=..]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from data import y_all, N, BAD, CYCLES, H
from blocks import SeriesDecomp, AutoCorrelation, FFN

torch.set_num_threads(int(os.environ.get('Q2_THREADS', '4')))
DEV, OUT = 'cuda', os.environ.get('Q2_OUT', 'runs')
os.makedirs(OUT, exist_ok=True)
CTX = 168          # length of y history the model sees


class YOnly(nn.Module):
    def __init__(self, L=CTX, H=168, d=32, heads=4, layers=2, dk=97, drop=0.1,
                 mixer='autocorr', decomp=1):
        super().__init__()
        self.L, self.H = L, H
        self.embed = nn.Conv1d(1, d, 3, padding=1, padding_mode='replicate')
        self.blocks = nn.ModuleList(_Blk(d, heads, drop, mixer, decomp, dk) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(L * d, H)
        self.tout = nn.Linear(L * d, H) if decomp else None
        for m in (self.out, self.tout):
            if m is not None:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):                                   # x: [B, L, 1]
        h = self.embed(x.transpose(1, 2)).transpose(1, 2)
        trend = 0
        for b in self.blocks:
            h, t = b(h)
            trend = trend + t
        z = self.out(self.norm(h).flatten(1))
        if self.tout is not None:
            z = z + self.tout(trend.flatten(1))
        return z


class _Blk(nn.Module):
    def __init__(self, d, heads, drop, mixer, decomp, dk):
        super().__init__()
        self.mixer = mixer
        self.mix = AutoCorrelation(d, heads, drop) if mixer == 'autocorr' \
            else nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn, self.drop = FFN(d, 2, drop), nn.Dropout(drop)
        self.d1 = SeriesDecomp(dk) if decomp in (1, 2) else None
        self.d2 = SeriesDecomp(dk) if decomp == 2 else None

    def forward(self, x):
        m = self.mix(x) if self.mixer == 'autocorr' else self.mix(x, x, x, need_weights=False)[0]
        x = x + self.drop(m)
        trend = 0
        if self.d1 is not None:
            x, t = self.d1(x)
            trend = trend + t
        x = x + self.drop(self.ffn(x))
        if self.d2 is not None:
            x, t = self.d2(x)
            trend = trend + t
        return x, trend


def run(name, fold, seed, epochs, stride, o):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ly = np.log1p(y_all)
    mu, sd = ly[~BAD].mean(), ly[~BAD].std()
    lyt = torch.tensor((ly - mu) / sd, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y_all, dtype=torch.float32, device=DEV)
    arL, arH = torch.arange(CTX, device=DEV), torch.arange(H, device=DEV)
    ybad = torch.tensor(BAD, device=DEV)
    origins = np.arange(CTX, N - H + 1)
    if fold >= 0:
        a, b = CYCLES[fold]
        b = min(b, N)
        origins = origins[(origins + H <= a) | (origins >= b)]
        evs = np.arange(max(a, CTX), b - H + 1)[::8]
    else:
        evs = np.arange(N - H, N + 1)
    batch = lambda q: lyt[q[:, None] - CTX + arL][..., None]
    model = YOnly(d=int(o.get('d', 32)), layers=int(o.get('layers', 2)), dk=int(o.get('dk', 97)),
                  decomp=int(o.get('decomp', 1)), mixer=o.get('mixer', 'autocorr')).to(DEV)
    P = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tr = torch.tensor(origins[::stride], device=DEV)
    n, bs = len(tr), 128
    spe = max(1, n // bs)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * spe, pct_start=0.3)
    tp = lambda z: torch.expm1(z * sd + mu)
    sc = float(y_all.std())
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(spe):
            q = tr[perm[i * bs:(i + 1) * bs]]
            m = (~ybad[q[:, None] + arH]).float()
            p = tp(model(batch(q)))
            loss = ((((p - yt[q[:, None] + arH]) / sc) ** 2) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
        if ep % 10 == 9 or ep == epochs - 1:
            print(f'  [{name} f{fold} s{seed}] ep {ep+1}/{epochs} ({time.time()-t0:.0f}s)', flush=True)
    model.eval()
    with torch.no_grad():
        pr = np.concatenate([tp(model(batch(torch.tensor(evs[i:i + 512], device=DEV)))).cpu().numpy()
                             for i in range(0, len(evs), 512)])
    grid = np.arange(evs[0], evs[-1] + H)
    W = (np.arange(H) + 1.0) ** -2
    acc, cnt = np.zeros(len(grid)), np.zeros(len(grid))
    for j, o0 in enumerate(evs):
        s0 = o0 - grid[0]
        acc[s0:s0 + H] += W * pr[j]
        cnt[s0:s0 + H] += W
    p = acc / np.maximum(cnt, 1e-12)
    ins = grid < N
    tgt = np.r_[y_all[grid[ins]], np.zeros((~ins).sum())]
    ok = np.r_[BAD[grid[ins]], np.zeros((~ins).sum(), bool)]
    rm = float(np.sqrt((((p - tgt) ** 2) * ~ok).sum() / (~ok).sum()))
    np.savez(f'{OUT}/{name}_f{fold}_s{seed}.npz', pred=p, grid=grid, tgt=tgt, ok=ok,
             params=P, epochs=epochs, fold=fold, seed=seed)
    return rm, P


if __name__ == '__main__':
    name, seeds, folds = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], [int(x) for x in sys.argv[3].split(',')]
    ep, st = int(sys.argv[4]), int(sys.argv[5])
    o = dict(a.split('=') for a in sys.argv[6:])
    res = {}
    for s in seeds:
        for f in folds:
            r, P = run(name, f, s, ep, st, o)
            res[(f, s)] = r
            print(f'{name} fold {f} seed {s}: rmse {r:.2f}  P={P}', flush=True)
    if -1 not in folds:
        print(f'{name} mean {np.mean(list(res.values())):.2f}')
