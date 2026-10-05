"""Leave-one-year-out CV and final fitting for the encoder-only Autoformer.

usage:
  python train.py NAME SEEDS FOLDS EPOCHS STRIDE [key=value ...]

SEEDS  e.g. 0,1,2        FOLDS  0,1,2,3,4 (leave-one-year-out) or -1 (all data -> hidden block)

keys (only what the final model actually uses; everything else is a fixed design choice):
  spec=v1        which covariate channels enter (v0 = raw only, v1 = + harmonics/polynomials)
  win=6,24,168   trailing causal mean widths added to the raw channels
  decomp=1       Series Decomposition inside every block (0 off, 1 after the mixer, 2 after both)
  mixer=autocorr Auto-Correlation (attention = point-wise, for the ablation)
  dk=97          width of the decomposition's moving average
  gsum=1         window-summary read-out
  bridge=1,6,24,168  trailing-y windows in the recent-y bridge
  bex=3          exact recent-y lags in the bridge
  d=32 layers=2 drop=0.1
  lr=1e-3 wd=1e-4 bs=128
  ow=2           rolling origins are combined with weight (horizon step + 1)^-ow
  evs=8          stride between evaluation origins (test-time inference uses every origin)

A run writes runs/NAME_f{FOLD}_s{SEED}.npz with the per-step predictions on the evaluation grid,
the targets and the validity mask, so ensembles can be formed offline without re-training.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import y_all, N, BAD, CYCLES, H, covariates, base_channels
import model as M
from encdec_model import AFEncDec

DEV = 'cuda'
OUT = os.environ.get('Q2_OUT', 'runs')
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(int(os.environ.get('Q2_THREADS', '4')))
try:
    torch.backends.cuda.cufft_plan_cache.max_size = 0
except Exception:
    pass

DEFAULTS = dict(spec='v1', win='6,24,168', decomp=1, mixer='autocorr', dk=97, gsum=1,
                bridge='1,6,24,168', bex=3, d=32, layers=2, drop=0.1,
                lr=1e-3, wd=1e-4, bs=128, ow=2, evs=8,
                arch='enc', lb=168, elayers=2, dlayers=1, gmlp=0, clamp=0)


def parse(v):
    if isinstance(v, str):
        for f in (int, float):
            try:
                return f(v)
            except ValueError:
                pass
    return v


def opts_from(argv):
    o = dict(DEFAULTS)
    for a in argv:
        k, v = a.split('=')
        o[k] = parse(v)
    o['win'] = [int(x) for x in str(o['win']).split(',') if x]
    o['bridge'] = [] if str(o['bridge']) in ('none', '0', '') else [int(x) for x in str(o['bridge']).split(',')]
    o['bex'] = int(o['bex'])
    o['dk'] = int(o['dk'])
    o['layers'] = int(o['layers'])
    o['bs'] = int(o['bs'])
    o['evs'] = int(o['evs'])
    return o


_COV = {}


def get_cov(o):
    key = (o['spec'], tuple(o['win']))
    if key not in _COV:
        Z, names = covariates(o['spec'], o['win'])
        _COV[key] = (torch.tensor(Z, device=DEV), names)
    return _COV[key]


def run(name, fold, seed, epochs, stride, o):
    """fold >= 0: train on the other four years, score the held-out year.  fold = -1: train on
    everything and predict the hidden block with every rolling origin."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    final = fold == -1
    wb = max(o['bridge'] + list(range(o['bex']))) if (o['bridge'] or o['bex']) else 0   # y look-back
    lb = int(o['lb']) if o['arch'] == 'encdec' else 0                                  # covariate look-back
    # the margin must cover BOTH the y look-back and the covariate look-back, so that no training
    # window reaches into the held-out year through either channel
    wb = max(wb, lb)
    arw = torch.arange(max(wb, 1), device=DEV)
    ybad = torch.tensor(BAD, device=DEV)

    origins = np.arange(0, N - H + 1)                        # windows fully inside the observed data
    if not final:
        a, b = CYCLES[fold]
        b = min(b, N)
        origins = origins[(origins + H <= a - wb) | (origins >= b + wb)]   # no leakage across the fold
        evs = np.arange(max(a, wb), b - H + 1)[::int(o['evs'])]
    else:
        evs = np.arange(N - H, N + 1)                        # every origin that covers the hidden block

    ly_all = np.log1p(y_all)
    rows = np.zeros(N, bool)
    for q in origins[::H]:
        rows[q:q + H] = True
    mu, sd = ly_all[rows & ~BAD].mean(), ly_all[rows & ~BAD].std()
    lyt = torch.tensor((ly_all - mu) / sd, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y_all, dtype=torch.float32, device=DEV)
    Zc, names = get_cov(o)
    arH = torch.arange(H, device=DEV)

    def batch(q):
        z = Zc[q[:, None] + arH]                            # the H covariate steps to be predicted
        if not lb:
            return z, None
        arL = torch.arange(lb, device=DEV)
        return z, Zc[q[:, None] - lb + arL]                # the lb steps before the origin

    def yfeat(q):
        """Summaries of the y values strictly before each origin: trailing means over q-w..q-1 and
        the exact lags q-1, q-2, ...  y[q] is the first target and must never appear here."""
        cols = [lyt[q[:, None] - 1 - arw[:w]].mean(1) for w in o['bridge']]
        cols += [lyt[q - 1 - i] for i in range(o['bex'])]
        return torch.stack(cols, 1) if cols else None

    nr = len(base_channels(o['spec'])) if o['gsum'] else 0
    if o['arch'] == 'encdec':
        model = AFEncDec(Zc.shape[1], H=H, L=max(lb, 1), d=int(o['d']),
                         elayers=int(o['elayers']), dlayers=int(o['dlayers']), dk=o['dk'],
                         drop=float(o['drop']), mixer=o['mixer'], decomp=int(o['decomp']),
                         n_raw=nr, bridge=o['bridge'], bridge_exact=o['bex'],
                         gmlp=int(o['gmlp'])).to(DEV)
    else:
        model = M.AFNet(Zc.shape[1], H=H, d=int(o['d']), layers=o['layers'], dk=o['dk'],
                        drop=float(o['drop']), mixer=o['mixer'], decomp=int(o['decomp']),
                        n_raw=nr, bridge=o['bridge'], bridge_exact=o['bex'],
                        gmlp=int(o['gmlp'])).to(DEV)
    P = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tr = torch.tensor(origins[::stride], device=DEV)
    n, bs = len(tr), o['bs']
    spe = max(1, n // bs)
    opt = torch.optim.AdamW(model.parameters(), float(o['lr']), weight_decay=float(o['wd']))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, float(o['lr']), total_steps=epochs * spe, pct_start=0.3)
    to_phys = lambda z: torch.expm1(z * sd + mu)
    if o['clamp']:
        # The expm1 link is explosive: pred = expm1(z*sd + mu), so a standardised output of z = 6
        # already means exp(6*sd), far beyond anything the series ever reaches.  Clamping z from
        # ABOVE at the largest value the observed series implies keeps a diverging run from
        # producing a meaningless forecast.  Gradients inside the normal range are untouched, and
        # for the encoder-only model nothing ever hits the bound.
        hi = float(np.log1p(float(y_all[~BAD].max()) * 1.5) - mu) / sd
        base_phys = to_phys
        to_phys = lambda z: base_phys(torch.clamp(z, max=hi))
    scale, t0 = float(y_all.std()), time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.
        for i in range(spe):
            q = tr[perm[i * bs:(i + 1) * bs]]
            tgt = yt[q[:, None] + arH]
            m = (~ybad[q[:, None] + arH]).float()
            bx, bxb = batch(q)
            pred = to_phys(model(bx, bxb, yfeat(q)))
            loss = (((pred - tgt) / scale) ** 2 * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % 10 == 9 or ep == epochs - 1:
            print(f'  [{name} f{fold} s{seed}] ep {ep+1}/{epochs} train rmse~{scale*(tot/spe)**.5:.1f} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    model.eval()

    @torch.no_grad()
    def predict(qs, chunk=512):
        out = []
        for i in range(0, len(qs), chunk):
            q = torch.tensor(qs[i:i + chunk], device=DEV)
            bx, bxb = batch(q)
            out.append(to_phys(model(bx, bxb, yfeat(q))).cpu().numpy())
        return np.concatenate(out)

    pr = predict(evs)
    grid = np.arange(evs[0], evs[-1] + H)
    # every step is predicted by every origin that covers it, at a different horizon step k;
    # ow > 0 down-weights large k, i.e. trusts short horizons more
    W = np.ones(H) if o['ow'] <= 0 else (np.arange(H) + 1.0) ** (-o['ow'])
    acc, cnt = np.zeros(len(grid)), np.zeros(len(grid))
    for j, o0 in enumerate(evs):
        s0 = o0 - grid[0]
        acc[s0:s0 + H] += W * pr[j]
        cnt[s0:s0 + H] += W
    pavg = acc / np.maximum(cnt, 1e-12)
    inside = grid < N
    tgt = np.r_[y_all[grid[inside]], np.zeros((~inside).sum())]
    ok = np.r_[BAD[grid[inside]], np.zeros((~inside).sum(), bool)]
    rmse = float(np.sqrt((((pavg - tgt) ** 2) * ~ok).sum() / (~ok).sum()))
    np.savez(f'{OUT}/{name}_f{fold}_s{seed}.npz', pred=pavg, grid=grid, tgt=tgt, ok=ok,
             params=P, epochs=epochs, fold=fold, seed=seed)
    return rmse, P, pavg, tgt, ok


if __name__ == '__main__':
    import sys
    name, seeds, folds = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], [int(x) for x in sys.argv[3].split(',')]
    epochs, stride = int(sys.argv[4]), int(sys.argv[5])
    o = opts_from(sys.argv[6:])
    res = {}
    for s in seeds:
        for f in folds:
            r, P, *_ = run(name, f, s, epochs, stride, o)
            res[(f, s)] = r
            print(f'{name} fold {f} seed {s}: rmse {r:.2f}  P={P}', flush=True)
    if -1 not in folds:
        for s in seeds:
            v = [res[(f, s)] for f in folds]
            print(f'{name} seed {s}: folds ' + ' '.join(f'{x:.1f}' for x in v) + f'  mean {np.mean(v):.2f}', flush=True)
        print(f'{name} SINGLE-SEED POOLED over {len(res)} runs: '
              f'{np.sqrt(np.mean(np.square(list(res.values())))):.2f}', flush=True)
