"""Builds the report's Q2 figures from the finished runs.

  fig_calibration.pdf   where the error lives (concentration by magnitude, and E[y|p] against p)
  fig_shift.pdf         the covariate-shifted validation (level bias and RMSE vs state-1 share)
  fig_forecast.pdf      the submitted forecast against the last 336 observed steps

usage: python report_figs.py
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.environ.get('Q2_OUT', 'runs')
FIG = os.environ.get('Q2_FIG', 'figures')
# the scripts sit next to this file, so the default data locations are resolved relative to it
# rather than to the working directory, which means this runs from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
MEMBERS = ['C1_l3k97', 'C2_l2k25', 'C3_e25', 'C4_l2k97']
DATA = os.environ.get('Q2_DATA', os.path.join(HERE, '..', 'Data'))
os.makedirs(FIG, exist_ok=True)

INK, ACCENT, WARM, GREY = '#1a1a1a', '#2b6cb0', '#c05621', '#718096'


def tidy(a):
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.25, linewidth=0.6)
    a.set_axisbelow(True)
    return a


def load_ens():
    per = {}
    for n in MEMBERS:
        for f in sorted(glob.glob(f'{OUT}/{n}_f*_s*.npz')):
            z = np.load(f, allow_pickle=True)
            per.setdefault(int(z['fold']), []).append(z)
    return {fold: (np.mean([x['pred'] for x in ds], 0), ds[0]['tgt'], ds[0]['ok'])
            for fold, ds in per.items()}


def fig_calibration(ens):
    ps = np.concatenate([p[~ok] for p, t, ok in ens.values()])
    ts = np.concatenate([t[~ok] for p, t, ok in ens.values()])
    sq = (ps - ts) ** 2
    order = np.argsort(sq)[::-1]
    frac = np.array([0.001, 0.005, 0.01, 0.05, 0.10])
    share = [sq[order[:max(1, int(f * len(sq)))]].sum() / sq.sum() * 100 for f in frac]

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
    b = tidy(ax[0])
    xs = [f * 100 for f in frac]
    b.bar(xs, share, width=[f * 45 for f in frac], color=ACCENT, edgecolor='white', lw=0.8)
    for x, y in zip(xs, share):
        b.text(x, y + 1.2, f'{y:.0f}%', ha='center', fontsize=8.5, color=INK)
    b.set_xlabel('worst share of steps (%)', fontsize=9)
    b.set_ylabel('share of squared error (%)', fontsize=9)
    b.set_title('A few steps carry most of the error', fontsize=10.5, color=INK)
    b.set_ylim(0, 80)

    a = tidy(ax[1])
    edges = [0, 40, 70, 100, 140, 180, 220, 260, 320, 2000]
    ctr, dif = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ps >= lo) & (ps < hi)
        if m.sum() >= 50:
            ctr.append(ps[m].mean()); dif.append(ts[m].mean() - ps[m].mean())
    a.axhline(0, color=GREY, lw=1, ls='--')
    a.plot(ctr, dif, marker='o', color=WARM, lw=1.6, markersize=5)
    a.set_xlabel('mean prediction in the bin', fontsize=9)
    a.set_ylabel('true mean $-$ predicted mean', fontsize=9)
    a.set_title('The forecast is already calibrated', fontsize=10.5, color=INK)
    a.text(0.02, 0.94, 'on the line = calibrated', transform=a.transAxes,
           fontsize=8.5, color=GREY, va='top')

    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_calibration.pdf', bbox_inches='tight')
    plt.close(fig)
def fig_shift():
    X = pd.read_csv(DATA + '/optional_external_data.csv')
    X.columns = list('tABCDEFGHIJ')
    X = X.apply(pd.to_numeric, errors='coerce')
    st = np.argmax(np.stack([X[c].values.astype(float) for c in 'GHIJ']), 0)
    hid = float((st[-168:] == 1).mean())

    B = pd.read_csv(os.path.join(HERE, 'diagnostics', 'level_blocks.csv'))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
    a = tidy(ax[0])
    a.scatter(B.f1, B.bias, s=14, color=ACCENT, alpha=0.55, linewidths=0)
    a.axhline(0, color=GREY, lw=1, ls='--')
    a.axhline(B.bias.mean(), color=WARM, lw=1.4)
    a.axvline(hid, color=WARM, lw=1.2, ls=':')
    a.text(hid, 0.96, ' hidden week', transform=a.get_xaxis_transform(),
           fontsize=8.5, color=WARM, va='top')
    a.set_xlabel('state-1 share of the week (how shifted it is)', fontsize=9)
    a.set_ylabel('predicted week mean $-$ true mean', fontsize=9)
    a.set_title('The level error does not grow with the shift', fontsize=10.5, color=INK)

    b = tidy(ax[1])
    bins = [(-.01, .30), (.30, .40), (.40, .45), (.45, .50), (.50, 1.01)]
    xs, rm, nn = [], [], []
    for lo, hi in bins:
        m = (B.f1 >= lo) & (B.f1 < hi)
        if m.sum() >= 3:
            xs.append(f'{lo:.2f}'); rm.append(B[m].rmse.mean()); nn.append(m.sum())
    b.bar(range(len(xs)), rm, color=[WARM if lo >= .50 else ACCENT for lo, hi in bins[:len(xs)]],
          edgecolor='white', lw=0.8, width=0.66)
    for i, (r, k) in enumerate(zip(rm, nn)):
        b.text(i, r + 1.0, f'{r:.0f}\n(n={k})', ha='center', fontsize=8, color=INK)
    b.set_xticks(range(len(xs)))
    b.set_xticklabels(xs, fontsize=8.5)
    b.set_xlabel('state-1 share of the week (bin)', fontsize=9)
    b.set_ylabel('mean block RMSE', fontsize=9)
    b.set_title('Shifted weeks are not harder', fontsize=10.5, color=INK)
    b.set_ylim(0, max(rm) * 1.34)

    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_shift.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote fig_shift.pdf')


def fig_forecast():
    y = pd.read_csv(DATA + '/student_train.csv').value.values
    m = np.load(os.path.join(HERE, 'final_members.npy'))
    sub = np.clip(m.mean(0), 0, None)
    fig, axt = plt.subplots(figsize=(9.6, 3.4))
    ax = tidy(axt)
    ax.plot(np.arange(len(y))[-336:], y[-336:], color=INK, lw=1.0, label='observed')
    tf = np.arange(len(y), len(y) + 168)
    ax.fill_between(tf, m.min(0), m.max(0), color=ACCENT, alpha=0.18, lw=0,
                    label='range of the 20 members')
    ax.plot(tf, sub, color=ACCENT, lw=1.7, label='submitted forecast (their mean)')
    ax.axvline(len(y) - 0.5, color=GREY, lw=1, ls=':')
    ax.set_xlabel('step', fontsize=9)
    ax.set_ylabel('value', fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_forecast.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote fig_forecast.pdf')


if __name__ == '__main__':
    fig_calibration(load_ens())
    for fn in (fig_shift, fig_forecast):
        try:
            fn()
        except Exception as e:
            print(f'skip {fn.__name__}: {e}')
    print('wrote fig_calibration.pdf')