"""How much of the leaderboard improvement is MODEL and how much is LUCK?

The previous submission (P2_final) and this one (P3_final) are both on disk as per-step CV
predictions, so the comparison can be made PAIRED, block by block, on all 258 held-out 168-blocks.
That separates the two things the leaderboard conflates:

    leaderboard delta  =  model improvement  +  re-draw luck on ONE block

If the paired mean difference is small but the paired spread is large, then most of a single
block's movement is luck, and the expected value of the next submission is governed by that spread
rather than by the mean.
"""
import glob
import os

import numpy as np

RUNS = os.environ.get('Q2_OUT', 'runs')
H = 168

# the OLD submission's four members (P2_cook/final.py SPECS, defaults bridge=1,6,24,168 bex=3)
OLD = ['r1', 'r2', 'r3', 'r4']
# the NEW submission's four members (P3_cook/code/P3_final_spec.txt)
NEW = ['C1_l3k97', 'C2_l2k25', 'C3_e25', 'C4_l2k97']


def ens_pred(names, root):
    """Per (fold, grid-length), the seed-ensembled prediction of every member listed.

    Keyed by (fold, length) because the archive contains runs of the same name made with
    different evs strides, so a fold can hold grids of different lengths.
    """
    per = {}
    for n in names:
        for f in sorted(glob.glob(f'{root}/{n}_f*_s*.npz')):
            d = np.load(f, allow_pickle=True)
            per.setdefault((int(d['fold']), d['pred'].shape[0]), []).append(d)
    out = {}
    for k, ds in per.items():
        out[k] = np.mean([x['pred'] for x in ds], 0)
    return out


def blocks(pred_by_fold, tgt_by_fold, ok_by_fold):
    rows = []
    for f in sorted(pred_by_fold):
        p = pred_by_fold[f]
        t, ok = tgt_by_fold[f], ok_by_fold[f]
        for i in range(len(p) // H):
            sl = slice(i * H, (i + 1) * H)
            if ok[sl].all():
                continue
            w = (~ok[sl]).astype(float)
            rows.append(dict(
                fold=f,
                rmse=float(np.sqrt(((p[sl] - t[sl]) ** 2 * w).sum() / w.sum())),
                pmean=float((p[sl] * w).sum() / w.sum()),
                tmean=float((t[sl] * w).sum() / w.sum()),
            ))
    return rows


def load_ref(root):
    ref = {}
    for f in sorted(glob.glob(f'{root}/{OLD[0]}_f*_s*.npz')):
        d = np.load(f, allow_pickle=True)
        ref[int(d['fold'])] = (d['tgt'], d['ok'])
    return ref


def main():
    old_root = os.environ.get('Q2_OLD', 'runs')
    old_e = ens_pred(OLD, old_root)
    new_e = ens_pred(NEW, RUNS)
    common = sorted(set(old_e) & set(new_e))
    print(f'{len(common)} (fold, grid) cells common to both ensembles: '
          f'{[(f, n) for f, n in common]}\n')

    # targets/ok come from any run with that (fold, length)
    tgt, ok = {}, {}
    for n in OLD + NEW:
        for f in sorted(glob.glob(f'{old_root}/{n}_f*_s*.npz')) + \
                   sorted(glob.glob(f'{RUNS}/{n}_f*_s*.npz')):
            z = np.load(f, allow_pickle=True)
            tgt[(int(z['fold']), z['pred'].shape[0])] = z['tgt']
            ok[(int(z['fold']), z['pred'].shape[0])] = z['ok']

    rows_old = blocks({k: old_e[k] for k in common}, tgt, ok)
    rows_new = blocks({k: new_e[k] for k in common}, tgt, ok)
    assert len(rows_old) == len(rows_new), (len(rows_old), len(rows_new))

    r_old = np.array([x['rmse'] for x in rows_old])
    r_new = np.array([x['rmse'] for x in rows_new])
    pm_o = np.array([x['pmean'] for x in rows_old])
    pm_n = np.array([x['pmean'] for x in rows_new])
    tm = np.array([x['tmean'] for x in rows_old])

    print(f'=== paired over {len(r_old)} held-out 168-blocks ===')
    print(f'  OLD ensemble (24 members, P=784k) pooled RMSE {np.sqrt((r_old**2).mean()):.2f}')
    print(f'  NEW ensemble (20 members, P=693k) pooled RMSE {np.sqrt((r_new**2).mean()):.2f}')
    d = r_old - r_new
    print(f'\n  mean paired improvement (old - new): {d.mean():+.2f}   median {np.median(d):+.2f}')
    print(f'  standard deviation of that difference: {d.std():.2f}')
    print(f'  -> the MODEL change is worth about {d.mean():.2f} RMSE on an average block.')

    print('\n=== so what does a RE-DRAW of one block look like? ===')
    # the leaderboard movement you actually observed
    observed = 76.1401 - 71.7090
    print(f'  observed leaderboard improvement: {observed:.2f} RMSE')
    pct = float(np.mean(d >= observed) * 100)
    print(f'  blocks where the new model beats the old by >= {observed:.2f}: '
          f'{np.mean(d >= observed)*100:.1f}%')
    print(f'  blocks where the new model is WORSE by >= {observed:.2f}: '
          f'{np.mean(d <= -observed)*100:.1f}%')
    print(f'  90th percentile of the improvement: {np.percentile(d, 90):+.2f}')
    print(f'  10th percentile of the improvement: {np.percentile(d, 10):+.2f}')

    print('\n=== how correlated are the two forecasts? ===')
    po = np.concatenate([old_e[f] for f in common])
    pn = np.concatenate([new_e[f] for f in common])
    print(f'  correlation {np.corrcoef(po, pn)[0,1]:.5f}   RMSE between them '
          f'{np.sqrt(((po-pn)**2).mean()):.2f}')

    print('\n=== level: did the level change? ===')
    print(f'  old mean predicted level {pm_o.mean():.1f}   new {pm_n.mean():.1f}   '
          f'truth {tm.mean():.1f}')
    print(f'  old level bias {(pm_o-tm).mean():+.2f}   new {(pm_n-tm).mean():+.2f}')

    print('\n=== VERDICT ===')
    print(f'  of the {observed:.2f} RMSE you gained, about {d.mean():.2f} is the model and the')
    print(f'  remaining ~{observed - d.mean():.2f} is which block you happened to score.')


if __name__ == '__main__':
    main()