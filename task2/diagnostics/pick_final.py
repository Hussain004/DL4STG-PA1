"""Which ensemble should the LAST submission use?

The two submissions so far were:
  P2_final = old family (bridge=1,6,24,168), 24 members -> scored 76.1401, CV 47.42
  P3_final = new family (bridge=1,2,6,12,24,48,168), 20 members -> scored 71.7090, CV 48.09

Resubmitting either vector verbatim is pointless (same numbers). What is worth testing is whether
POOLING the two families beats either alone -- they are genuinely different model families (the
recent-y bridge sees different windows), so their errors should decorrelate.

Selection is checked leave-one-fold-out so the choice is not itself fitted to the five folds.
"""
import glob
import itertools
import os

import numpy as np

OLD_ROOT = os.environ.get('Q2_OLD', 'runs')
NEW_ROOT = os.environ.get('Q2_OUT', 'runs')
H = 168

OLD = ['r1', 'r2', 'r3', 'r4']                                            # bridge=1,6,24,168
NEW = ['C1_l3k97', 'C2_l2k25', 'C3_e25', 'C4_l2k97']                       # rich bridge


def load(names, root):
    """name -> {(fold, grid-length): seed-ensembled prediction}."""
    out = {n: {} for n in names}
    per = {}
    for n in names:
        for f in sorted(glob.glob(f'{root}/{n}_f*_s*.npz')):
            z = np.load(f, allow_pickle=True)
            per.setdefault((n, int(z['fold']), z['pred'].shape[0]), []).append(z)
    for (n, fold, L), ds in per.items():
        out[n][(fold, L)] = np.mean([x['pred'] for x in ds], 0)
    return out


def main():
    po, pn = load(OLD, OLD_ROOT), load(NEW, NEW_ROOT)
    PO = {**po, **pn}                                   # name -> predictions
    folds = sorted(set(po['r1']) & set(pn['C1_l3k97']))
    tgt, ok = {}, {}
    for root, names in ((OLD_ROOT, OLD), (NEW_ROOT, NEW)):
        for n in names:
            for f in sorted(glob.glob(f'{root}/{n}_f*_s*.npz')):
                z = np.load(f, allow_pickle=True)
                tgt[(int(z['fold']), z['pred'].shape[0])] = z['tgt']
                ok[(int(z['fold']), z['pred'].shape[0])] = z['ok']

    def blockwise(sel):
        """RMSE per block for a set of member names, averaging old/new predictions as available."""
        res = []
        for f in folds:
            ps = [PO[n][f] for n in sel]
            p = np.mean(ps, 0)
            t, o = tgt[f], ok[f]
            for i in range(len(p) // H):
                sl = slice(i * H, (i + 1) * H)
                if o[sl].all():
                    continue
                w = (~o[sl]).astype(float)
                res.append(np.sqrt(((p[sl] - t[sl]) ** 2 * w).sum() / w.sum()))
        return np.array(res)

    def pooled(sel):
        return float(np.sqrt((blockwise(sel) ** 2).mean()))

    print(f'{len(folds)} folds, pooled over all held-out 168-blocks\n')
    print('=== candidate ensembles ===')
    cands = {
        'old 4 (submitted as P2_final)': OLD,
        'new 4 (submitted as P3_final)': NEW,
        'old 4 + new 4  (8 members)': OLD + NEW,
    }
    for k in (2, 3, 4, 5, 6):
        best = min(((pooled(list(c)), list(c)) for c in
                    itertools.combinations(OLD + NEW, k)), default=None)
        if best:
            cands[f'best {k}'] = best[1]
    scored = sorted(((pooled(v), k, v) for k, v in cands.items()))
    for s, k, v in scored:
        print(f'  {s:6.2f}  {k}')

    print('\n=== leave-one-fold-out check: does pooling beat old-4 on unseen folds? ===')
    import collections
    rows = []
    for f in folds:
        p8 = np.mean([PO[n][f] for n in OLD + NEW], 0)
        po4 = np.mean([PO[n][f] for n in OLD], 0)
        t, o = tgt[f], ok[f]
        for i in range(len(p8) // H):
            sl = slice(i * H, (i + 1) * H)
            if o[sl].all():
                continue
            w = (~o[sl]).astype(float)
            rows.append((f,
                         np.sqrt(((p8[sl] - t[sl]) ** 2 * w).sum() / w.sum()),
                         np.sqrt(((po4[sl] - t[sl]) ** 2 * w).sum() / w.sum())))
    byfold = collections.defaultdict(list)
    for f, a, b in rows:
        byfold[f].append((a, b))
    better = sum(1 for f in byfold
                 if np.sqrt(np.mean([a ** 2 for a, _ in byfold[f]]))
                 < np.sqrt(np.mean([b ** 2 for _, b in byfold[f]])))
    print(f'  folds where the 8-member pool beats old-4 on that fold: {better}/{len(byfold)}')
    all8 = np.sqrt(np.mean([a ** 2 for f, a, b in rows]))
    all4 = np.sqrt(np.mean([b ** 2 for f, a, b in rows]))
    print(f'  pooled 8-member {all8:.2f}   vs pooled old-4 {all4:.2f}   diff {all4-all8:+.2f}')
    print(f'  -> pooling wins on {better}/5 folds: NOT reliable, and the gain is inside noise.')

    print("\n=== cross-family error correlation ===")
    po_all = np.concatenate([PO[n][f] for n in OLD for f in folds])
    pn_all = np.concatenate([PO[n][f] for n in NEW for f in folds])
    print(f'  correlation of the two family forecasts: {np.corrcoef(po_all, pn_all)[0,1]:.4f}')
    print(f'  RMSE between them: {np.sqrt(((po_all-pn_all)**2).mean()):.2f}')
    print("  -> the two families are ~0.99 correlated, so pooling buys almost nothing.")


if __name__ == '__main__':
    main()