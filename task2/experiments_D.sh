#!/bin/bash
# Requirement 2.4 #4: the optional-file ablation, on the FINAL configuration (C4_l2k97), 3 seeds,
# 5 folds, same chronological split.  Plus the two Autoformer-mechanism ablations the manual asks
# for: point-wise attention instead of Auto-Correlation, and no Series Decomposition.
# These need the y-only read-out path, so they live in ablate_yonly.py / here rather than in the
# main sweep.
cd "$(dirname "$0")"
export Q2_DATA="${Q2_DATA:-../Data/}"
export Q2_OUT="${Q2_OUT:-runs}"
export Q2_THREADS=3
L="${Q2_LOG:-logs}"; mkdir -p "$L"
R="win=6,24,168 gsum=1 bridge=1,2,6,12,24,48,168 bex=3 d=32 drop=0.1 lr=1e-3 wd=1e-4 bs=128 ow=2 mixer=autocorr arch=enc clamp=1"

# full model (reference for this table)
python3 train.py ab_full   0,1,2 0,1,2,3,4 15 4 spec=v1 $R decomp=1 dk=97 layers=2 > $L/ab_full.log 2>&1
# mechanism ablation 1: point-wise attention instead of Auto-Correlation
python3 train.py ab_attn   0,1,2 0,1,2,3,4 15 4 spec=v1 $R decomp=1 dk=97 layers=2 mixer=attention > $L/ab_attn.log 2>&1
# mechanism ablation 2: no Series Decomposition
python3 train.py ab_nodec  0,1,2 0,1,2,3,4 15 4 spec=v1 $R decomp=0 dk=97 layers=2 > $L/ab_nodec.log 2>&1
echo ABLATION_DONE