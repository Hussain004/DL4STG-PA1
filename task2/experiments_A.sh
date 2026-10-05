#!/bin/bash
# Round A: does the architecture change help? 5 folds x 3 seeds each, pooled RMSE.
# Baseline (encoder-only, linear window read-out) is the reference = 51.90 single-seed pooled.
cd "$(dirname "$0")"
export Q2_DATA="${Q2_DATA:-../Data/}"
export Q2_OUT="${Q2_OUT:-runs}"
export Q2_THREADS=4
L="${Q2_LOG:-logs}"; mkdir -p "$L"

B="spec=v1 win=6,24,168 decomp=1 dk=97 gsum=1 bridge=1,6,24,168 bex=3 d=32 layers=2 drop=0.1 lr=1e-3 wd=1e-4 bs=128"

# A1: non-linear window read-out on the encoder-only model (targets the level/threshold issue)
python3 train.py A1_gmlp32  0,1,2 0,1,2,3,4 15 4 $B arch=enc gmlp=32  > $L/A1.log 2>&1
# A2: encoder-decoder, 168-step look-back
python3 train.py A2_ed168   0,1,2 0,1,2,3,4 15 4 $B arch=encdec lb=168 elayers=2 dlayers=1 > $L/A2.log 2>&1
# A3: encoder-decoder with a shorter 336-step look-back is A2; instead vary depth
python3 train.py A3_ed_d48  0,1,2 0,1,2,3,4 15 4 $B arch=encdec lb=168 elayers=2 dlayers=1 d=48 > $L/A3.log 2>&1
echo ROUND_A_DONE