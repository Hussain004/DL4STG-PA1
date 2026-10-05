#!/bin/bash
# Round C: rebuild the four ensemble members that greedy selection picked out of the archived runs,
# in the current harness, and add fresh seeds.  Configurations are reconstructed from the sweep
# files that produced those runs (see sweeps/round*.txt in P2_cook).
cd "$(dirname "$0")"
export Q2_DATA="${Q2_DATA:-../Data/}"
export Q2_OUT="${Q2_OUT:-runs}"
export Q2_THREADS=4
L="${Q2_LOG:-logs}"; mkdir -p "$L"

# the rich-bridge channel set, spec=v0 (raw variables only) with the harmonics added by ych=1
R="win=6,24,168 gsum=1 bridge=1,2,6,12,24,48,168 bex=3 d=32 drop=0.1 lr=1e-3 wd=1e-4 bs=128 ow=2 decomp=1 mixer=autocorr gsum=1 arch=enc gmlp=0"

# C1 = g3_l3k97 : layers=3, dk=97, 15 epochs      (best single config found: 47.68)
python3 train.py C1_l3k97  0,1,2,3,4 0,1,2,3,4 15 4 spec=v1 $R layers=3 dk=97 > $L/C1.log 2>&1
# C2 = s3_v1    : layers=2, dk=25, 15 epochs
python3 train.py C2_l2k25  0,1,2,3,4 0,1,2,3,4 15 4 spec=v1 $R layers=2 dk=25 > $L/C2.log 2>&1
# C3 = s1_e25   : layers=2, dk=25, 25 epochs
python3 train.py C3_e25    0,1,2,3,4 0,1,2,3,4 25 4 spec=v1 $R layers=2 dk=25 > $L/C3.log 2>&1
# C4 = n1_dk97  : layers=2, dk=97, 15 epochs
python3 train.py C4_l2k97  0,1,2,3,4 0,1,2,3,4 15 4 spec=v1 $R layers=2 dk=97 > $L/C4.log 2>&1
echo ROUND_C_DONE