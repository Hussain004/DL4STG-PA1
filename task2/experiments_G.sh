cd "$(dirname "$0")"
export Q2_DATA="${Q2_DATA:-../Data/}"
export Q2_OUT="${Q2_OUT:-runs}"
export Q2_THREADS=5
L="${Q2_LOG:-logs}"; mkdir -p "$L"
# the reference configuration: C4_l2k97, 49.85 seed-ensembled, 51.91 single-seed
R="win=6,24,168 gsum=1 bridge=1,2,6,12,24,48,168 bex=3 d=32 drop=0.1 lr=1e-3 wd=1e-4 bs=128 ow=2 decomp=1 mixer=autocorr arch=enc clamp=1 layers=2 dk=97"

# G1 = rank-gauss + spike indicators   G2 = rank-gauss only   G3 = spike indicators only
python3 train.py G1_rg_ind 0,1,2 0,1,2,3,4 15 4 spec=g  $R > $L/G1.log 2>&1 &
python3 train.py G2_rg     0,1,2 0,1,2,3,4 15 4 spec=gr $R > $L/G2.log 2>&1 &
python3 train.py G3_ind    0,1,2 0,1,2,3,4 15 4 spec=vi $R > $L/G3.log 2>&1 &
wait
echo ROUND_G_DONE