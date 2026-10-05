#!/bin/bash
# Round B: the encoder-decoder is unstable on fold 0 (the hard year).  Two hypotheses:
#   B1  the cross-attention decoder is simply the wrong inductive bias here (expect: no fix helps)
#   B2  the blow-up comes from the expm1 output link, which turns a large standardised z into an
#       explosive value; clamping z should tame it (expect: fold 0 stabilises)
# Also B3: the encoder-only model with a wider/deeper stack, as a control.
cd "$(dirname "$0")"
export Q2_DATA="${Q2_DATA:-../Data/}"
export Q2_OUT="${Q2_OUT:-runs}"
export Q2_THREADS=4
L="${Q2_LOG:-logs}"; mkdir -p "$L"
B="spec=v1 win=6,24,168 decomp=1 dk=97 gsum=1 bridge=1,6,24,168 bex=3 d=32 layers=2 drop=0.1 lr=1e-3 wd=1e-4 bs=128"

# B2: encoder-decoder + clamped output link (tames the expm1 explosion)
python3 train.py B2_ed_clamp 0,1,2 0,1,2,3,4 15 4 $B arch=encdec lb=168 elayers=2 dlayers=1 clamp=1 > $L/B2.log 2>&1
# B3: encoder-only, deeper, as an architecture-strength control
python3 train.py B3_enc_l3d48 0,1,2 0,1,2,3,4 15 4 $B arch=enc layers=3 d=48 > $L/B3.log 2>&1
# B4: encoder-only with the clamp too (isolates the clamp from the architecture)
python3 train.py B4_enc_clamp 0,1,2 0,1,2,3,4 15 4 $B arch=enc clamp=1 > $L/B4.log 2>&1
echo ROUND_B_DONE