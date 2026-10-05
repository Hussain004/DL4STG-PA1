# AI651 Assignment 1, Task 2: Autoformer for the 168-step block

A small encoder-only Autoformer that estimates the target for the 168 hidden steps from the optional covariates over those steps
(see the report, Q2 section, for the reasoning and the experiments).

The model is a *nowcast*: the optional file covers the hidden week as well as the history, so the network reads the
covariate channels of the 168 steps it has to predict and emits one value per step. No decoder, no recurrence.

## Files

| file | what it does |
|---|---|
| `data.py` | loads the data, finds interpolated steps, builds the covariate channels (rate recovered from column D, harmonics, trailing causal means) |
| `autoformer_blocks.py` | series decomposition (moving average) and auto-correlation (FFT, top-k delays, delay band, learned temperature), the two Autoformer ingredients |
| `model.py` | the encoder-only network: embedding, 2-3 blocks (auto-correlation, decomposition, feed-forward, decomposition), slow-part read-out, window-summary read-out, per-step output |
| `train.py` | leave-one-year-out validation (`FOLDS` 0 to 4) and final training (`-1`); every run saves its per-step predictions |
| `final.py` | trains the ensemble on all five years from `final_spec.txt`, averages it, checks the 168 numbers, writes `submission.txt`, prints P and E |
| `final_spec.txt` | the four configurations of the submitted ensemble, each with its seed list |
| `encdec_model.py` | the encoder-**decoder** Autoformer that I built and tested; it is worse, and is kept because the report quotes its numbers |
| `ablate_yonly.py` | the "without the optional file" ablation (same Autoformer, only the past 168 steps of `y`, different read-out) |
| `pick_members.py` | greedy forward selection of ensemble members over saved predictions |
| `check_selection.py` | leaves one fold out, re-runs the whole selection on the other four, and scores on the held-out fold |
| `summarize.py` | per-seed, seed-ensemble and per-fold RMSE of finished runs |
| `report_figs.py` | builds the report's Task 2 figures (`fig_calibration`, `fig_shift`, `fig_forecast`) from the finished runs |
| `diagnostics/` | the analyses behind Sections 5.3–5.7: the shape-repeatability check, the covariate statistics, the paired comparison of two submissions, the calibration test, the covariate-shifted validation, and the attempt-5 ensemble choice |
| `experiments_A.sh` ... `experiments_G.sh` | every run behind the report's tables |
| `submission.txt` | the 168 submitted values |
| `submission_manifest.txt` | per-member parameter count and epochs, and the totals |
| `final_members.npy` | the 20 x 168 member forecasts behind `submission.txt` |
| `FINDINGS.md` | my working notes for this round, including the full experimental trail |

## How to run

```
pip install -r requirements.txt
export Q2_DATA=path/to/Data/          # the folder with student_train.csv and optional_external_data.csv
export Q2_OUT=runs

python autoformer_blocks.py && python model.py && python encdec_model.py    # self-tests
bash experiments_C.sh                  # the four ensemble members, 5 seeds x 5 folds
bash experiments_D.sh && python ablate_yonly.py ab_yonly 0,1,2 0,1,2,3,4 15 4
python summarize.py C1_l3k97 C2_l2k25 C3_e25 C4_l2k97
python final.py P3_final final_spec.txt      # writes submission.txt and the manifest
```

One 5-fold x 5-seed configuration takes about 3 minutes on an RTX 5060 Ti.

`report_figs.py` writes its three figures into `$Q2_FIG` (default `figures/`). Point it at the
report's own figure directory to overwrite them in place:

```bash
Q2_FIG=../figures python report_figs.py
```

## Declared at submission

**P = 692,565** trainable parameters (20 models: 40,941 / 32,524 / 32,524 / 32,524, five seeds each) and
**E = 350** epochs (three configurations at 15 epochs and one at 25, five seeds each). Both are reproduced by
`final.py`, which rebuilds every network and counts its parameters again, and are listed member by member in
`submission_manifest.txt`.

From the posted scores, the leaderboard's score adds about 2e-9 per parameter and 5e-4 per epoch, so these cost
0.0014 and 0.175 respectively - 0.176 in total, negligible next to the 63-120 RMSE spread on the board.