# AI651 Assignment 1 (Fall 2026)

Muhammad Hussain Habib (27100016)

Deep Learning for Space, Time and Graphs. Task 1 is five written responses backed by an executed
notebook; Task 2 is a leaderboard challenge on a 168-step block.

| path | what it is |
|---|---|
| `report.pdf` | the report for both tasks — **start here** |
| `report.tex`, `tables/` | the LaTeX source of the report |
| `figures/` | every figure the report includes: `1.2-1.pdf`–`4.2-1.pdf` (Task 1 notebook outputs) and `fig_*.pdf` (Task 2) |
| `task1/` | the executed full-preset notebook `Assignment1.ipynb`, with `harness/` and `requirements.txt` beside it |
| `task2/` | the Task 2 code: the Autoformer, the validation harness, every experiment script, and `submission.txt` |

## Task 2 in one paragraph

The model is an encoder-only Autoformer in a nowcast form. The optional file covers the hidden week as well as the history, so the network reads the covariate channels of the 168 steps it must predict and emits one value per step so no decoder, no recurrence. Inside every encoder block sit the two Autoformer mechanisms: an auto-correlation mixer (FFT delay scores, top-*k* delays, learned softmax temperature) and a series decomposition (centred moving average), whose removed slow parts
are accumulated and read out rather than discarded. Submission: 20 models (4 configurations × 5 seeds) declaring P = 692,565 parameters and E = 350 epochs.

See `task2/README.md` for the file-by-file map and the exact commands.

## Reproducing Task 2

```bash
cd task2
pip install -r requirements.txt
export Q2_DATA=path/to/Data/       # folder with student_train.csv + optional_external_data.csv
export Q2_OUT=runs

python autoformer_blocks.py && python model.py && python encdec_model.py   # self-tests
bash experiments_C.sh                                                     # the 4 ensemble members
bash experiments_D.sh                                                     # the ablations
python ablate_yonly.py ab_yonly 0,1,2 0,1,2,3,4 15 4                     # no-optional-file ablation
python summarize.py C1_l3k97 C2_l2k25 C3_e25 C4_l2k97
python final.py P3_final final_spec.txt        # writes submission.txt + submission_manifest.txt
python report_figs.py                          # regenerates the three Task 2 figures
```

`runs/` (the per-step predictions of every training run) and the training logs are not committed; they are large and regenerable with the scripts above.
