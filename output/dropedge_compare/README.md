# DropEdge Comparison Experiment

Compare three training settings on the macro pretraining graph to disentangle
the **regularization** effect from the **noise-removal** effect of edge
dropping.

## Setup

| Setting | Training graph | Eval graph |
|---|---|---|
| **Baseline** | Full graph | Full graph |
| **DropEdge** | Random 20% of edges dropped per epoch | Full graph |
| **Fixed Remove** | 20% of edges removed once before training | The reduced graph |

- Config: `configs/macro.yml`
- Drop rate: 0.2
- Repeats: 5 (seeds 18-22)
- Epochs: 30, early stop patience 5
- Hardware: NVIDIA RTX 3090 (gpu06)

## Results

| Method | Test AUC | Test AP | Val AUC |
|---|---|---|---|
| Baseline     | 0.8757 ± 0.0072 | 0.8201 ± 0.0128 | 0.8754 |
| DropEdge     | 0.8796 ± 0.0111 | 0.8304 ± 0.0153 | 0.8796 |
| Fixed Remove | 0.8734 ± 0.0125 | 0.8246 ± 0.0265 | 0.8746 |

## Interpretation

- DropEdge improves Test AUC by only **+0.0040** over Baseline, well within
  one standard deviation - **not statistically meaningful** at this drop rate.
- Fixed Remove changes Test AUC by **-0.0022** vs Baseline, also inside the
  noise band.
- Conclusion: at drop_rate = 0.2, the macro pretraining graph shows
  **neither strong overfitting to structure** (which DropEdge would alleviate)
  **nor a substantial amount of noisy edges** (which Fixed Remove would
  alleviate). The baseline appears to be already adequately regularized.

## Files

- `dropedge_compare_summary.csv` - mean / std per method
- `dropedge_compare_raw.csv` - per-run results (15 rows)
- `run.log` - full training log

## Reproducing

```bash
python exp_dropedge_compare.py     --config configs/macro.yml     --drop-rate 0.2     --repeats 5     --epochs 30     --gpu 0
```

## Possible Next Steps

1. Sweep `--drop-rate` over {0.05, 0.1, 0.3, 0.5} to find an effective regime.
2. Increase `--epochs` and `--patience` to confirm full convergence.
3. Report per-edge-type AUC instead of the weighted average - the effect may
   be diluted across heterogeneous link types.
4. Repeat with more seeds (e.g. 10) to tighten the standard deviation.
