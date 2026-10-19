# Assignment 3: Optimization 2

Handout: [worksheets/a3_optimization/assignment3.pdf](../../worksheets/a3_optimization/assignment3.pdf). Due Oct 20, 1 PM. Report limit: 20 pages.
Starters and launchers: [experiments/a3_optimization](../../experiments/a3_optimization/README.md).

## Workflow

1. Launch the runs for each part (starters in `experiments/a3_optimization`).
2. Run the analysis script in `scripts/a3_p*.py`; it writes the figures into the problem's folder.
3. Fill in the tables and the one-paragraph synthesis in the note.
4. `scripts/build_a3_submission.py` (to write) merges the four notes into `a3_submission.pdf`.

## Notes

| Problem | Note | Data source |
| --- | --- | --- |
| P1 Mode connectivity | [p1/p1_mode_connectivity.md](p1/p1_mode_connectivity.md) | `experiments/a3_optimization/results/a3-mode-connectivity.json`; W&B tag `a3-mode-connectivity` |
| P2 River valley | [p2/p2_river_valley.md](p2/p2_river_valley.md) | W&B tag `a3-river-valley` (`wandb_runs.load_runs`) |
| P3 Edge of stability | [p3/p3_edge_of_stability.md](p3/p3_edge_of_stability.md) | W&B tag `a3-edge-of-stability` (`sharpness.py`) |
| P4 Inside the Hessian | [p4/p4_inside_the_hessian.md](p4/p4_inside_the_hessian.md) | `experiments/a3_optimization/results/a3-inside-the-hessian/` (`hessian_jobs.load_result`) |

All training runs carry the W&B tags `a3` and the problem tag. Everything runs in the class workspace `cs312-kesavanr`.

## Run status

All runs are finished and analyzed.

| Problem | Runs | Analysis script |
| --- | --- | --- |
| P1 | baseline: prefix, 12 branches, 30-cell measurement; (d): 5 variants (LR 1e-3 and 6e-3, batch 16 and 128, cosine over 1,000 updates), each a prefix, 8 branches and a 20-cell measurement | `scripts/a3_p1_mode_connectivity.py` (`--variants` for part d) |
| P2 | 7 baselines, 8 peak-LR runs, 4 dip variants, 4 budget-tuned linear runs | `scripts/a3_p2_river_valley.py` |
| P3 | 3 full-batch runs, minibatch baseline, 18 exploration runs | `scripts/a3_p3_edge_of_stability.py` |
| P4 | (a) 13 checkpoints plus 13 random-partition baselines; (b) 6 rescalings and 3 `head_init_scale` runs; (c) 10 subspace jobs | `scripts/a3_p4_hessian.py` |

`scripts/build_a3_submission.py` merges the four notes into `a3_submission.pdf` (13 pages).
