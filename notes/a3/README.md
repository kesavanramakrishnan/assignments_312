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

| Problem | Done | Still to run |
| --- | --- | --- |
| P1 | prefix, 12 branches, measure stage (30 cells) | (d) variants: LR, batch size, schedule |
| P2 | 7 baselines (4 schedules, 1 dip, invsqrt, SGDR); launched: 8 LR-tuning, 4 dip variants, 4 budget-tuned linear | none planned |
| P3 | (a)i, (b); launched: (a)ii halve/double η, 18 exploration runs | none planned |
| P4 | (a) 13 checkpoints, (c) three arms; launched: 6 rescalings | (b) training change, (c) variants |
