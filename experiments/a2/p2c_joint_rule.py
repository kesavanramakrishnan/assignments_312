"""P2(c): rescue a fixed peak LR of .003 at 2.4576B tokens by setting WD from the product law."""
import argparse

from experiments.a2.modal_launcher import config, launch_training_jobs


EXPERIMENT_KEY = 'a2-p2c'
TOKENS = 2_457_600_000
FIXED_LR = .003
# Product law fitted on the P2(a) optima at {.1536,.3072,.6144,1.2288}B (scripts/a2_p2_lr_wd.py):
# lr*wd* = 2.782e-4 at 2.4576B, so WD = 2.782e-4 / .003.
PREDICTED_WD = .0927
# Lowest measured val loss in the supplied 153.6M LR-WD sweep, reused without retuning.
SMALL_SCALE_BEST = dict(learning_rate=.0015, weight_decay=1.6)
# WD .1 at LR .003 and 2.4576B is supplied as P1b.
FIXED_LR_WDS = (.05, .2)
MAX_PARALLEL_RUNS = 10


def runs():
    pairs = [dict(learning_rate=FIXED_LR, weight_decay=PREDICTED_WD), SMALL_SCALE_BEST,
             *(dict(learning_rate=FIXED_LR, weight_decay=wd) for wd in FIXED_LR_WDS)]
    return [config(tokens=TOKENS, run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,), **pair)
            for pair in pairs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Submit the runs to Modal.')
    args = parser.parse_args(argv)
    configs = runs()
    for c in configs:
        print(f'tokens={c.num_train_sequences * 1024:,} lr={c.learning_rate:g} wd={c.weight_decay:g}')
    if args.execute:
        launch_training_jobs(configs, max_parallel_runs=MAX_PARALLEL_RUNS)
    else:
        print('Preview only. Add --execute to train these configurations.')


if __name__ == '__main__':
    main()
