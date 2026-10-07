"""P1(d) Hyperball target runs at 1.2288B tokens."""
import argparse

from experiments.a2.modal_launcher import config, launch_training_jobs


EXPERIMENT_KEY = 'a2-hyperball-target'
TARGET_TOKENS = 1_228_800_000
PREDICTED_LR = .0114
# The 614.4M source fit window, so the target bracket matches the largest source budget.
NEARBY_LRS = (.006, .01, .015)


def runs():
    return [config(tokens=TARGET_TOKENS, optimizer_name='adamh', learning_rate=lr,
                   weight_decay=0., run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,))
            for lr in sorted({PREDICTED_LR, *NEARBY_LRS})]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Submit the runs to Modal.')
    args = parser.parse_args(argv)
    configs = runs()
    for c in configs:
        print(f'tokens={TARGET_TOKENS} optimizer={c.optimizer_name} lr={c.learning_rate:g} '
              f'wd={c.weight_decay:g} builder={c.optimizer_builder}')
    if args.execute:
        launch_training_jobs(configs)
    else:
        print('Preview only. Add --execute to train these configurations.')


if __name__ == '__main__':
    main()
