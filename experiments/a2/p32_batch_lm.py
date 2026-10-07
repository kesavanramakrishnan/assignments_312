"""P3.2 language-model batch-size experiments at 614.4M tokens."""
import argparse

from experiments.a2.modal_launcher import config, launch_training_jobs


TOKENS = 614_400_000
MAX_PARALLEL_RUNS = 10
# B=64 at {.0015, .003, .006} is supplied as P1a, so only smaller batches are new.
PART_A_GRID = {
    8: (.00075, .0015, .003),
    16: (.0015, .003, .006),
    32: (.0015, .003, .006),
}


def part_a():
    key = 'a2-p32a'
    return [config(tokens=TOKENS, batch=batch, learning_rate=lr, weight_decay=.1,
                   run_name_suffix=key, wandb_tags=(key, f'bs{batch}'))
            for batch, lrs in PART_A_GRID.items() for lr in lrs]


PARTS = {'a': part_a}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=PARTS, required=True)
    parser.add_argument('--execute', action='store_true', help='Submit the runs to Modal.')
    args = parser.parse_args(argv)
    configs = PARTS[args.part]()
    for c in configs:
        print(f'batch={c.batch_size} lr={c.learning_rate:g} wd={c.weight_decay:g} '
              f'sequences={c.num_train_sequences} microbatches={c.num_micro_batches}')
    if args.execute:
        launch_training_jobs(configs, max_parallel_runs=MAX_PARALLEL_RUNS)
    else:
        print('Preview only. Add --execute to train these configurations.')


if __name__ == '__main__':
    main()
