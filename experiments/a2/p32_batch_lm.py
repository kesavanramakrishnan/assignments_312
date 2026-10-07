"""P3.2 language-model batch-size experiments at 614.4M tokens."""
import argparse
import time

import wandb

from experiments.a2.modal_launcher import config, launch_training_jobs


TOKENS = 614_400_000
MAX_PARALLEL_RUNS = 10
WANDB_PROJECT = 'kesavanramakrishnan/cs312-assignments'
PART_KEYS = {'a': 'a2-p32a', 'b': 'a2-p32b', 'c': 'a2-p32c'}

# B=64 at {.0015, .003, .006} is supplied as P1a, so only smaller batches are new.
PART_A_GRID = {
    8: (.00075, .0015, .003),
    16: (.00075, .0015, .003, .006),
    32: (.0015, .003, .006),
}

# Hypothesis (ii) sources: WD sweep at LR .0015. WD .1 comes from part (a), and B=64 at
# WD {.1, .2, .4} is supplied as P2a, where loss still falls at .4.
FIXED_LR = .0015
PART_B_SOURCE_WDS = {
    8: (.05, .2, .4, .8),
    16: (.2, .4, .8, 1.6),
    32: (.2, .4, .8, 1.6),
    64: (.8, 1.6),
}
# Targets bracket both hypotheses: an LR sweep at WD .1 and a WD sweep at LR .0015,
# sharing the (.0015, .1) run.
FIXED_WD = .1
TARGET_BATCHES = (128, 256)
TARGET_LRS = (.0015, .003, .006)
TARGET_WDS = (.4, .8, 1.6, 3.2)

# Momentum ablation around each batch's best measured (a)/(b) pair, which already has beta1=.9.
MOMENTUM_BATCHES = (8, 256)
MOMENTUM_BETA1S = (0., .5, .8, .95, .98)
SOURCE_POLL_SECONDS = 300


def run(part, batch, learning_rate, weight_decay, **overrides):
    key = PART_KEYS[part]
    return config(tokens=TOKENS, batch=batch, learning_rate=learning_rate, weight_decay=weight_decay,
                  run_name_suffix=key, wandb_tags=(key, f'bs{batch}'), **overrides)


def part_a():
    return [run('a', batch, lr, FIXED_WD) for batch, lrs in PART_A_GRID.items() for lr in lrs]


def part_b():
    sources = [run('b', batch, FIXED_LR, wd) for batch, wds in PART_B_SOURCE_WDS.items() for wd in wds]
    targets = [run('b', batch, lr, FIXED_WD) for batch in TARGET_BATCHES for lr in TARGET_LRS]
    targets += [run('b', batch, FIXED_LR, wd) for batch in TARGET_BATCHES for wd in TARGET_WDS]
    return sources + targets


def finished_source_runs():
    runs = wandb.Api().runs(WANDB_PROJECT, filters={
        'tags': {'$in': [PART_KEYS['a'], PART_KEYS['b']]}, 'state': 'finished'})
    return [r for r in runs if r.config['num_train_sequences'] * 1024 == TOKENS]


def wait_for_sources(timeout_hours):
    """Block until every part (a)/(b) run has finished, or until the deadline."""
    expected = len(part_a()) + len(part_b())
    deadline = time.time() + timeout_hours * 3600
    while (done := len(finished_source_runs())) < expected and time.time() < deadline:
        print(f'{time.strftime("%H:%M")}: {done}/{expected} source runs finished', flush=True)
        time.sleep(SOURCE_POLL_SECONDS)
    print(f'{time.strftime("%H:%M")}: proceeding with {len(finished_source_runs())}/{expected} finished', flush=True)


def best_measured_pair(runs, batch):
    best = min((r for r in runs if r.config['batch_size'] == batch), key=lambda r: r.summary['val_loss'])
    return best.config['learning_rate'], best.config['weight_decay']


def part_c():
    runs = finished_source_runs()
    configs = []
    for batch in MOMENTUM_BATCHES:
        lr, wd = best_measured_pair(runs, batch)
        print(f'B={batch}: best measured pair lr={lr:g} wd={wd:g}')
        configs += [run('c', batch, lr, wd, beta1=beta1) for beta1 in MOMENTUM_BETA1S]
    return configs


PARTS = {'a': part_a, 'b': part_b, 'c': part_c}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=PARTS, nargs='+', required=True)
    parser.add_argument('--wait-hours', type=float, default=0.,
                        help='Before building part (c), wait up to this long for the (a)/(b) runs.')
    parser.add_argument('--execute', action='store_true', help='Submit the runs to Modal.')
    args = parser.parse_args(argv)
    if args.wait_hours:
        wait_for_sources(args.wait_hours)
    configs = [c for part in args.part for c in PARTS[part]()]
    for c in configs:
        print(f'batch={c.batch_size} lr={c.learning_rate:g} wd={c.weight_decay:g} beta1={c.beta1:g} '
              f'sequences={c.num_train_sequences} microbatches={c.num_micro_batches}')
    if args.execute:
        launch_training_jobs(configs, max_parallel_runs=MAX_PARALLEL_RUNS)
    else:
        print('Preview only. Add --execute to train these configurations.')


if __name__ == '__main__':
    main()
