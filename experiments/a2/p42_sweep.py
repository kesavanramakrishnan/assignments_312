"""P4.2: LR sweeps across width and depth at 153.6M tokens (course baseline vs muP, Depth-muP, CompleteP)."""
import argparse

from model_config import LMConfig
from experiments.a2.modal_launcher import config, launch_training_jobs


TOKENS = 153_600_000
REFERENCE_WIDTH = 512
REFERENCE_DEPTH = 8
MAX_PARALLEL_RUNS = 10
MODEL_BUILDER = 'experiments.a2.p42:ParameterizedLM'
OPTIMIZER_BUILDER = 'experiments.a2.p42:build_optimizer'
DEPTH_RULE = {'base': 'none', 'mup': 'none', 'depth-mup': 'depth-mup', 'completep': 'completep'}

# (a) widths 128 and 256: the transferred source LR is Problem 1's best sampled LR at 153.6M tokens, .003.
WIDTH_LRS = {'base': (.0015, .003, .006, .012, .024), 'mup': (.00075, .0015, .003, .006, .012)}
SMALL_WIDTHS = (128, 256)
# (c) held-out width 1024: local sweep (predicted-LR runs are added after the (b) fits).
LOCAL_SWEEP_LRS = (.00075, .0015, .003, .006)
HELD_OUT_WIDTH = 1024
# Recorded in notes/a2/p4 before any width-1024 result was inspected: power laws over the optimal base LRs at
# widths 128, 256 and 512. The transferred source LR (.003) is already in the local sweep.
PREDICTED_LRS = {'base': .00132, 'mup': .00194}
# (d) depths 4 and 16 at width 512 (depth 8 is the reference, where all depth rules coincide).
DEPTHS = (4, 16)
DEPTH_LRS = (.0015, .003, .006)
DEPTH_PRESCRIPTIONS = ('mup', 'depth-mup', 'completep')


def run(part, prescription, width, depth, lr):
    c = LMConfig(f'a2-w{width}-d{depth}', 4096, 1024, width, int(3.5 * width), depth, width // 64, width // 64)
    overrides = {}
    if prescription != 'base':
        kwargs = dict(mup=True, ref_width=REFERENCE_WIDTH, ref_depth=REFERENCE_DEPTH,
                      depth_rule=DEPTH_RULE[prescription])
        overrides = dict(model_builder=MODEL_BUILDER, model_builder_kwargs=kwargs,
                         optimizer_builder=OPTIMIZER_BUILDER, optimizer_kwargs=kwargs)
    return config(tokens=TOKENS, model_config=c, learning_rate=lr, run_name_suffix=f'p42-{prescription}',
                  wandb_tags=(f'a2-p42{part}', f'p42-{prescription}', f'w{width}', f'd{depth}'), **overrides)


def part_a():
    return [run('a', p, w, REFERENCE_DEPTH, lr) for w in SMALL_WIDTHS for p in WIDTH_LRS for lr in WIDTH_LRS[p]]


def part_c():
    return [run('c', p, HELD_OUT_WIDTH, REFERENCE_DEPTH, lr) for p in WIDTH_LRS for lr in LOCAL_SWEEP_LRS]


def part_c_predicted():
    return [run('c', p, HELD_OUT_WIDTH, REFERENCE_DEPTH, lr) for p, lr in PREDICTED_LRS.items()]


def part_d():
    return [run('d', p, REFERENCE_WIDTH, d, lr) for d in DEPTHS for p in DEPTH_PRESCRIPTIONS for lr in DEPTH_LRS]


PARTS = {'a': part_a, 'c': part_c, 'cp': part_c_predicted, 'd': part_d}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=PARTS, nargs='+', required=True)
    parser.add_argument('--execute', action='store_true', help='Submit the runs to Modal.')
    args = parser.parse_args(argv)
    configs = [c for part in args.part for c in PARTS[part]()]
    for c in configs:
        m = c.model_config
        print(f'{c.run_name_suffix:16s} width={m.hidden_size} depth={m.num_hidden_layers} lr={c.learning_rate:g} '
              f'wd={c.weight_decay:g}')
    print(f'{len(configs)} runs')
    if args.execute:
        launch_training_jobs(configs, max_parallel_runs=MAX_PARALLEL_RUNS)
    else:
        print('Preview only. Add --execute to train these configurations.')


if __name__ == '__main__':
    main()
