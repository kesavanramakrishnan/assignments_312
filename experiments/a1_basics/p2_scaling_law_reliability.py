import sys
from dataclasses import replace

from modal_train import launch_training_jobs
from model_config import LMConfig, depth_model_config
from train import TrainConfig, training_run_name


RUNS = []
seen_configs = set()
for depth in range(4, 10):
    for learning_rate, lr_schedule, dropout in [
        (0.003, "linear", 0.0),
        (0.003, "constant", 0.0),
        (0.003, "linear", 0.2),
        (0.03, "linear", 0.0),
    ]:
        config_key = (depth, learning_rate, lr_schedule, dropout)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        RUNS.append(
            TrainConfig(
                model_config=depth_model_config(depth),
                learning_rate=learning_rate,
                lr_schedule=lr_schedule,
                dropout=dropout,
            )
        )


# TODO: Add your own slope-bending and scaling-law-breaking interventions for
# parts (b) and (c). The generated run name will encode the config changes.

MAX_PARALLEL_RUNS = 10
PILOT_MAX_DEPTH = 7


def _tagged(runs):
    return [replace(run, wandb_tags=("2a",)) for run in runs]


def pilot_runs():
    """d4–d7 only."""
    return _tagged(
        run
        for run in RUNS
        if run.model_config.num_hidden_layers <= PILOT_MAX_DEPTH
    )


def _already_measured(run):
    """Skip ladder cells whose recipe already finished under another name."""
    if run.model_config.num_hidden_layers != 8 or run.learning_rate != 0.003 or run.dropout != 0.0:
        return False
    # Problem 1 default run, and the part (c) constant run at the same recipe.
    return run.lr_schedule in {"linear", "constant"}


def heldout_runs():
    """d8 and d9, excluding recipes that already have a finished run."""
    return _tagged(
        run
        for run in RUNS
        if run.model_config.num_hidden_layers > PILOT_MAX_DEPTH and not _already_measured(run)
    )


def part_b_runs():
    """Data scaling at d8, and a cosine depth ladder.

    The d8 run at 600,000 sequences already exists, and so does d8 with the
    cosine schedule from part 1(c). Token counts double from 75k to 2.4M
    sequences. The cosine ladder uses the default 600,000 sequences.
    """
    runs = []
    for num_train_sequences in (75_000, 150_000, 300_000, 1_200_000, 2_400_000):
        runs.append(
            TrainConfig(
                model_config=depth_model_config(8),
                num_train_sequences=num_train_sequences,
                wandb_tags=("2b", "2b-tokens"),
            )
        )
    for depth in (4, 5, 6, 7, 9):
        runs.append(
            TrainConfig(
                model_config=depth_model_config(depth),
                lr_schedule="cos",
                wandb_tags=("2b", "2b-cosine"),
            )
        )
    return runs


def width_model_config(hidden_size, depth=8):
    """Fixed depth, width set by hidden size. Heads stay at head_dim 64."""
    return LMConfig(
        name=f"d{depth}w{hidden_size}",
        vocab_size=4096,
        context_length=1024,
        hidden_size=hidden_size,
        intermediate_size=int(hidden_size * 3.5),
        num_hidden_layers=depth,
        num_attention_heads=hidden_size // 64,
        num_key_value_heads=hidden_size // 64,
        head_dim=64,
    )


def part_c_runs():
    """Breaks: repeated data, a tiny token budget across depth, and width at fixed depth.

    d8 at 75k sequences is already in the part (b) token sweep. The default d8
    width, 512, is the usual baseline.
    """
    runs = []
    for num_epochs in (2, 4, 8, 16, 32):
        runs.append(
            TrainConfig(
                model_config=depth_model_config(8),
                num_train_sequences=75_000,
                num_epochs=num_epochs,
                wandb_tags=("2c", "2c-epochs"),
            )
        )
    for depth in (4, 5, 6, 7, 9):
        runs.append(
            TrainConfig(
                model_config=depth_model_config(depth),
                num_train_sequences=75_000,
                wandb_tags=("2c", "2c-small-data"),
            )
        )
    for hidden_size in (256, 320, 384, 448, 576, 768):
        runs.append(
            TrainConfig(
                model_config=width_model_config(hidden_size),
                wandb_tags=("2c", "2c-width"),
            )
        )
    return runs


def fixed_width_depth_runs():
    """Depth ladder at the d8 hidden size. The default d8 run is depth 8, width 512."""
    return [
        TrainConfig(
            model_config=width_model_config(512, depth=depth),
            wandb_tags=("2c", "2c-depth"),
        )
        for depth in (4, 5, 6, 7, 9, 12)
    ]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = "--dry-run" in args
    parts = [arg for arg in args if arg != "--dry-run"]
    commands = {
        "pilots": pilot_runs,
        "heldout": heldout_runs,
        "b": part_b_runs,
        "c": part_c_runs,
        "depth": fixed_width_depth_runs,
    }
    if parts not in (["pilots"], ["heldout"], ["b"], ["c"], ["depth"]):
        print(
            "Usage: python -m experiments.a1_basics.p2_scaling_law_reliability "
            "<pilots|heldout|b|c|depth> [--dry-run]"
        )
        return 2

    runs = commands[parts[0]]()
    print(f"{len(runs)} runs")
    for index, config in enumerate(runs):
        print(f"{index:02d} {training_run_name(config)}")
    if dry_run:
        return 0

    launch_training_jobs(runs, max_parallel_runs=MAX_PARALLEL_RUNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
