"""Assignment 1, problem 1: hyperparameter sweeps on the default d8 recipe.

Every run keeps d8 and 600,000 sequences (614.4M tokens), so parameter count
times token count stays fixed. When the sequence count is not divisible by the
batch size, the trainer drops the final partial batch (at most 192 sequences).

Grids use a factor of about 3 around each default. Batch size uses powers of
two, since that is the natural scale for tokens per step.

The constant schedule in this repo ignores warmup_percent, so warmup is not
crossed with it.

Launch one part at a time. Parallelism is capped at 10, the personal-workspace GPU limit.

    uv run python -m experiments.a1_basics.p1_hyperparameters a
    uv run python -m experiments.a1_basics.p1_hyperparameters b
    uv run python -m experiments.a1_basics.p1_hyperparameters c
    uv run python -m experiments.a1_basics.p1_hyperparameters a --dry-run
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from modal_train import launch_training_jobs
from train import TrainConfig, training_run_name


MAX_PARALLEL_RUNS = 10

LEARNING_RATES = (3.33e-4, 1e-3, 3e-3, 9e-3, 2.7e-2)
WARMUP_PERCENTS = (0.001, 0.003, 0.01, 0.03, 0.1)
BATCH_SIZES = (16, 32, 64, 128, 256)
WEIGHT_DECAYS = (0.01, 0.03, 0.1, 0.3, 1.0)

DEFAULT_LR = 3e-3
DEFAULT_WARMUP = 0.01
DEFAULT_BATCH_SIZE = 64
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_SCHEDULE = "linear"

OFF_DEFAULT_SCHEDULES = ("cos", "constant", "wsd0.2")


@dataclass(frozen=True)
class SweepPoint:
    tags: tuple[str, ...]
    learning_rate: float = DEFAULT_LR
    warmup_percent: float = DEFAULT_WARMUP
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    lr_schedule: str = DEFAULT_SCHEDULE

    def signature(self):
        return (
            self.learning_rate,
            self.warmup_percent,
            self.batch_size,
            self.weight_decay,
            self.lr_schedule,
        )


def _add(points, tags, **overrides):
    points.append(SweepPoint(tags=tuple(tags), **overrides))


def part_a_points():
    """One-at-a-time sweeps. The shared default is recorded once."""
    points = []
    for learning_rate in LEARNING_RATES:
        _add(points, ("p1a", "p1a-lr"), learning_rate=learning_rate)
    for warmup_percent in WARMUP_PERCENTS:
        _add(points, ("p1a", "p1a-warmup"), warmup_percent=warmup_percent)
    for batch_size in BATCH_SIZES:
        _add(points, ("p1a", "p1a-batch"), batch_size=batch_size)
    for weight_decay in WEIGHT_DECAYS:
        _add(points, ("p1a", "p1a-wd"), weight_decay=weight_decay)
    return points


def part_b_points():
    """Off-axis cells for lr with batch, warmup, and weight decay.

    Part (a) already runs every value of each knob at the default of the
    others, so these runs only fill the corners. The third knob stays at
    its default.
    """
    points = []
    for learning_rate in (1e-3, 9e-3, 2.7e-2):
        for batch_size in (32, 256):
            _add(
                points,
                ("1b", "1b-lr-batch"),
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
    for learning_rate in (9e-3, 2.7e-2):
        for warmup_percent in (0.001, 0.1):
            _add(
                points,
                ("1b", "1b-lr-warmup"),
                learning_rate=learning_rate,
                warmup_percent=warmup_percent,
            )
    for learning_rate in (1e-3, 9e-3, 2.7e-2):
        for weight_decay in (0.01, 1.0):
            _add(
                points,
                ("1b", "1b-lr-wd"),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
    return points


def easy_quiz_points():
    """Example question: learning rate 1e-3 and 9e-3 at batch 128.

    Part (b) crossed those learning rates with batch 32 and 256 only.
    """
    points = []
    for learning_rate in (1e-3, 9e-3):
        _add(
            points,
            ("1b", "1b-lr-batch"),
            learning_rate=learning_rate,
            batch_size=128,
        )
    return points


def part_c_points():
    """Schedule interactions with learning rate and weight decay.

    Linear at these settings already exists from parts (a) and (b). Constant
    ignores warmup_percent, so warmup is only crossed on wsd and linear in
    the quiz trio.
    """
    points = []
    for learning_rate in (1e-3, 3e-3, 9e-3):
        for lr_schedule in OFF_DEFAULT_SCHEDULES:
            _add(
                points,
                ("1c", "1c-lr-schedule"),
                learning_rate=learning_rate,
                lr_schedule=lr_schedule,
            )
    for lr_schedule in ("constant", "wsd0.2"):
        _add(
            points,
            ("1c", "1c-lr-schedule"),
            learning_rate=2.7e-2,
            lr_schedule=lr_schedule,
        )
    for weight_decay in (0.3, 1.0):
        for lr_schedule in ("constant", "wsd0.2"):
            _add(
                points,
                ("1c", "1c-wd-schedule"),
                weight_decay=weight_decay,
                lr_schedule=lr_schedule,
            )
    for lr_schedule in ("wsd0.1", "wsd0.4"):
        _add(points, ("1c", "1c-wsd-fraction"), lr_schedule=lr_schedule)
    for lr_schedule, warmup_percent in (
        ("wsd0.2", 0.0),
        ("wsd0.2", 0.2),
        ("linear", 0.1),
    ):
        _add(
            points,
            ("1c", "1c-quiz"),
            learning_rate=0.009,
            weight_decay=1.0,
            lr_schedule=lr_schedule,
            warmup_percent=warmup_percent,
        )
    return points


PART_BUILDERS = {
    "a": part_a_points,
    "b": part_b_points,
    "c": part_c_points,
    "quiz": easy_quiz_points,
}


def _merge(points):
    merged = {}
    order = []
    for point in points:
        signature = point.signature()
        existing = merged.get(signature)
        if existing is None:
            merged[signature] = point
            order.append(signature)
            continue
        tags = tuple(dict.fromkeys((*existing.tags, *point.tags)))
        merged[signature] = SweepPoint(
            tags=tags,
            learning_rate=point.learning_rate,
            warmup_percent=point.warmup_percent,
            batch_size=point.batch_size,
            weight_decay=point.weight_decay,
            lr_schedule=point.lr_schedule,
        )
    return [merged[signature] for signature in order]


def catalog():
    points = []
    for builder in PART_BUILDERS.values():
        points.extend(builder())
    return _merge(points)


def points_for(parts):
    points = []
    for part in parts:
        points.extend(PART_BUILDERS[part]())
    return _merge(points)


def _run_name_suffix(tags):
    if "1b" in tags:
        return "1b"
    if "1c" in tags:
        return "1c"
    return None


def to_train_config(point):
    # An H100 OOMs on a compiled batch of 256 when validation copies logits to fp32.
    # Four microbatches keep the optimizer step at 256 sequences and the forward at 64.
    micro_batches = 4 if point.batch_size == 256 else 1
    return TrainConfig(
        learning_rate=point.learning_rate,
        warmup_percent=point.warmup_percent,
        batch_size=point.batch_size,
        weight_decay=point.weight_decay,
        lr_schedule=point.lr_schedule,
        num_micro_batches=micro_batches,
        run_name_suffix=_run_name_suffix(point.tags),
        wandb_tags=point.tags,
    )


def build_runs(parts):
    return [to_train_config(point) for point in points_for(parts)]


def _describe(runs):
    print(f"{len(runs)} runs")
    for index, config in enumerate(runs):
        print(
            f"{index:02d} lr={config.learning_rate:g} "
            f"warmup={config.warmup_percent:g} "
            f"bs={config.batch_size} "
            f"wd={config.weight_decay:g} "
            f"schedule={config.lr_schedule} "
            f"tags={','.join(config.wandb_tags)}"
        )
        print(f"   {training_run_name(config)}")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = "--dry-run" in args
    parts = [arg for arg in args if arg != "--dry-run"]
    if not parts or any(part not in PART_BUILDERS for part in parts):
        valid = ", ".join(PART_BUILDERS)
        print(f"Usage: python -m experiments.a1_basics.p1_hyperparameters <{valid}> [--dry-run]")
        return 2

    runs = build_runs(parts)
    _describe(runs)
    if dry_run:
        return 0

    launch_training_jobs(runs, max_parallel_runs=MAX_PARALLEL_RUNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
