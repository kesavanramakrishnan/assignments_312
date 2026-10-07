import sys

from modal_train import launch_training_jobs
from train import TrainConfig, training_run_name


MAX_PARALLEL_RUNS = 10

RUNS = [
    TrainConfig(
        learning_rate=0.0003,
        run_name_suffix="5quiz",
        wandb_tags=("5-quiz",),
    ),
]


# TODO: Add additional runs that test the loss-curve factors you want to study.


def quiz_runs():
    return list(RUNS)


PART_BUILDERS = {
    "quiz": quiz_runs,
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = "--dry-run" in args
    parts = [arg for arg in args if arg != "--dry-run"]
    if len(parts) != 1 or parts[0] not in PART_BUILDERS:
        valid = ", ".join(PART_BUILDERS)
        print(
            "Usage: python -m experiments.a1_basics.p5_loss_curve_augury "
            f"<{valid}> [--dry-run]"
        )
        return 2

    runs = PART_BUILDERS[parts[0]]()
    print(f"{len(runs)} runs")
    for index, config in enumerate(runs):
        tags = ",".join(config.wandb_tags) if config.wandb_tags else "(none)"
        print(f"{index:02d} tags={tags} {training_run_name(config)}")
    if dry_run:
        return 0

    launch_training_jobs(runs, max_parallel_runs=MAX_PARALLEL_RUNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
