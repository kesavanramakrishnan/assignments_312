import sys

from modal_train import TRAINING_GPU, launch_training_jobs
from train import TrainConfig, training_run_name


MAX_PARALLEL_RUNS = 10

RUNS = [
    TrainConfig(deterministic=True, wandb_tags=("4a",)),
    TrainConfig(deterministic=True, perturb_one_token=True, wandb_tags=("4a",)),
]


def part_a_runs():
    return list(RUNS)


def part_b_runs():
    """Weight noise at step T, then train to the end. Three scales, four times."""
    runs = []
    for perturb_step in (100, 2000, 5000, 8000):
        for perturb_scale in (1e-3, 1e-2, 1e-1):
            runs.append(
                TrainConfig(
                    deterministic=True,
                    perturb_step=perturb_step,
                    perturb_scale=perturb_scale,
                    run_name_suffix="4b",
                    wandb_tags=("4b",),
                )
            )
    return runs


PART_BUILDERS = {
    "a": part_a_runs,
    "b": part_b_runs,
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = "--dry-run" in args
    parts = [arg for arg in args if arg != "--dry-run"]
    if len(parts) != 1 or parts[0] not in PART_BUILDERS:
        valid = ", ".join(PART_BUILDERS)
        print(
            "Usage: python -m experiments.a1_basics.p4_amplification "
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

    launch_training_jobs(
        runs, gpu=TRAINING_GPU, max_parallel_runs=MAX_PARALLEL_RUNS
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
