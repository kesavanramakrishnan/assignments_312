import os
import subprocess
import sys

from modal_train import TRAINING_GPU, launch_training_jobs
from train import TrainConfig, training_run_name


RUNS = [
    TrainConfig(
        deterministic=True,
        run_name_suffix="reference-1",
        wandb_tags=("3-ref",),
    ),
    TrainConfig(
        deterministic=True,
        run_name_suffix="reference-2",
        wandb_tags=("3-ref",),
    ),
]


# TODO: Vary model_seed and data_seed separately and together.
# For hardware nondeterminism, run the same config on at least two GPU
# types by passing `gpu=...` to launch_training_jobs from a local experiment.

MAX_PARALLEL_RUNS = 10
A100_GPU = "A100-80GB"
H100_GPU = "H100"
B200_GPU = "B200"


def part_a_runs():
    """Three d8 runs with initialization, data order, and GPU noise all free.

    Two run on H100 and one on A100. Seeds differ on every run, and
    deterministic stays off.
    """
    runs = []
    for seed, gpu in ((1, H100_GPU), (2, H100_GPU), (3, A100_GPU)):
        suffix = "3a" if gpu == H100_GPU else "3a-a100"
        runs.append(
            (
                gpu,
                TrainConfig(
                    model_seed=seed,
                    data_seed=seed,
                    deterministic=False,
                    run_name_suffix=suffix,
                    wandb_tags=("3a",),
                ),
            )
        )
    return runs


def part_ref_runs():
    return [(H100_GPU, config) for config in RUNS]


def part_b_runs():
    """Isolate initialization, data order, and hardware nondeterminism."""
    runs = []
    for model_seed in (0, 1, 2):
        runs.append(
            (
                H100_GPU,
                TrainConfig(
                    deterministic=True,
                    data_seed=42,
                    model_seed=model_seed,
                    run_name_suffix="3b-init",
                    wandb_tags=("3b",),
                ),
            )
        )
    for data_seed in (0, 1, 2):
        runs.append(
            (
                H100_GPU,
                TrainConfig(
                    deterministic=True,
                    model_seed=42,
                    data_seed=data_seed,
                    run_name_suffix="3b-data",
                    wandb_tags=("3b",),
                ),
            )
        )
    runs.append(
        (
            A100_GPU,
            TrainConfig(
                deterministic=False,
                model_seed=42,
                data_seed=42,
                run_name_suffix="3b-a100",
                wandb_tags=("3b",),
            ),
        )
    )
    return runs


def part_c_runs():
    """Four settings, three joint seeds each.

    Dropout 0.2 and learning rate 2.7e-2 test whether a hyperparameter
    widens the seed spread. Built-in d12 and 2.4M sequences test whether
    a larger or longer run narrows it. Part (a) is the baseline spread.
    """
    settings = (
        {"dropout": 0.2},
        {"learning_rate": 2.7e-2},
        {"model_name": "d12"},
        {"num_train_sequences": 2_400_000},
    )
    runs = []
    for changes in settings:
        for seed in (0, 1, 2):
            runs.append(
                (
                    B200_GPU,
                    TrainConfig(
                        model_seed=seed,
                        data_seed=seed,
                        run_name_suffix="3c",
                        wandb_tags=("3c",),
                        **changes,
                    ),
                )
            )
    return runs


PART_BUILDERS = {
    "a": part_a_runs,
    "ref": part_ref_runs,
    "b": part_b_runs,
    "c": part_c_runs,
}


def _only_gpu(args):
    if "--gpu" not in args:
        return None
    index = args.index("--gpu")
    if index + 1 >= len(args):
        raise SystemExit("--gpu requires a GPU name")
    return args[index + 1]


def _launch_group(part, gpu, configs):
    if gpu == TRAINING_GPU:
        launch_training_jobs(configs, gpu=gpu, max_parallel_runs=MAX_PARALLEL_RUNS)
        return
    env = os.environ.copy()
    env["DL_ALCHEMY_MODAL_GPU"] = gpu
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.a1_basics.p3_measuring_variation",
            part,
            "--gpu",
            gpu,
        ],
        env=env,
        check=True,
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = "--dry-run" in args
    only_gpu = _only_gpu(args)
    skipped = {"--dry-run"}
    if only_gpu is not None:
        skipped.update({"--gpu", only_gpu})
    parts = [arg for arg in args if arg not in skipped]
    if len(parts) != 1 or parts[0] not in PART_BUILDERS:
        valid = ", ".join(PART_BUILDERS)
        print(
            "Usage: python -m experiments.a1_basics.p3_measuring_variation "
            f"<{valid}> [--dry-run] [--gpu GPU]"
        )
        return 2

    part = parts[0]
    runs = PART_BUILDERS[part]()
    if only_gpu is not None:
        runs = [(gpu, config) for gpu, config in runs if gpu == only_gpu]
        if not runs:
            print(f"No part ({part}) runs for gpu={only_gpu}")
            return 2
    print(f"{len(runs)} runs")
    print(f"function gpu={TRAINING_GPU}")
    for index, (gpu, config) in enumerate(runs):
        tags = ",".join(config.wandb_tags) if config.wandb_tags else "(none)"
        print(f"{index:02d} gpu={gpu} tags={tags} {training_run_name(config)}")
    if dry_run:
        return 0

    by_gpu = {}
    for gpu, config in runs:
        by_gpu.setdefault(gpu, []).append(config)
    for gpu, configs in by_gpu.items():
        if only_gpu is not None and gpu != TRAINING_GPU:
            raise RuntimeError(
                f"Refusing to launch {gpu} from a process whose function "
                f"was created with {TRAINING_GPU}. Set DL_ALCHEMY_MODAL_GPU={gpu} "
                "before starting Python."
            )
        _launch_group(part, gpu, configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
