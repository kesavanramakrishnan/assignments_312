"""Queue assignment runs whose configs are written in the handout.

The course account allows 2 concurrent GPUs. This waits until no detached
training app is left in cs312-kesavanr, then submits one app:

- problem 2, part a, depths d4–d7 only
- problem 3, the two deterministic reference runs from the starter
- problem 4, the unperturbed run and the one-token edit from the starter

Problem 1 parts b and c are not included. Those pairs were chosen here rather
than taken from the handout. Problem 2 depths d8 and d9 stay unlaunched until
predictions are written down. Problem 2 parts b and c, and any later seed or
perturbation sweep, are not included.
"""

from __future__ import annotations

import subprocess
import sys
import time

from experiments.a1_basics.p2_scaling_law_reliability import RUNS as P2_RUNS
from experiments.a1_basics.p3_measuring_variation import RUNS as P3_RUNS
from experiments.a1_basics.p4_amplification import RUNS as P4_RUNS
from modal_train import launch_training_jobs
from train import training_run_name


COURSE_ENVIRONMENT = "cs312-kesavanr"
MAX_PARALLEL_RUNS = 2
POLL_SECONDS = 60
SETTLE_SECONDS = 30


def prescribed_runs():
    runs = [
        run for run in P2_RUNS if run.model_config.num_hidden_layers <= 7
    ]
    runs.extend(P3_RUNS)
    runs.extend(P4_RUNS)
    return runs


def detached_training_running() -> bool:
    result = subprocess.run(
        ["uv", "run", "modal", "app", "list", "--env", COURSE_ENVIRONMENT],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return True
    return "detached" in result.stdout


def wait_until_gpus_free() -> None:
    while True:
        if detached_training_running():
            print(f"Course GPUs still busy; checking again in {POLL_SECONDS}s.")
            time.sleep(POLL_SECONDS)
            continue
        print(f"No detached training app. Rechecking in {SETTLE_SECONDS}s.")
        time.sleep(SETTLE_SECONDS)
        if not detached_training_running():
            return
        print("A training app appeared again; going back to waiting.")


def main(argv=None) -> int:
    dry_run = "--dry-run" in (sys.argv[1:] if argv is None else argv)
    runs = prescribed_runs()
    print(f"{len(runs)} prescribed configs")
    for index, config in enumerate(runs):
        print(f"{index:02d} {training_run_name(config)}")
    if dry_run:
        return 0
    wait_until_gpus_free()
    launch_training_jobs(runs, max_parallel_runs=MAX_PARALLEL_RUNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
