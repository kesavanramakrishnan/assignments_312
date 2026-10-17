"""Problem 2: schedules, parameter averaging, rewarming, and horizon-free schedules.

    uv run python -m experiments.a3_optimization.p2_river_valley
"""

from modal_train import launch_training_jobs
from train import TrainConfig


EXPERIMENT_KEY = "a3-river-valley"
# Each run logs val_ema{decay}_loss next to val_loss at every evaluation.
EMA_DECAYS = (0.9, 0.99, 0.999, 0.9995)

# Parts (a) and (b): one run per schedule, with evaluation-only EMAs.
SCHEDULE_RUNS = [
    TrainConfig(lr_schedule=lr_schedule, ema_decays=EMA_DECAYS, wandb_tags=("a3", EXPERIMENT_KEY))
    for lr_schedule in (
        "wsd0.0",  # constant after warmup
        "linear",
        "cos",
        "wsd0.2",  # plateau, then decay over the last 20% of training
    )
]

# Part (c): the LR falls to 0.1x between 40% and 45% of training and is back at
# full LR by 50%, before the same 20% cooldown as the wsd0.2 control above.
REWARM_RUNS = [
    TrainConfig(lr_schedule="wsd0.2", lr_dip=(0.4, 0.45, 0.5, 0.1), wandb_tags=("a3", EXPERIMENT_KEY)),
]

# Part (d): horizon-free schedules. warmup_steps fixes the warmup in updates,
# so it does not depend on the budget; sgdrT restarts every T updates.
HORIZON_FREE_RUNS = [
    TrainConfig(lr_schedule="invsqrt", warmup_steps=94, wandb_tags=("a3", EXPERIMENT_KEY)),
    TrainConfig(lr_schedule="sgdr2000", warmup_steps=94, min_lr_ratio=0.1,
                wandb_tags=("a3", EXPERIMENT_KEY)),
]

TAGS = ("a3", EXPERIMENT_KEY)

# Part (a): peak-LR robustness check for every schedule (default is 3e-3).
TUNE_RUNS = [
    TrainConfig(lr_schedule=lr_schedule, learning_rate=lr, ema_decays=EMA_DECAYS, wandb_tags=TAGS)
    for lr in (1.5e-3, 6e-3)
    for lr_schedule in ("wsd0.0", "linear", "cos", "wsd0.2")
]

# Part (c): vary the dip depth, duration and timing around the (0.4, 0.45, 0.5, 0.1)
# control (the wsd0.2 baseline is the no-dip control; all share the 20% cooldown).
DIP_RUNS = [
    TrainConfig(lr_schedule="wsd0.2", lr_dip=dip, wandb_tags=TAGS)
    for dip in (
        (0.4, 0.45, 0.5, 0.5),    # shallow
        (0.4, 0.45, 0.5, 0.01),   # deep
        (0.3, 0.4, 0.6, 0.1),     # long
        (0.1, 0.15, 0.2, 0.1),    # early
    )
]

# Part (d): budget-tuned linear schedules at 12.5%, 25%, 50%, 75% of the 600k-sequence run
# (76.8M, 153.6M, 307.2M, 460.8M tokens), to compare with the invsqrt / SGDR runs read at
# the same token counts.
BUDGET_RUNS = [
    TrainConfig(lr_schedule="linear", num_train_sequences=n, wandb_tags=TAGS)
    for n in (75_000, 150_000, 300_000, 450_000)
]

RUNS = SCHEDULE_RUNS + REWARM_RUNS + HORIZON_FREE_RUNS + TUNE_RUNS + DIP_RUNS + BUDGET_RUNS


# TODO: Part (b): try other EMA timescales.
# TODO: Part (c): vary the depth and duration of lr_dip.
# TODO: Part (d): run the horizon-free schedules to your largest budget, read
# them off at several smaller budgets, and train budget-tuned schedules
# (e.g. linear with a smaller num_train_sequences) at each budget.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
