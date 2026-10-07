from collections import defaultdict
from pathlib import Path

import wandb


PROJECT = "kesavanramakrishnan/cs312-assignments"
OUTPUT_PATH = Path(__file__).parents[1] / "notes" / "a1" / "run_index.md"

GROUPS = (
    ("1a", "Problem 1(a): one-variable sweeps"),
    ("1b", "Problem 1(b): hyperparameter interactions"),
    ("1c", "Problem 1(c): learning-rate schedules"),
    ("2a", "Problem 2(a): model-size ladders"),
    ("2b", "Problem 2(b): alternate scaling axes"),
    ("2c", "Problem 2(c): scaling-law failures"),
    ("3a", "Problem 3(a): joint variation"),
    ("3ref", "Problem 3: deterministic references"),
    ("3b", "Problem 3(b): isolated variation"),
    ("3c", "Problem 3(c): variability under interventions"),
    ("3quiz", "Problem 3: additional seed sweeps"),
    ("4a", "Problem 4(a): one-token perturbation"),
    ("4b", "Problem 4(b): perturbation time and magnitude"),
    ("5", "Problem 5: diagnostic loss curves"),
)
GROUP_TITLES = dict(GROUPS)


def group_for_run(run):
    tags = set(run.tags)
    if "p1a" in tags:
        return "1a"
    if "1b" in tags:
        return "1b"
    if "1c" in tags:
        return "1c"
    if "2a" in tags:
        return "2a"
    if "2b" in tags:
        return "2b"
    if "2c" in tags:
        return "2c"
    if "3a" in tags:
        return "3a"
    if "3-ref" in tags:
        return "3ref"
    if "3b" in tags:
        return "3b"
    if "3c-quiz" in tags:
        return "3quiz"
    if "3c" in tags:
        return "3c"
    if "4a" in tags:
        return "4a"
    if "4b" in tags:
        return "4b"
    if "5-quiz" in tags:
        return "5"
    return None


def format_float(value):
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.6g}"
    return f"{value:g}"


def format_tokens(tokens):
    if tokens >= 1_000_000_000:
        return f"{tokens / 1_000_000_000:.3g}B"
    return f"{tokens / 1_000_000:.3g}M"


def model_description(config):
    model_config = config.get("model_config") or {}
    depth = model_config.get("num_hidden_layers")
    width = model_config.get("hidden_size")
    if depth == 8 and width == 512:
        return None
    if depth is None or width is None:
        return config.get("model_name")
    return f"{depth} layers, width {width}"


def schedule_description(schedule):
    if schedule.startswith("wsd"):
        fraction = float(schedule.removeprefix("wsd"))
        return f"warmup-stable-decay ({fraction:.0%} decay)"
    if schedule == "cos":
        return "cosine schedule"
    return f"{schedule} schedule"


def configuration_changes(run):
    config = run.config
    changes = []

    model = model_description(config)
    if model:
        changes.append(model)

    learning_rate = config.get("learning_rate")
    if learning_rate != 0.003:
        changes.append(f"learning rate {format_float(learning_rate)}")

    schedule = config.get("lr_schedule")
    if schedule != "linear":
        changes.append(schedule_description(schedule))

    warmup = config.get("warmup_percent")
    if warmup != 0.01:
        changes.append(f"warmup {warmup:.1%}")

    batch_size = config.get("batch_size")
    if batch_size != 64:
        changes.append(f"batch {batch_size}")

    weight_decay = config.get("weight_decay")
    if weight_decay != 0.1:
        changes.append(f"weight decay {format_float(weight_decay)}")

    dropout = config.get("dropout")
    if dropout:
        changes.append(f"dropout {format_float(dropout)}")

    sequences = config.get("num_train_sequences")
    epochs = config.get("num_epochs", 1)
    if sequences != 600_000 or epochs != 1:
        tokens = int(sequences * 1024 * epochs)
        data = f"{sequences:,} sequences"
        if epochs != 1:
            data += f" × {format_float(epochs)} epochs"
        changes.append(f"{data} ({format_tokens(tokens)} tokens processed)")

    model_seed = config.get("model_seed")
    data_seed = config.get("data_seed")
    if model_seed != 42 or data_seed != 42:
        if model_seed == data_seed:
            changes.append(f"model/data seed {model_seed}")
        else:
            changes.append(f"model seed {model_seed}, data seed {data_seed}")

    if config.get("deterministic"):
        changes.append("deterministic execution")

    if config.get("perturb_one_token"):
        changes.append("one training token changed")

    perturb_step = config.get("perturb_step")
    perturb_scale = config.get("perturb_scale")
    if perturb_step is not None:
        changes.append(
            f"weight perturbation {format_float(perturb_scale)} at step {perturb_step}"
        )

    if "a100" in run.name:
        changes.append("A100")
    elif "3c" in run.tags or "4b" in run.tags:
        changes.append("B200")

    return "; ".join(changes) if changes else "standard recipe"


def render_index(runs):
    grouped = defaultdict(list)
    for run in runs:
        group = group_for_run(run)
        if group is not None:
            grouped[group].append(run)

    completed_count = sum(len(grouped[key]) for key, _ in GROUPS)
    lines = [
        "# Assignment 1 run index",
        "",
        (
            f"This appendix lists {completed_count} completed training runs with a recorded "
            "terminal validation loss. Interrupted runs and duplicate records without terminal "
            "metrics are excluded. A run reused in a later analysis appears only once."
        ),
        "",
        (
            "Unless a row says otherwise, the configuration is d8 (8 layers, width 512), "
            "614.4M fresh tokens, AdamW, learning rate 0.003, 1% warmup, linear decay, "
            "batch size 64, weight decay 0.1, no dropout, and model/data seed 42."
        ),
        "",
    ]

    for group, title in GROUPS:
        group_runs = sorted(grouped[group], key=lambda run: run.created_at)
        if not group_runs:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                "| ID | Changes from the standard recipe | Final train loss | Final validation loss |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for index, run in enumerate(group_runs, start=1):
            train_loss = run.summary.get("train_loss")
            val_loss = run.summary["val_loss"]
            train_text = "—" if train_loss is None else f"{train_loss:.4f}"
            lines.append(
                f"| {group.upper()}-{index:02d} | {configuration_changes(run)} "
                f"| {train_text} | {val_loss:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    api = wandb.Api(timeout=60)
    runs = [
        run
        for run in api.runs(PROJECT, per_page=300)
        if run.state == "finished" and run.summary.get("val_loss") is not None
    ]
    OUTPUT_PATH.write_text(render_index(runs) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
