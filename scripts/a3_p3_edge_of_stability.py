"""A3 P3: edge-of-stability plots and tables from the sharpness logged to W&B."""
import json
import os
import statistics
import sys
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization import p3_edge_of_stability as p3
from experiments.a3_optimization.sharpness import history
from train import checked_train_config, training_run_name


ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "notes" / "a3" / "p3"
CACHE = ROOT / "tmp" / "a3_p3_cache.json"
TOKENS_PER_UPDATE_64 = 64 * 1024
WARMUP_TOKENS = 6.2e6
WINDOW = (100e6, 190e6)  # tokens over which the late-training plateau is summarized


def fetch(refresh=False):
    """Fetch every run's history, saving after each run so a timeout does not lose earlier ones."""
    os.environ.setdefault("WANDB_HTTP_TIMEOUT", "180")
    cache = {} if refresh or not CACHE.exists() else json.loads(CACHE.read_text())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    for config in [*p3.FULL_BATCH_RUNS, p3.MINIBATCH_RUNS[0], *p3.EXPLORE_RUNS]:
        name = training_run_name(checked_train_config(config))
        if name in cache:
            continue
        for attempt in range(3):
            try:
                rows = history(config)
                break
            except Exception as error:  # transient W&B timeouts
                print("retry", name, repr(error)[:80], flush=True)
        else:
            raise RuntimeError(f"could not fetch {name}")
        cache[name] = dict(rows=list(rows), losses=rows.losses, batch_size=checked_train_config(config).batch_size,
                           lr=config.learning_rate, beta1=config.beta1, beta2=config.beta2)
        CACHE.write_text(json.dumps(cache))
        print("fetched", name, len(rows), flush=True)
    return cache


def load(refresh=False):
    return fetch(refresh)


def configs_by_name():
    out = {}
    for config in [*p3.FULL_BATCH_RUNS, p3.MINIBATCH_RUNS[0], *p3.EXPLORE_RUNS]:
        out[training_run_name(checked_train_config(config))] = config
    return out


def series(rows, key, x="step"):
    pts = [(r[x], r[key]) for r in rows if r.get(key) is not None]
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def full_batch(cache):
    names = {"base": "constant", "halve": "step750x0.5", "double": "step750x2"}
    return {label: next(v for k, v in cache.items() if k.endswith(suffix)) for label, suffix in names.items()}


def plot_full_batch(cache):
    runs = full_batch(cache)
    base = runs["base"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    steps, lam = series(base, "lambda_max")
    axes[0].plot(steps, lam, "o-", ms=3, color="C0")
    axes[0].axhline(2 / 0.01, color="k", ls="--", label="2/η = 200")
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("sharpness λ_max")
    axes[0].set_yscale("log")
    axes[0].legend()
    loss = [r["train_loss"] for r in base if r.get("lambda_max") is not None]
    axes[1].plot(loss, lam, "o-", ms=3, color="C0")
    axes[1].axhline(200, color="k", ls="--")
    axes[1].set_xlabel("training loss")
    axes[1].set_ylabel("sharpness λ_max")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.grid(alpha=.3)
    fig.savefig(OUTPUT_DIR / "p3a_clean_case.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    colors = {"base": "C0", "halve": "C1", "double": "C3"}
    labels = {"base": "η = 0.01", "halve": "η halved at 750", "double": "η doubled at 750"}
    for label, run in runs.items():
        steps, lam = series(run["rows"], "lambda_max")
        axes[0].plot(steps, lam, "o-", ms=2.5, color=colors[label], label=labels[label])
        steps, lr = series(run["rows"], "lr")
        axes[1].plot(steps, lam * lr, "o-", ms=2.5, color=colors[label], label=labels[label])
    for threshold, color in ((200, "C0"), (400, "C1"), (100, "C3")):
        axes[0].hlines(threshold, 750 if threshold != 200 else 0, 1500 if threshold != 200 else 1500, color=color, ls="--", lw=1)
    axes[1].axhline(2, color="k", ls="--")
    axes[0].axvline(750, color="gray", lw=.8)
    axes[0].set_ylabel("sharpness λ_max (dashed: 2/η_t)")
    axes[1].set_ylabel("η_t λ_max (dashed: 2)")
    for ax in axes:
        ax.set_xlabel("update")
        ax.set_yscale("log")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / "p3a_lr_change.png", dpi=150)
    plt.close(fig)


def window_stat(run, key="eta_lambda_pre", window=WINDOW, stat=statistics.median):
    per_update = run["batch_size"] * 1024
    values = [r[key] for r in run["rows"] if r.get(key) is not None and window[0] <= r["step"] * per_update < window[1]]
    return stat(values) if values else None


def plot_minibatch(cache):
    run = next(v for k, v in cache.items() if k.endswith("epochs0.32-tok614M-warmup0.03125-wsd0.0") and "bs" not in k
               and v["lr"] == 0.003 and v["beta1"] == 0.9 and v["beta2"] == 0.95 and "wd" not in k)
    steps, value = series(run["rows"], "eta_lambda_pre")
    tokens = steps * run["batch_size"] * 1024
    keep = tokens > WARMUP_TOKENS
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(tokens[keep] / 1e6, value[keep], "o-", ms=3)
    ax.axhline(38, color="k", ls="--", label="38 (full-batch Adam, β₁ = 0.9)")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("η_t λ_max^pre")
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "p3b_minibatch.png", dpi=150)
    plt.close(fig)


GROUPS = (
    ("Learning rate", lambda c: c.learning_rate, (0.001, 0.0015, 0.003, 0.006, 0.01), "η"),
    ("Batch size", lambda c: c.batch_size, (16, 32, 64, 128, 256), "B"),
    ("β₂", lambda c: c.beta2, (0.9, 0.95, 0.99, 0.999), "β₂"),
    ("β₁", lambda c: c.beta1, (0.0, 0.8, 0.9, 0.95), "β₁"),
    ("Weight decay", lambda c: c.weight_decay, (0.0, 0.1, 1.0), "λ"),
)
DEFAULTS = dict(learning_rate=0.003, batch_size=64, beta1=0.9, beta2=0.95, weight_decay=0.1, grad_norm=1.0, warmup_percent=0.03125)


def is_default_except(config, field):
    return all(getattr(config, k) == v for k, v in DEFAULTS.items() if k != field)


def plot_exploration(cache):
    configs = configs_by_name()
    mini = {n: c for n, c in configs.items() if c.num_epochs < 1}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, (title, getter, values, symbol) in zip(axes.flat, GROUPS):
        field = {"Learning rate": "learning_rate", "Batch size": "batch_size", "β₂": "beta2", "β₁": "beta1",
                 "Weight decay": "weight_decay"}[title]
        colors = plt.cm.viridis(np.linspace(0, .9, len(values)))
        for value, color in zip(values, colors):
            name = next((n for n, c in mini.items() if getter(c) == value and is_default_except(c, field)), None)
            if name is None or name not in cache:
                continue
            run = cache[name]
            steps, y = series(run["rows"], "eta_lambda_pre")
            tokens = steps * run["batch_size"] * 1024
            keep = tokens > WARMUP_TOKENS
            ax.plot(tokens[keep] / 1e6, y[keep], "-", color=color, label=f"{symbol} = {value:g}")
        ax.axhline(38, color="k", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("training tokens (M)")
        ax.set_ylabel("η_t λ_max^pre")
        ax.set_yscale("log")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    ax = axes.flat[5]
    for (label, name_filter), color in zip((("no clipping", lambda c: c.grad_norm is None),
                                            ("warmup 10%", lambda c: c.warmup_percent == 0.1),
                                            ("default", lambda c: all(getattr(c, k) == v for k, v in DEFAULTS.items()))),
                                           ("C1", "C2", "C0")):
        name = next((n for n, c in mini.items() if name_filter(c)), None)
        if name and name in cache:
            run = cache[name]
            steps, y = series(run["rows"], "eta_lambda_pre")
            tokens = steps * run["batch_size"] * 1024
            keep = tokens > WARMUP_TOKENS
            ax.plot(tokens[keep] / 1e6, y[keep], color=color, label=label)
    ax.axhline(38, color="k", ls="--", lw=1)
    ax.set_title("Clipping and warmup")
    ax.set_xlabel("training tokens (M)")
    ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / "p3c_exploration.png", dpi=150)
    plt.close(fig)


def print_tables(cache):
    from experiments.a3_optimization.sharpness import History, summarize

    for label, run in full_batch(cache).items():
        rows = History(run["rows"])
        rows.losses = run["losses"]
        print(label, summarize(rows))
        lam = lambda lo, hi: statistics.median([r["lambda_max"] for r in run["rows"] if r.get("lambda_max") and lo <= r["step"] <= hi])
        print("   median lambda_max 600-750:", lam(600, 750), " 1100-1500:", lam(1100, 1500))
    print()
    print("| Run | η | B | β₁ | β₂ | WD | median λ_max^pre | median η_t λ_max^pre (100–190M tokens) | final train loss |")
    for name, config in configs_by_name().items():
        if config.num_epochs >= 1 or name not in cache:
            continue
        run = cache[name]
        losses = run["losses"]
        print(f"| {name.split('tok614M-')[-1][:40]} | {config.learning_rate:g} | {config.batch_size} | {config.beta1:g} | "
              f"{config.beta2:g} | {config.weight_decay:g} | {window_stat(run, key='lambda_max_pre'):.0f} | {window_stat(run):.1f} | {losses[-1][1]:.3f} |")


def main():
    cache = load(refresh="--refresh" in sys.argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_full_batch(cache)
    plot_minibatch(cache)
    plot_exploration(cache)
    print_tables(cache)


if __name__ == "__main__":
    main()
