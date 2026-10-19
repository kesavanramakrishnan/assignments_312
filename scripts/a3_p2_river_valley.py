"""A3 P2: river-valley plots and tables from the W&B runs tagged a3-river-valley."""
import json
import sys
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import WANDB_ENTITY, WANDB_PROJECT


ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "notes" / "a3" / "p2"
CACHE = ROOT / "tmp" / "a3_p2_cache.json"
TAG = "a3-river-valley"
TOKENS_PER_STEP = 64 * 1024
EMA_DECAYS = (0.9, 0.99, 0.999, 0.9995)
EMA_KEYS = [f"val_ema{d:g}_loss" for d in EMA_DECAYS]
SCHEDULES = ("wsd0.0", "linear", "cos", "wsd0.2")
SCHEDULE_LABELS = {"wsd0.0": "constant", "linear": "linear", "cos": "cosine", "wsd0.2": "plateau + 20% decay"}
SCHEDULE_COLORS = {"wsd0.0": "C0", "linear": "C1", "cos": "C2", "wsd0.2": "C3"}
LRS = (1.5e-3, 3e-3, 6e-3)


def fetch():
    import wandb

    api = wandb.Api()
    latest = {}
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"tags": {"$in": [TAG]}}):
        if run.state != "finished":
            continue
        if run.name not in latest or run.created_at > latest[run.name].created_at:
            latest[run.name] = run
    runs = []
    for name, run in sorted(latest.items()):
        cfg = run.config
        train = [(r["optimizer_step"] + 1, r["train_loss"], r["learning_rate"])
                 for r in run.scan_history(keys=["optimizer_step", "train_loss", "learning_rate"])]
        val_keys = ["val_loss"] + [k for k in EMA_KEYS if k in run.summary]
        val = [[r["optimizer_step"] + 1] + [r[k] for k in val_keys]
               for r in run.scan_history(keys=["optimizer_step", *val_keys])]
        runs.append(dict(name=name, schedule=cfg["lr_schedule"], lr=cfg["learning_rate"],
                         dip=cfg.get("lr_dip"), num_train_sequences=cfg["num_train_sequences"],
                         ema=bool(cfg.get("ema_decays")), val_keys=val_keys,
                         train=sorted(train), val=sorted(val)))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(runs))
    return runs


def load(refresh=False):
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    return fetch()


def find(runs, schedule, lr=3e-3, dip=None, seqs=600_000):
    for r in runs:
        if (r["schedule"] == schedule and abs(r["lr"] - lr) < 1e-12 and r["dip"] == (list(dip) if dip else None)
                and r["num_train_sequences"] == seqs):
            return r
    raise KeyError((schedule, lr, dip, seqs))


def val_series(run, key="val_loss"):
    val = np.array(run["val"])
    column = 1 + (0 if key == "val_loss" else 1 + run["val_keys"].index(key) - 1)
    return val[:, 0], val[:, column]


def smooth(values, window=100):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def tokens_m(steps):
    return np.asarray(steps) * TOKENS_PER_STEP / 1e6


def regret(run, reference, key="val_loss"):
    steps, values = val_series(run, key)
    ref_steps, ref_values = val_series(reference)
    return steps, values - np.interp(steps, ref_steps, ref_values)


def reference_run(runs):
    full = [r for r in runs if r["num_train_sequences"] == 600_000 and r["dip"] is None and r["ema"]]
    return min(full, key=lambda r: r["val"][-1][1])


def plot_schedules(runs):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for schedule in SCHEDULES:
        run = find(runs, schedule)
        color, label = SCHEDULE_COLORS[schedule], SCHEDULE_LABELS[schedule]
        steps, _, lr = np.array(run["train"]).T
        axes[0].plot(tokens_m(steps), lr, color=color, label=label)
        window = 100
        axes[1].plot(tokens_m(steps[window - 1:]), smooth(np.array(run["train"])[:, 1], window), color=color, label=label)
        vs, vl = val_series(run)
        axes[2].plot(tokens_m(vs), vl, color=color, label=label)
    axes[0].set_ylabel("learning rate")
    axes[1].set_ylabel("training loss (100-step mean)")
    axes[2].set_ylabel("validation loss")
    axes[1].set_ylim(2.9, 3.8)
    axes[2].set_ylim(2.9, 3.8)
    for ax in axes:
        ax.set_xlabel("training tokens (M)")
        ax.grid(alpha=.3)
    axes[0].legend()
    fig.savefig(OUTPUT_DIR / "p2a_schedules.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for schedule in SCHEDULES:
        ax.plot(LRS, [find(runs, schedule, lr)["val"][-1][1] for lr in LRS], "o-", color=SCHEDULE_COLORS[schedule],
                label=SCHEDULE_LABELS[schedule])
    ax.set_xscale("log")
    ax.set_xticks(LRS, [f"{lr:g}" for lr in LRS])
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("final validation loss")
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "p2a_peak_lr.png", dpi=150)
    plt.close(fig)


def plot_ema(runs, reference):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, schedule in zip(axes, SCHEDULES):
        run = find(runs, schedule)
        steps, raw = val_series(run)
        ax.plot(tokens_m(steps), raw, color="black", lw=2, label="raw")
        for decay, key, color in zip(EMA_DECAYS, EMA_KEYS, plt.cm.viridis(np.linspace(0, .9, 4))):
            ax.plot(tokens_m(steps), val_series(run, key)[1], color=color, label=f"EMA {decay:g}")
        ax.plot(tokens_m(*[val_series(reference)[0]]), val_series(reference)[1], color="gray", ls=":", label="reference")
        ax.set_title(SCHEDULE_LABELS[schedule])
        ax.set_xlabel("training tokens (M)")
        ax.set_ylim(2.9, 3.6)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("validation loss")
    axes[0].legend(fontsize=6.5)
    fig.savefig(OUTPUT_DIR / "p2b_ema_val_loss.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (key, title) in zip(axes, (("val_loss", "Raw"), ("val_ema0.999_loss", "EMA 0.999"),
                                      ("val_ema0.9995_loss", "EMA 0.9995"))):
        for schedule in SCHEDULES:
            steps, reg = regret(find(runs, schedule), reference, key)
            ax.plot(tokens_m(steps), reg, color=SCHEDULE_COLORS[schedule], label=SCHEDULE_LABELS[schedule])
        ax.axhline(0, color="gray", lw=.8)
        ax.set_title(f"{title} regret vs reference")
        ax.set_xlabel("training tokens (M)")
        ax.set_ylim(-.1, .5)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("regret (validation loss)")
    axes[0].legend()
    fig.savefig(OUTPUT_DIR / "p2b_regret.png", dpi=150)
    plt.close(fig)


DIPS = ((0.1, 0.15, 0.2, 0.1), (0.4, 0.45, 0.5, 0.5), (0.4, 0.45, 0.5, 0.1), (0.4, 0.45, 0.5, 0.01), (0.3, 0.4, 0.6, 0.1))
DIP_LABELS = {(0.1, 0.15, 0.2, 0.1): "early (10–20%), ×0.1", (0.4, 0.45, 0.5, 0.5): "shallow, ×0.5",
              (0.4, 0.45, 0.5, 0.1): "control dip, ×0.1", (0.4, 0.45, 0.5, 0.01): "deep, ×0.01",
              (0.3, 0.4, 0.6, 0.1): "long (30–60%), ×0.1"}


def plot_dip(runs):
    control = find(runs, "wsd0.2")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for dip, color in zip(DIPS, plt.cm.plasma(np.linspace(0, .85, len(DIPS)))):
        run = find(runs, "wsd0.2", dip=dip)
        steps, reg = regret(run, control)
        axes[0].plot(tokens_m(steps), reg, color=color, label=DIP_LABELS[dip])
        steps, _, lr = np.array(run["train"]).T
        axes[1].plot(tokens_m(steps), lr, color=color, label=DIP_LABELS[dip])
    steps, _, lr = np.array(control["train"]).T
    axes[1].plot(tokens_m(steps), lr, color="black", ls=":", label="control (no dip)")
    axes[0].axhline(0, color="gray", lw=.8)
    axes[0].set_ylabel("regret vs no-dip control (validation loss)")
    axes[1].set_ylabel("learning rate")
    for ax in axes:
        ax.set_xlabel("training tokens (M)")
        ax.grid(alpha=.3)
    axes[0].set_ylim(-.25, .25)
    axes[0].legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / "p2c_dip_regret.png", dpi=150)
    plt.close(fig)


BUDGETS = (75_000, 150_000, 300_000, 450_000, 600_000)


def plot_horizon_free(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    tuned = {n: find(runs, "linear", seqs=n) for n in BUDGETS}
    tuned_loss = {n: tuned[n]["val"][-1][1] for n in BUDGETS}
    steps_at = {n: tuned[n]["val"][-1][0] for n in BUDGETS}
    for name, color, label in (("invsqrt", "C4", "inverse sqrt"), ("sgdr2000", "C5", "SGDR (T = 2000, min 0.1×)")):
        run = find(runs, name)
        vs, vl = val_series(run)
        axes[0].plot(tokens_m(vs), vl, color=color, label=label)
        reg_steps = [steps_at[n] for n in BUDGETS]
        reg = [np.interp(s, vs, vl) - tuned_loss[n] for s, n in zip(reg_steps, BUDGETS)]
        axes[1].plot(tokens_m(reg_steps), reg, "o-", color=color, label=label)
    axes[0].plot(tokens_m([steps_at[n] for n in BUDGETS]), [tuned_loss[n] for n in BUDGETS], "ks--", label="linear tuned to each budget")
    axes[0].set_ylim(2.9, 3.8)
    axes[0].set_ylabel("validation loss")
    axes[1].set_ylabel("regret vs budget-tuned linear")
    axes[1].axhline(0, color="gray", lw=.8)
    for ax in axes:
        ax.set_xlabel("training tokens (M)")
        ax.grid(alpha=.3)
    axes[0].legend()
    fig.savefig(OUTPUT_DIR / "p2d_horizon_free.png", dpi=150)
    plt.close(fig)


def print_tables(runs, reference):
    ref_final = reference["val"][-1][1]
    print(f"Reference: {reference['schedule']} lr={reference['lr']:g}, final {ref_final:.4f}\n")
    print("| Schedule | Peak LR | Final val loss | EMA 0.9 | EMA 0.99 | EMA 0.999 | EMA 0.9995 | Raw regret | Best-EMA regret |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for schedule in SCHEDULES:
        for lr in LRS:
            v = find(runs, schedule, lr)["val"][-1][1:]
            print(f"| {SCHEDULE_LABELS[schedule]} | {lr:g} | " + " | ".join(f"{x:.4f}" for x in v) +
                  f" | {v[0] - ref_final:+.4f} | {min(v) - ref_final:+.4f} |")
    print()
    control = find(runs, "wsd0.2")
    print("| Dip | min regret (tokens M) | final regret |")
    print("| --- | ---: | ---: |")
    for dip in DIPS:
        steps, reg = regret(find(runs, "wsd0.2", dip=dip), control)
        i = int(np.argmin(reg))
        print(f"| {DIP_LABELS[dip]} | {reg[i]:+.4f} ({tokens_m(steps[i]):.0f}) | {reg[-1]:+.4f} |")
    print()
    print("| Budget (tokens) | linear tuned | invsqrt | regret | SGDR | regret |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    inv, sg = find(runs, "invsqrt"), find(runs, "sgdr2000")
    for n in BUDGETS:
        t = find(runs, "linear", seqs=n)
        step, loss = t["val"][-1][0], t["val"][-1][1]
        a = np.interp(step, *val_series(inv))
        b = np.interp(step, *val_series(sg))
        print(f"| {step * TOKENS_PER_STEP / 1e6:.0f}M | {loss:.4f} | {a:.4f} | {a - loss:+.4f} | {b:.4f} | {b - loss:+.4f} |")


def main():
    runs = load(refresh="--refresh" in sys.argv)
    reference = reference_run(runs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_schedules(runs)
    plot_ema(runs, reference)
    plot_dip(runs)
    plot_horizon_free(runs)
    print_tables(runs, reference)


if __name__ == "__main__":
    main()
