"""A2 P3.2: batch-size scaling of LR and WD in the d8 language model (614.4M tokens)."""
import sys
from collections import defaultdict
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.a2.provided_sweeps import load
from scripts import a2_p1_lr_scaling as p1


OUTPUT_DIR = Path(__file__).parents[1] / "notes" / "a2" / "p3"
WANDB_PROJECT = p1.WANDB_PROJECT
TOKENS = 614_400_000
BATCHES = (8, 16, 32, 64)


def load_runs(tag):
    """Finished runs at 614.4M tokens with this tag: dicts of batch, LR, WD, beta1 and final val loss."""
    runs = wandb.Api().runs(WANDB_PROJECT, filters={"tags": tag, "state": "finished"})
    return [dict(batch=r.config["batch_size"], learning_rate=r.config["learning_rate"],
                 weight_decay=r.config["weight_decay"], beta1=r.config["beta1"],
                 final_val_loss=r.summary["val_loss"])
            for r in runs if r.config["num_train_sequences"] * 1024 == TOKENS]


def supplied_batch64():
    """Supplied B=64 runs at 614.4M tokens (P1a) with WD .1."""
    return [dict(batch=64, learning_rate=r["learning_rate"], weight_decay=r["weight_decay"], beta1=r["beta1"],
                 final_val_loss=r["final_val_loss"])
            for r in load("P1a") if r["tokens"] == TOKENS and r["weight_decay"] == .1]


def lr_points_by_batch(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["batch"]].append((row["learning_rate"], row["final_val_loss"]))
    return {batch: np.array(sorted(points)) for batch, points in sorted(groups.items())}


def fit_batch_optima(points, rng):
    windows = {b: p1.fit_window(p) for b, p in points.items()}
    optima = {b: p1.quadratic_optimum(w[:, 0], w[:, 1]) for b, w in windows.items()}
    draws = np.array([[p1.quadratic_optimum(w[:, 0], w[:, 1] + rng.normal(0, p1.SEED_NOISE_SD, len(w)))[0]
                       for w in windows.values()] for _ in range(p1.NUM_NOISE_DRAWS)])
    return windows, optima, draws[np.all(np.isfinite(draws), axis=1)]


def save(fig, name):
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def part_a():
    rng = np.random.default_rng(0)
    points = lr_points_by_batch(load_runs("a2-p32a") + supplied_batch64())
    windows, optima, draws = fit_batch_optima(points, rng)
    batches = np.array(list(points))
    lr_stars = np.array([optima[b][0] for b in batches])
    low, high = np.percentile(draws, [16, 84], axis=0)
    best = np.array([points[b][:, 1].min() for b in batches])
    best_lr = np.array([points[b][points[b][:, 1].argmin(), 0] for b in batches])
    exponent, log_coefficient = p1.fit_power_law(batches, lr_stars)

    lr_grid = sorted({lr for p in points.values() for lr in p[:, 0]})
    print("final val loss (rows: LR, columns: batch size)")
    for lr in lr_grid:
        cells = [f"{dict(points[b][:, ::1]).get(lr, np.nan):.4f}" for b in batches]
        print(f"{lr:g}".ljust(8), *cells)
    for b, lr, lo, hi, loss, blr in zip(batches, lr_stars, low, high, best, best_lr):
        print(f"B={b:3d} updates={TOKENS // 1024 // b:6d} fitted lr*={lr:.5f} [{lo:.5f}, {hi:.5f}] "
              f"best measured {loss:.4f} at lr {blr:g}")
    print(f"power law over the four fitted optima: lr* ∝ B^{exponent:.3f}")
    print(f"ratio of best loss to B=64 best: {np.round(best - best[-1], 4)}")

    fig, (curve_ax, lr_ax, loss_ax) = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {8: "C3", 16: "C1", 32: "C2", 64: "C0"}
    for b in batches:
        pts, w = points[b], windows[b]
        curve_ax.plot(pts[:, 0], pts[:, 1], "o", color=colors[b])
        grid = np.geomspace(pts[:, 0].min() / 1.15, pts[:, 0].max() * 1.15, 100)
        a, bb, c = np.polyfit(np.log2(w[:, 0]), w[:, 1], 2)
        curve_ax.plot(grid, a * np.log2(grid)**2 + bb * np.log2(grid) + c, "-", color=colors[b], label=f"B = {b}")
        curve_ax.plot(optima[b][0], optima[b][1], "*", color=colors[b], markersize=14, markeredgecolor="k")
    curve_ax.set_xscale("log")
    curve_ax.set_xticks([.00075, .0015, .003, .006], ["0.00075", "0.0015", "0.003", "0.006"])
    curve_ax.minorticks_off()
    curve_ax.set_xlabel("peak learning rate")
    curve_ax.set_ylabel("final val loss")
    curve_ax.set_title("Loss vs LR at each batch size (WD 0.1)")
    curve_ax.legend(fontsize=9)

    lr_ax.errorbar(batches, lr_stars, yerr=[lr_stars - low, high - lr_stars], fmt="o-", color="C0", capsize=4,
                   label="fitted optimum (±1σ)")
    line = np.geomspace(batches.min() / 1.2, batches.max() * 1.2, 50)
    lr_ax.plot(line, np.exp(log_coefficient) * line**exponent, "--", color="C0", alpha=.7,
               label=f"power law ∝ B^{exponent:.2f}")
    lr_ax.set_xscale("log", base=2)
    lr_ax.set_yscale("log")
    ticks = [.001, .0015, .002, .003, .004]
    lr_ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    lr_ax.set_xticks(batches, [str(b) for b in batches])
    lr_ax.minorticks_off()
    lr_ax.set_xlabel("batch size B")
    lr_ax.set_ylabel("fitted optimal peak LR")
    lr_ax.set_title("Optimal LR vs batch size")
    lr_ax.legend(fontsize=9)

    loss_ax.plot(batches, best, "o-", color="C2", label="best measured")
    loss_ax.axhspan(best.min() - p1.SEED_NOISE_SD, best.min() + p1.SEED_NOISE_SD, color="gray", alpha=.2,
                    label="±1σ seed noise around lowest")
    loss_ax.set_xscale("log", base=2)
    loss_ax.set_xticks(batches, [str(b) for b in batches])
    loss_ax.minorticks_off()
    loss_ax.set_xlabel("batch size B")
    loss_ax.set_ylabel("best measured final val loss")
    loss_ax.set_title("Best loss vs batch size")
    loss_ax.legend(fontsize=9)
    save(fig, "p32a_batch_lr.png")


FIXED_LR = .0015
FIXED_WD = .1
TARGET_BATCHES = (128, 256)


def supplied_wd_sweep():
    """Supplied B=64 runs at LR .0015, 614.4M tokens: WD .1 (P1a) and WD {.1, .2, .4} (P2a)."""
    return [dict(batch=64, learning_rate=r["learning_rate"], weight_decay=r["weight_decay"], beta1=r["beta1"],
                 final_val_loss=r["final_val_loss"])
            for r in load("P1a") + load("P2a") if r["tokens"] == TOKENS and r["learning_rate"] == FIXED_LR]


def wd_points_by_batch(rows):
    """WD sweeps at LR .0015 and B <= 64; duplicate (B, WD) rows keep the first."""
    seen, groups = set(), defaultdict(list)
    for row in rows:
        key = (row["batch"], row["weight_decay"])
        if row["batch"] <= max(BATCHES) and row["learning_rate"] == FIXED_LR and key not in seen:
            seen.add(key)
            groups[row["batch"]].append((row["weight_decay"], row["final_val_loss"]))
    return {batch: np.array(sorted(points)) for batch, points in sorted(groups.items())}


def power_law_prediction(batches, draws, optima, targets):
    exponent, log_coefficient = p1.fit_power_law(batches, optima)
    fits = [p1.fit_power_law(batches, d) for d in draws]
    predictions = {t: float(np.exp(log_coefficient + exponent * np.log(t))) for t in targets}
    bands = {t: np.percentile([np.exp(c + e * np.log(t)) for e, c in fits], [16, 84]) for t in targets}
    return exponent, log_coefficient, predictions, bands


def part_b_predict():
    """Fit both hypotheses on B <= 64 and extrapolate to the target batches; touches no B > 64 run."""
    rng = np.random.default_rng(0)
    unique = {}
    for r in load_runs("a2-p32a") + load_runs("a2-p32b") + supplied_batch64() + supplied_wd_sweep():
        if r["batch"] <= max(BATCHES):
            unique.setdefault((r["batch"], r["learning_rate"], r["weight_decay"]), r)
    source = list(unique.values())
    lr_points = lr_points_by_batch([r for r in source if r["weight_decay"] == FIXED_WD])
    wd_points = wd_points_by_batch(source)
    batches = np.array(BATCHES)

    results = {}
    for name, points in (("LR at WD .1", lr_points), ("WD at LR .0015", wd_points)):
        windows, optima, draws = fit_batch_optima(points, rng)
        stars = np.array([optima[b][0] for b in batches])
        low, high = np.percentile(draws, [16, 84], axis=0)
        exponent, log_coefficient, predictions, bands = power_law_prediction(batches, draws, stars, TARGET_BATCHES)
        results[name] = dict(points=points, windows=windows, optima=optima, stars=stars, low=low, high=high,
                             exponent=exponent, log_coefficient=log_coefficient, predictions=predictions, bands=bands)
        print(f"\n{name}: optima " + ", ".join(f"B={b}: {x:.5f} [{lo:.5f}, {hi:.5f}]"
                                               for b, x, lo, hi in zip(batches, stars, low, high)))
        print(f"  power law over B in {BATCHES}: ∝ B^{exponent:.3f}")
        for t in TARGET_BATCHES:
            print(f"  predicted optimum at B={t}: {predictions[t]:.5f} (±1σ {bands[t][0]:.5f}–{bands[t][1]:.5f})")
        for b, pts in points.items():
            print(f"  B={b} points: " + ", ".join(f"{x:g}:{y:.4f}" for x, y in pts))
    plot_b_sources(results)
    return results


def plot_b_sources(results):
    batches = np.array(BATCHES)
    colors = {8: "C3", 16: "C1", 32: "C2", 64: "C0"}
    wd = results["WD at LR .0015"]
    fig, (curve_ax, lr_ax, wd_ax) = plt.subplots(1, 3, figsize=(17, 4.8))
    for b in batches:
        pts, w = wd["points"][b], wd["windows"][b]
        curve_ax.plot(pts[:, 0], pts[:, 1], "o", color=colors[b])
        grid = np.geomspace(pts[:, 0].min() / 1.15, pts[:, 0].max() * 1.15, 100)
        a, bb, c = np.polyfit(np.log2(w[:, 0]), w[:, 1], 2)
        curve_ax.plot(grid, a * np.log2(grid)**2 + bb * np.log2(grid) + c, "-", color=colors[b], label=f"B = {b}")
        curve_ax.plot(wd["optima"][b][0], wd["optima"][b][1], "*", color=colors[b], markersize=14, markeredgecolor="k")
    curve_ax.set_xscale("log")
    curve_ax.set_ylim(2.915, 3.05)
    curve_ax.set_xlabel("weight decay")
    curve_ax.set_ylabel("final val loss")
    curve_ax.set_title("Loss vs WD at each batch size (LR 0.0015)")
    curve_ax.legend(fontsize=9)
    for ax, key, ylabel, title in ((lr_ax, "LR at WD .1", "optimal peak LR (WD 0.1)", "Hypothesis (i): optimal LR vs batch size"),
                                   (wd_ax, "WD at LR .0015", "optimal weight decay (LR 0.0015)", "Hypothesis (ii): optimal WD vs batch size")):
        r = results[key]
        ax.errorbar(batches, r["stars"], yerr=[r["stars"] - r["low"], r["high"] - r["stars"]], fmt="o", color="C0",
                    capsize=4, label="fitted optimum (±1σ)")
        line = np.geomspace(batches.min() / 1.2, max(TARGET_BATCHES) * 1.2, 50)
        ax.plot(line, np.exp(r["log_coefficient"]) * line**r["exponent"], "--", color="C0", alpha=.7,
                label=f"power law ∝ B^{r['exponent']:.2f}")
        for t in TARGET_BATCHES:
            lo, hi = r["bands"][t]
            ax.errorbar(t, r["predictions"][t], yerr=[[r["predictions"][t] - lo], [hi - r["predictions"][t]]],
                        fmt="x", color="k", markersize=10, markeredgewidth=2, capsize=4,
                        label="prediction (±1σ)" if t == TARGET_BATCHES[0] else None)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([*batches, *TARGET_BATCHES], [str(b) for b in (*batches, *TARGET_BATCHES)])
        ax.minorticks_off()
        ax.set_xlabel("batch size B")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
    save(fig, "p32b_predictions.png")


PREDICTED = {128: dict(lr=.00537, wd=.887), 256: dict(lr=.00783, wd=1.7)}


def family_points(rows, hypothesis):
    """Runs for hypothesis (i) (WD .1, LR varies) or (ii) (LR .0015, WD varies), as (x, loss) rows."""
    if hypothesis == "i":
        return np.array(sorted((r["learning_rate"], r["final_val_loss"]) for r in rows if r["weight_decay"] == FIXED_WD))
    return np.array(sorted((r["weight_decay"], r["final_val_loss"]) for r in rows if r["learning_rate"] == FIXED_LR))


def part_b_test():
    """Compare the recorded predictions with the grid runs and the runs at the predicted configurations."""
    grid = [r for r in load_runs("a2-p32b") if r["batch"] in TARGET_BATCHES]
    predicted_runs = load_runs("a2-p32b-pred")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for col, batch in enumerate(TARGET_BATCHES):
        grid_b = [r for r in grid if r["batch"] == batch]
        best_all = min(grid_b, key=lambda r: r["final_val_loss"])
        print(f"\nB={batch}: best grid run overall: LR {best_all['learning_rate']:g}, WD {best_all['weight_decay']:g}, "
              f"loss {best_all['final_val_loss']:.4f}")
        for row, (hyp, name, xlabel) in enumerate((("i", "(i) WD 0.1, LR scaled", "peak learning rate"),
                                                   ("ii", "(ii) LR 0.0015, WD scaled", "weight decay"))):
            ax = axes[row, col]
            x_pred = PREDICTED[batch]["lr" if hyp == "i" else "wd"]
            pred_run = [r for r in predicted_runs if r["batch"] == batch and
                        (r["weight_decay"] == FIXED_WD and r["learning_rate"] == PREDICTED[batch]["lr"] if hyp == "i"
                         else r["learning_rate"] == FIXED_LR and r["weight_decay"] == PREDICTED[batch]["wd"])]
            points = family_points(grid_b, hyp)
            best = points[:, 1].min()
            best_x = points[points[:, 1].argmin(), 0]
            window = p1.fit_window(points)
            vertex, vertex_loss, (a, b, c) = p1.quadratic_optimum(window[:, 0], window[:, 1])
            ax.plot(points[:, 0], points[:, 1], "o", color="C0", label="grid runs")
            xs = np.geomspace(points[:, 0].min() / 1.3, points[:, 0].max() * 1.3, 100)
            ax.plot(xs, a * np.log2(xs)**2 + b * np.log2(xs) + c, "-", color="C0", alpha=.6)
            line = f"({hyp}) B={batch}: predicted {x_pred:g}; best grid {best_x:g} (loss {best:.4f}); fitted optimum {vertex:.5f}"
            if pred_run:
                loss = pred_run[0]["final_val_loss"]
                ax.plot(x_pred, loss, "x", color="k", markersize=12, markeredgewidth=2.5, label=f"run at prediction ({loss:.4f})")
                line += (f"; loss at prediction {loss:.4f}; gap to best grid {loss - best:+.4f}; "
                         f"gap to best grid overall {loss - best_all['final_val_loss']:+.4f}")
            else:
                ax.axvline(x_pred, color="k", linestyle="--", label="prediction (run not finished)")
            print(line)
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("final val loss")
            ax.set_title(f"B = {batch}, hypothesis {name}")
            ax.legend(fontsize=9)
    save(fig, "p32b_test.png")


MOMENTUM_BATCHES = (8, 256)


def momentum_runs(batch):
    """beta1 -> loss at the batch's best (LR, WD) pair: beta1=.9 is the best (a)/(b) run, the rest are (c) runs."""
    reference = [r for r in load_runs("a2-p32a") + load_runs("a2-p32b") if r["batch"] == batch]
    best = min(reference, key=lambda r: r["final_val_loss"])
    runs = [best] + [r for r in load_runs("a2-p32c") if r["batch"] == batch]
    return best, sorted(runs, key=lambda r: r["beta1"])


def part_c():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, batch in zip(axes, MOMENTUM_BATCHES):
        best, runs = momentum_runs(batch)
        beta1s = [r["beta1"] for r in runs]
        losses = [r["final_val_loss"] for r in runs]
        ax.plot(beta1s, losses, "o-", color="C0", label=f"LR {best['learning_rate']:g}, WD {best['weight_decay']:g}")
        for x, y in zip(beta1s, losses):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        retuned = [r for r in load_runs("a2-p32c-retune") if r["batch"] == batch]
        for r in retuned:
            ax.plot(0, r["final_val_loss"], "s", color="C3", label=f"β₁ = 0, LR {r['learning_rate']:g}")
            ax.annotate(f"{r['final_val_loss']:.4f}", (0, r["final_val_loss"]), textcoords="offset points",
                        xytext=(14, -3), fontsize=8)
        ax.set_xlabel("β₁")
        ax.set_ylabel("final val loss")
        ax.set_title(f"B = {batch}: loss vs β₁ at the best (LR, WD) pair")
        ax.legend(fontsize=8)
        print(f"B={batch}: LR {best['learning_rate']:g} WD {best['weight_decay']:g}  " +
              "  ".join(f"β1={b:g}: {l:.4f}" for b, l in zip(beta1s, losses)) +
              f"  | benefit of momentum (β1=0 minus best): {losses[0] - min(losses):.4f}")
        for r in retuned:
            print(f"   retuned β1=0 at LR {r['learning_rate']:g}: {r['final_val_loss']:.4f}")
    save(fig, "p32c_momentum.png")


def main():
    for part in sys.argv[1:] or ("a",):
        {"a": part_a, "b": part_b_predict, "bt": part_b_test, "c": part_c}[part]()


if __name__ == "__main__":
    main()
