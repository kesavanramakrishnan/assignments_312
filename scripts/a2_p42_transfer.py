"""A2 P4.2(b,c): optimal base LR vs width for the course baseline and muP, and the held-out width-1024 prediction."""
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.a2.provided_sweeps import load
from scripts import a2_p1_lr_scaling as p1


OUTPUT_DIR = Path(__file__).parents[1] / "notes" / "a2" / "p4"
SOURCE_WIDTHS = (128, 256, 512)
HELD_OUT_WIDTH = 1024
PRESCRIPTIONS = {"base": "Course baseline", "mup": "µP"}
TOKENS = 153_600_000
SOURCE_LR = .003
COLORS = {128: "C3", 256: "C1", 512: "C0", 1024: "C2"}


def load_sweeps(widths=(128, 256)):
    """(prescription, width) -> sorted (LR, final val loss). Width 512 is the supplied P1 curve for both."""
    sweeps = {}
    for run in wandb.Api().runs(p1.WANDB_PROJECT, filters={"tags": {"$in": ["a2-p42a", "a2-p42c"]}, "state": "finished"}):
        width = run.config["model_config"]["hidden_size"]
        if width not in widths:
            continue
        prescription = next(t for t in run.tags if t.startswith("p42-"))[4:]
        sweeps.setdefault((prescription, width), []).append((run.config["learning_rate"], run.summary["val_loss"]))
    supplied = sorted((r["learning_rate"], r["final_val_loss"]) for r in load("P1a") if r["tokens"] == TOKENS)
    for prescription in PRESCRIPTIONS:
        sweeps[(prescription, 512)] = supplied
    return {key: np.array(sorted(points)) for key, points in sweeps.items()}


def save(fig, name):
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def fit_prescription(sweeps, prescription, rng):
    windows = {w: p1.fit_window(sweeps[(prescription, w)]) for w in SOURCE_WIDTHS}
    optima = {w: p1.quadratic_optimum(win[:, 0], win[:, 1]) for w, win in windows.items()}
    draws = np.array([[p1.quadratic_optimum(win[:, 0], win[:, 1] + rng.normal(0, p1.SEED_NOISE_SD, len(win)))[0]
                       for win in windows.values()] for _ in range(p1.NUM_NOISE_DRAWS)])
    draws = draws[np.all(np.isfinite(draws), axis=1)]
    stars = np.array([optima[w][0] for w in SOURCE_WIDTHS])
    exponent, log_coefficient = p1.fit_power_law(SOURCE_WIDTHS, stars)
    fits = [p1.fit_power_law(SOURCE_WIDTHS, d) for d in draws]
    prediction = float(np.exp(log_coefficient + exponent * np.log(HELD_OUT_WIDTH)))
    band = np.percentile([np.exp(c + e * np.log(HELD_OUT_WIDTH)) for e, c in fits], [16, 84])
    low, high = np.percentile(draws, [16, 84], axis=0)
    return dict(windows=windows, optima=optima, stars=stars, low=low, high=high, exponent=exponent,
                log_coefficient=log_coefficient, prediction=prediction, band=band)


def predict():
    """Fit widths 128, 256, 512 only; touches no width-1024 run."""
    rng = np.random.default_rng(0)
    sweeps = load_sweeps()
    results = {p: fit_prescription(sweeps, p, rng) for p in PRESCRIPTIONS}
    for p, r in results.items():
        print(f"\n{PRESCRIPTIONS[p]}: " + ", ".join(f"w={w}: {x:.5f} [{lo:.5f}, {hi:.5f}]"
              for w, x, lo, hi in zip(SOURCE_WIDTHS, r["stars"], r["low"], r["high"])))
        print(f"  power law over widths {SOURCE_WIDTHS}: ∝ width^{r['exponent']:.3f}; predicted optimum at "
              f"{HELD_OUT_WIDTH}: {r['prediction']:.5f} (±1σ {r['band'][0]:.5f}–{r['band'][1]:.5f})")
        for w in SOURCE_WIDTHS:
            pts = sweeps[(p, w)]
            print(f"  w={w}: " + ", ".join(f"{lr:g}:{loss:.4f}" for lr, loss in pts))
    plot_sources(results, sweeps)
    return results


def plot_sources(results, sweeps):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, (p, name) in enumerate(PRESCRIPTIONS.items()):
        curve_ax, lr_ax = axes[row]
        r = results[p]
        for w in SOURCE_WIDTHS:
            pts, win = sweeps[(p, w)], r["windows"][w]
            curve_ax.plot(pts[:, 0], pts[:, 1], "o", color=COLORS[w])
            grid = np.geomspace(pts[:, 0].min() / 1.15, pts[:, 0].max() * 1.15, 100)
            a, b, c = np.polyfit(np.log2(win[:, 0]), win[:, 1], 2)
            curve_ax.plot(grid, a * np.log2(grid)**2 + b * np.log2(grid) + c, "-", color=COLORS[w], label=f"width {w}")
            curve_ax.plot(r["optima"][w][0], r["optima"][w][1], "*", color=COLORS[w], markersize=14, markeredgecolor="k")
        curve_ax.set_xscale("log")
        curve_ax.set_xlabel("peak base LR")
        curve_ax.set_ylabel("final val loss")
        curve_ax.set_title(f"{name}: loss vs LR")
        curve_ax.legend(fontsize=9)
        lr_ax.errorbar(SOURCE_WIDTHS, r["stars"], yerr=[r["stars"] - r["low"], r["high"] - r["stars"]], fmt="o",
                       color="C0", capsize=4, label="fitted optimum (±1σ)")
        line = np.geomspace(100, 1300, 50)
        lr_ax.plot(line, np.exp(r["log_coefficient"]) * line**r["exponent"], "--", color="C0", alpha=.7,
                   label=f"power law ∝ width^{r['exponent']:.2f}")
        lr_ax.errorbar(HELD_OUT_WIDTH, r["prediction"], yerr=[[r["prediction"] - r["band"][0]], [r["band"][1] - r["prediction"]]],
                       fmt="x", color="k", markersize=10, markeredgewidth=2, capsize=4, label="prediction at 1024 (±1σ)")
        lr_ax.axhline(SOURCE_LR, color="gray", linestyle=":", label="transferred source LR 0.003")
        lr_ax.set_xscale("log", base=2)
        lr_ax.set_yscale("log")
        lr_ax.set_xticks([*SOURCE_WIDTHS, HELD_OUT_WIDTH], [str(w) for w in (*SOURCE_WIDTHS, HELD_OUT_WIDTH)])
        lr_ax.minorticks_off()
        lr_ax.set_xlabel("width")
        lr_ax.set_ylabel("fitted optimal base LR")
        lr_ax.set_title(f"{name}: optimal LR vs width")
        lr_ax.legend(fontsize=8)
    save(fig, "p42bc_width_sources.png")


def main():
    predict()


if __name__ == "__main__":
    main()
