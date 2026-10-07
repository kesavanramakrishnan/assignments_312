"""A2 P1: per-budget optimal LR from local quadratic fits, and power laws in tokens."""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.provided_sweeps import load


OUTPUT_DIR = Path(__file__).parents[1] / "notes" / "a2" / "p1"
# Seed-to-seed SD of final val loss at the default recipe, measured in A1 P3.
SEED_NOISE_SD = 0.0024
NUM_NOISE_DRAWS = 4000
ADAMW_PREDICTION_TOKENS = (1.2288e9, 1.8432e9, 2.4576e9)
HYPERBALL_PREDICTION_TOKENS = (1.2288e9,)


def points_by_budget(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["tokens"]].append((row["learning_rate"], row["final_val_loss"]))
    return {tokens: np.array(sorted(points)) for tokens, points in sorted(groups.items())}


def fit_window(points):
    """The lowest measured loss and its two neighbors in LR."""
    best = int(points[:, 1].argmin())
    start = min(max(best - 1, 0), len(points) - 3)
    return points[start:start + 3]


def quadratic_optimum(lrs, losses):
    """Vertex of a parabola in log2(LR); returns (optimal LR, loss at vertex, coefficients)."""
    a, b, c = np.polyfit(np.log2(lrs), losses, 2)
    if a <= 0:
        return np.nan, np.nan, (a, b, c)
    x = -b / (2 * a)
    return 2.0**x, c - b * b / (4 * a), (a, b, c)


def fit_power_law(tokens, optimal_lrs):
    """Least squares in log-log; returns (exponent, coefficient) for lr = coefficient * tokens**exponent."""
    exponent, log_coefficient = np.polyfit(np.log(tokens), np.log(optimal_lrs), 1)
    return exponent, np.exp(log_coefficient)


@dataclass
class LRScalingFit:
    label: str
    budgets: dict
    windows: dict
    optima: list
    exponent: float
    coefficient: float
    prediction_tokens: tuple
    optima_draws: np.ndarray
    exponent_draws: np.ndarray
    prediction_draws: np.ndarray

    @property
    def tokens(self):
        return np.array(list(self.budgets))

    @property
    def lr_stars(self):
        return np.array([optimum[0] for optimum in self.optima])

    def predict(self, tokens):
        return self.coefficient * np.asarray(tokens, dtype=float) ** self.exponent


def fit_scaling(label, rows, prediction_tokens, rng):
    budgets = points_by_budget(rows)
    windows = {tokens: fit_window(points) for tokens, points in budgets.items()}
    optima = [quadratic_optimum(w[:, 0], w[:, 1]) for w in windows.values()]
    tokens = np.array(list(budgets))
    exponent, coefficient = fit_power_law(tokens, [optimum[0] for optimum in optima])

    optima_draws, exponent_draws, prediction_draws = [], [], []
    for _ in range(NUM_NOISE_DRAWS):
        draw = [
            quadratic_optimum(w[:, 0], w[:, 1] + rng.normal(0, SEED_NOISE_SD, len(w)))[0]
            for w in windows.values()
        ]
        if not np.all(np.isfinite(draw)):
            continue
        draw_exponent, draw_coefficient = fit_power_law(tokens, draw)
        optima_draws.append(draw)
        exponent_draws.append(draw_exponent)
        prediction_draws.append(draw_coefficient * np.array(prediction_tokens) ** draw_exponent)
    return LRScalingFit(label, budgets, windows, optima, exponent, coefficient, prediction_tokens,
                        np.array(optima_draws), np.array(exponent_draws), np.array(prediction_draws))


def report(fit):
    print(f"== {fit.label} ==")
    print("tokens      fit window LRs           lr*        ±1σ (seed noise)       loss at lr*")
    low, high = np.percentile(fit.optima_draws, [16, 84], axis=0)
    for (t, window), (lr_star, loss_star, _), lo, hi in zip(fit.windows.items(), fit.optima, low, high):
        lrs = ", ".join(f"{lr:g}" for lr in window[:, 0])
        print(f"{t / 1e6:8.1f}M  {lrs:22s}  {lr_star:.5f}    [{lo:.5f}, {hi:.5f}]   {loss_star:.4f}")
    e_lo, e_hi = np.percentile(fit.exponent_draws, [16, 84])
    print(f"power law: lr* = {fit.coefficient:.4g} * D^{fit.exponent:.4f}   "
          f"exponent ±1σ [{e_lo:.3f}, {e_hi:.3f}]   per doubling x{2**fit.exponent:.3f}")
    pred_low, pred_high = np.percentile(fit.prediction_draws, [16, 84], axis=0)
    for t, predicted, lo, hi in zip(fit.prediction_tokens, fit.predict(fit.prediction_tokens), pred_low, pred_high):
        print(f"prediction {t / 1e9:.4f}B  lr* = {predicted:.5f}   ±1σ [{lo:.5f}, {hi:.5f}]")
    print()


def plot_curves(ax, fit):
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(fit.budgets)))
    for color, (tokens, points), window, (lr_star, loss_star, coeffs) in zip(
        colors, fit.budgets.items(), fit.windows.values(), fit.optima
    ):
        grid = np.geomspace(window[:, 0].min() / 1.3, window[:, 0].max() * 1.3, 200)
        ax.plot(points[:, 0], points[:, 1], "o", color=color, label=f"{tokens / 1e6:g}M")
        ax.plot(grid, np.polyval(coeffs, np.log2(grid)), "-", color=color, alpha=0.6)
        ax.plot(lr_star, loss_star, "*", color=color, markersize=14, markeredgecolor="k")
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("final validation loss")
    ax.set_title(f"{fit.label}: quadratic fits in log LR (stars: optima)")
    ax.legend(title="tokens")


def plot_optima(ax, fit, color, x_max, show_predictions=True):
    low, high = np.percentile(fit.optima_draws, [16, 84], axis=0)
    lr_stars = fit.lr_stars
    ax.errorbar(fit.tokens, lr_stars, yerr=[lr_stars - low, high - lr_stars], fmt="o",
                color=color, capsize=4, label=f"{fit.label} fitted optimum (±1σ)")
    line = np.geomspace(fit.tokens.min() / 1.2, x_max, 100)
    ax.plot(line, fit.predict(line), "--", color=color,
            label=rf"{fit.label}: $\eta^* \propto D^{{{fit.exponent:.3f}}}$")
    if show_predictions:
        predicted = fit.predict(fit.prediction_tokens)
        pred_low, pred_high = np.percentile(fit.prediction_draws, [16, 84], axis=0)
        ax.errorbar(fit.prediction_tokens, predicted, yerr=[predicted - pred_low, pred_high - predicted],
                    fmt="o", mfc="white", color=color, capsize=4, label=f"{fit.label} predictions")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(fit.tokens.min() / 1.4, x_max * 1.1)
    ax.set_xlabel("training tokens D")
    ax.set_ylabel("optimal peak LR")


def save(fig, name):
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    rng = np.random.default_rng(0)
    adamw = fit_scaling("AdamW", load("P1a"), ADAMW_PREDICTION_TOKENS, rng)
    hyperball = fit_scaling("Hyperball", load("P1d"), HYPERBALL_PREDICTION_TOKENS, rng)
    report(adamw)
    report(hyperball)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_curves(left, adamw)
    plot_optima(right, adamw, "C0", max(ADAMW_PREDICTION_TOKENS) * 1.2)
    right.set_title("AdamW optimal LR vs token budget")
    right.legend()
    save(fig, "p1a_lr_scaling.png")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_curves(left, hyperball)
    plot_optima(right, hyperball, "C3", max(HYPERBALL_PREDICTION_TOKENS) * 1.2)
    plot_optima(right, adamw, "C0", max(HYPERBALL_PREDICTION_TOKENS) * 1.2, show_predictions=False)
    right.set_title("Optimal LR vs token budget")
    right.legend(fontsize=8)
    save(fig, "p1d_hyperball_vs_adamw.png")


if __name__ == "__main__":
    main()
