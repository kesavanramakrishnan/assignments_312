"""A2 P1: per-budget optimal AdamW LR from quadratic fits, and a power law in tokens."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.provided_sweeps import load


OUTPUT_DIR = Path(__file__).parents[1] / "notes" / "a2" / "p1"
# Seed-to-seed SD of final val loss at the default recipe, measured in A1 P3.
SEED_NOISE_SD = 0.0024
PREDICTION_TOKENS = (1.2288e9, 1.8432e9, 2.4576e9)
NUM_NOISE_DRAWS = 4000


def points_by_budget(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["tokens"]].append((row["learning_rate"], row["final_val_loss"]))
    return {tokens: np.array(sorted(points)) for tokens, points in sorted(groups.items())}


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


def noise_draws(budgets, rng):
    """Refit optima and the power law after adding seed-level noise to every loss."""
    tokens = np.array(list(budgets))
    optima, exponents, predictions = [], [], []
    for _ in range(NUM_NOISE_DRAWS):
        draw = [
            quadratic_optimum(p[:, 0], p[:, 1] + rng.normal(0, SEED_NOISE_SD, len(p)))[0]
            for p in budgets.values()
        ]
        if not np.all(np.isfinite(draw)):
            continue
        exponent, coefficient = fit_power_law(tokens, draw)
        optima.append(draw)
        exponents.append(exponent)
        predictions.append(coefficient * np.array(PREDICTION_TOKENS) ** exponent)
    return np.array(optima), np.array(exponents), np.array(predictions)


def plot(budgets, fits, exponent, coefficient, optima_draws, predictions, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(budgets)))
    for color, (tokens, points), (lr_star, loss_star, coeffs) in zip(colors, budgets.items(), fits):
        label = f"{tokens / 1e6:g}M"
        grid = np.geomspace(points[:, 0].min() / 1.3, points[:, 0].max() * 1.3, 200)
        left.plot(points[:, 0], points[:, 1], "o", color=color, label=label)
        left.plot(grid, np.polyval(coeffs, np.log2(grid)), "-", color=color, alpha=0.6)
        left.plot(lr_star, loss_star, "*", color=color, markersize=14, markeredgecolor="k")
    left.set_xscale("log")
    left.set_xlabel("peak learning rate")
    left.set_ylabel("final validation loss")
    left.set_title("Quadratic fits in log LR (stars: fitted optima)")
    left.legend(title="tokens")

    tokens = np.array(list(budgets))
    lr_stars = np.array([fit[0] for fit in fits])
    low, high = np.percentile(optima_draws, [16, 84], axis=0)
    right.errorbar(tokens, lr_stars, yerr=[lr_stars - low, high - lr_stars], fmt="o",
                   color="C0", capsize=4, label="fitted optimum (±1σ seed noise)")
    line = np.geomspace(tokens.min() / 1.2, max(PREDICTION_TOKENS) * 1.2, 100)
    right.plot(line, coefficient * line**exponent, "--", color="C0",
               label=rf"$\eta^* \propto D^{{{exponent:.3f}}}$")
    predicted = coefficient * np.array(PREDICTION_TOKENS) ** exponent
    pred_low, pred_high = np.percentile(predictions, [16, 84], axis=0)
    right.errorbar(PREDICTION_TOKENS, predicted, yerr=[predicted - pred_low, pred_high - predicted],
                   fmt="o", mfc="white", color="C3", capsize=4, label="P1(b) predictions")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("training tokens D")
    right.set_ylabel("optimal peak LR")
    right.set_title("Optimal LR vs token budget")
    right.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", default="P1a", choices=("P1a",))
    args = parser.parse_args(argv)

    budgets = points_by_budget(load(args.part))
    fits = [quadratic_optimum(p[:, 0], p[:, 1]) for p in budgets.values()]
    tokens = np.array(list(budgets))
    lr_stars = np.array([fit[0] for fit in fits])
    exponent, coefficient = fit_power_law(tokens, lr_stars)
    optima_draws, exponent_draws, predictions = noise_draws(budgets, np.random.default_rng(0))

    print("tokens      fitted lr*   ±1σ (seed noise)       loss at lr*   curvature a")
    low, high = np.percentile(optima_draws, [16, 84], axis=0)
    for t, (lr_star, loss_star, (a, _, _)), lo, hi in zip(tokens, fits, low, high):
        print(f"{t / 1e6:8.1f}M  {lr_star:.5f}     [{lo:.5f}, {hi:.5f}]   {loss_star:.4f}        {a:.4f}")
    e_lo, e_hi = np.percentile(exponent_draws, [16, 84])
    print(f"\npower law: lr* = {coefficient:.4g} * D^{exponent:.4f}   exponent ±1σ [{e_lo:.3f}, {e_hi:.3f}]")
    print("doubling D multiplies lr* by", f"{2**exponent:.3f}")
    pred_low, pred_high = np.percentile(predictions, [16, 84], axis=0)
    print("\nP1(b) predictions from this rule:")
    for t, lo, hi in zip(PREDICTION_TOKENS, pred_low, pred_high):
        print(f"{t / 1e9:.4f}B  lr* = {coefficient * t**exponent:.5f}   ±1σ [{lo:.5f}, {hi:.5f}]")

    path = OUTPUT_DIR / "p1a_lr_scaling.png"
    plot(budgets, fits, exponent, coefficient, optima_draws, predictions, path)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
