"""A2 P1: per-budget optimal LR from local quadratic fits, and power laws in tokens."""
from collections import defaultdict
from dataclasses import dataclass, replace
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
TARGET_TOKENS = (4.9152e9,)


def token_label(tokens):
    return f"{tokens / 1e6:g}M" if tokens < 1e9 else f"{tokens / 1e9:g}B"


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
    """Least squares in log-log; returns (exponent, log coefficient) for lr = e**log_coefficient * tokens**exponent."""
    exponent, log_coefficient = np.polyfit(np.log(tokens), np.log(optimal_lrs), 1)
    return exponent, log_coefficient


def power_law_value(exponent, log_coefficient, tokens):
    return np.exp(log_coefficient + exponent * np.log(np.asarray(tokens, dtype=float)))


@dataclass
class OptimaFit:
    label: str
    budgets: dict
    windows: dict
    optima: list
    optima_draws: np.ndarray

    @property
    def tokens(self):
        return np.array(list(self.budgets))

    @property
    def lr_stars(self):
        return np.array([optimum[0] for optimum in self.optima])

    def lr_star_band(self):
        return np.percentile(self.optima_draws, [16, 84], axis=0)


@dataclass
class PowerLaw:
    label: str
    exponent: float
    log_coefficient: float
    exponent_draws: np.ndarray
    prediction_tokens: tuple
    prediction_draws: np.ndarray

    @property
    def coefficient(self):
        return np.exp(self.log_coefficient)

    def predict(self, tokens):
        return power_law_value(self.exponent, self.log_coefficient, tokens)


def combine(label, *fits):
    """Concatenate budgets from fits that used independent noise draws."""
    count = min(len(fit.optima_draws) for fit in fits)
    return OptimaFit(
        label,
        {t: p for fit in fits for t, p in fit.budgets.items()},
        {t: w for fit in fits for t, w in fit.windows.items()},
        [optimum for fit in fits for optimum in fit.optima],
        np.concatenate([fit.optima_draws[:count] for fit in fits], axis=1),
    )


def fit_optima(label, rows, rng):
    budgets = points_by_budget(rows)
    windows = {tokens: fit_window(points) for tokens, points in budgets.items()}
    optima = [quadratic_optimum(w[:, 0], w[:, 1]) for w in windows.values()]
    draws = np.array([
        [quadratic_optimum(w[:, 0], w[:, 1] + rng.normal(0, SEED_NOISE_SD, len(w)))[0]
         for w in windows.values()]
        for _ in range(NUM_NOISE_DRAWS)
    ])
    return OptimaFit(label, budgets, windows, optima, draws[np.all(np.isfinite(draws), axis=1)])


def fit_scaling(fit, prediction_tokens):
    exponent, log_coefficient = fit_power_law(fit.tokens, fit.lr_stars)
    draw_fits = [fit_power_law(fit.tokens, draw) for draw in fit.optima_draws]
    exponent_draws = np.array([e for e, _ in draw_fits])
    prediction_draws = np.array([power_law_value(e, c, prediction_tokens) for e, c in draw_fits])
    return PowerLaw(fit.label, exponent, log_coefficient, exponent_draws, prediction_tokens,
                    prediction_draws)


def report_optima(fit):
    print(f"== {fit.label} ==")
    print("tokens      fit window LRs           lr*        ±1σ (seed noise)       loss at lr*")
    low, high = fit.lr_star_band()
    for (t, window), (lr_star, loss_star, _), lo, hi in zip(fit.windows.items(), fit.optima, low, high):
        lrs = ", ".join(f"{lr:g}" for lr in window[:, 0])
        print(f"{t / 1e6:8.1f}M  {lrs:22s}  {lr_star:.5f}    [{lo:.5f}, {hi:.5f}]   {loss_star:.4f}")


def report_scaling(law):
    e_lo, e_hi = np.percentile(law.exponent_draws, [16, 84])
    print(f"power law: lr* = {law.coefficient:.4g} * D^{law.exponent:.4f}   "
          f"exponent ±1σ [{e_lo:.3f}, {e_hi:.3f}]   per doubling x{2**law.exponent:.3f}")
    pred_low, pred_high = np.percentile(law.prediction_draws, [16, 84], axis=0)
    for t, predicted, lo, hi in zip(law.prediction_tokens, law.predict(law.prediction_tokens), pred_low, pred_high):
        print(f"prediction {t / 1e9:.4f}B  lr* = {predicted:.5f}   ±1σ [{lo:.5f}, {hi:.5f}]")
    print()


def report_predictions(law, measured):
    print(f"== {law.label} predictions vs {measured.label} fitted optima ==")
    for t, predicted, fitted in zip(measured.tokens, law.predict(measured.tokens), measured.lr_stars):
        print(f"{t / 1e9:.4f}B  predicted {predicted:.5f}  fitted {fitted:.5f}  "
              f"predicted/fitted {predicted / fitted:.3f}")
    print()


def plot_curves(ax, fit):
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(fit.budgets)))
    for color, (tokens, points), window, (lr_star, loss_star, coeffs) in zip(
        colors, fit.budgets.items(), fit.windows.values(), fit.optima
    ):
        grid = np.geomspace(window[:, 0].min() / 1.3, window[:, 0].max() * 1.3, 200)
        ax.plot(points[:, 0], points[:, 1], "o", color=color, label=token_label(tokens))
        ax.plot(grid, np.polyval(coeffs, np.log2(grid)), "-", color=color, alpha=0.6)
        ax.plot(lr_star, loss_star, "*", color=color, markersize=14, markeredgecolor="k")
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("final validation loss")
    ax.set_title(f"{fit.label}: quadratic fits in log LR (stars: optima)")
    ax.legend(title="tokens")


def label_measured_lrs(ax, fit):
    lrs = sorted({lr for points in fit.budgets.values() for lr in points[:, 0]})
    ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
    ax.minorticks_off()


def plot_fitted_optima(ax, fit, color, label):
    low, high = fit.lr_star_band()
    lr_stars = fit.lr_stars
    ax.errorbar(fit.tokens, lr_stars, yerr=[lr_stars - low, high - lr_stars], fmt="o",
                color=color, capsize=4, label=label)


def plot_law(ax, law, color, x_min, x_max, show_predictions=True):
    line = np.geomspace(x_min, x_max, 100)
    ax.plot(line, law.predict(line), "--", color=color,
            label=rf"{law.label}: $\eta^* \propto D^{{{law.exponent:.3f}}}$")
    if show_predictions:
        predicted = law.predict(law.prediction_tokens)
        pred_low, pred_high = np.percentile(law.prediction_draws, [16, 84], axis=0)
        ax.errorbar(law.prediction_tokens, predicted, yerr=[predicted - pred_low, pred_high - predicted],
                    fmt="o", mfc="white", color=color, capsize=4, label=f"{law.label} predictions")


def plot_optima(ax, fit, law, color, x_max, show_predictions=True):
    plot_fitted_optima(ax, fit, color, f"{fit.label} fitted optimum (±1σ)")
    plot_law(ax, law, color, fit.tokens.min() / 1.2, x_max, show_predictions)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(fit.tokens.min() / 1.4, x_max * 1.1)
    ax.set_xlabel("training tokens D")
    ax.set_ylabel("optimal peak LR")


def label_token_budgets(ax, tokens, rotation=0):
    ax.set_xticks(tokens, [token_label(t) for t in tokens], rotation=rotation,
                  ha="right" if rotation else "center")
    ax.minorticks_off()


def save(fig, name):
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    rng = np.random.default_rng(0)
    adamw = fit_optima("AdamW", load("P1a"), rng)
    hyperball = fit_optima("Hyperball", load("P1d"), rng)
    adamw_large = fit_optima("AdamW P1(b)", load("P1b"), rng)
    adamw_law = fit_scaling(adamw, ADAMW_PREDICTION_TOKENS)
    hyperball_law = fit_scaling(hyperball, HYPERBALL_PREDICTION_TOKENS)
    for fit, law in ((adamw, adamw_law), (hyperball, hyperball_law)):
        report_optima(fit)
        report_scaling(law)
    report_optima(adamw_large)
    print()
    report_predictions(adamw_law, adamw_large)
    all_six = combine("All six budgets", adamw, adamw_large)
    all_six_law = fit_scaling(all_six, TARGET_TOKENS)
    larger_law = fit_scaling(replace(adamw_large, label="Larger three"), TARGET_TOKENS)
    for law in (all_six_law, larger_law):
        print(f"== {law.label} ==")
        report_scaling(law)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    plot_curves(ax, adamw)
    label_measured_lrs(ax, adamw)
    save(fig, "p1a_lr_loss_curves.png")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    plot_optima(ax, adamw, adamw_law, "C0", adamw.tokens.max() * 1.2, show_predictions=False)
    label_token_budgets(ax, adamw.tokens)
    ax.set_yticks([.0024, .0027, .003, .0033], ["0.0024", "0.0027", "0.0030", "0.0033"])
    ax.minorticks_off()
    ax.set_title("AdamW: fitted optimal LR vs token budget")
    ax.legend()
    save(fig, "p1a_optimal_lr_scaling.png")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    plot_curves(ax, adamw_large)
    label_measured_lrs(ax, adamw_large)
    save(fig, "p1b_lr_loss_curves.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_optima(ax, adamw, adamw_law, "C0", max(ADAMW_PREDICTION_TOKENS) * 1.2)
    plot_fitted_optima(ax, adamw_large, "C1", "P1(b) fitted optimum (±1σ)")
    label_token_budgets(ax, np.concatenate([adamw.tokens, adamw_large.tokens]), rotation=30)
    ax.set_yticks([.0025, .003, .0035, .004, .0045], ["0.0025", "0.0030", "0.0035", "0.0040", "0.0045"])
    ax.minorticks_off()
    ax.set_title("AdamW: predicted vs fitted optimal LR")
    ax.legend(fontsize=9)
    save(fig, "p1b_predicted_vs_fitted.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_fitted_optima(ax, adamw, "C0", "P1(a) fitted optimum (±1σ)")
    plot_fitted_optima(ax, adamw_large, "C1", "P1(b) fitted optimum (±1σ)")
    plot_law(ax, all_six_law, "C2", all_six.tokens.min() / 1.2, TARGET_TOKENS[0] * 1.2)
    plot_law(ax, larger_law, "C3", adamw_large.tokens.min() / 1.2, TARGET_TOKENS[0] * 1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training tokens D")
    ax.set_ylabel("optimal peak LR")
    label_token_budgets(ax, np.concatenate([all_six.tokens, TARGET_TOKENS]), rotation=30)
    ax.set_yticks([.0015, .002, .0025, .003, .0035, .004, .0045],
                  ["0.0015", "0.0020", "0.0025", "0.0030", "0.0035", "0.0040", "0.0045"])
    ax.minorticks_off()
    ax.set_title("AdamW: two power-law fits and their 4.9152B predictions")
    ax.legend(fontsize=9)
    save(fig, "p1c_two_rules.png")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_curves(left, hyperball)
    plot_optima(right, hyperball, hyperball_law, "C3", max(HYPERBALL_PREDICTION_TOKENS) * 1.2)
    plot_optima(right, adamw, adamw_law, "C0", max(HYPERBALL_PREDICTION_TOKENS) * 1.2,
                show_predictions=False)
    right.set_title("Optimal LR vs token budget")
    right.legend(fontsize=8)
    save(fig, "p1d_hyperball_vs_adamw.png")


if __name__ == "__main__":
    main()
