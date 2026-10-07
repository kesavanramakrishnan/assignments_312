"""A2 P2: joint LR-WD optima from local bivariate quadratics, and the product law in tokens."""
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.provided_sweeps import load
from scripts import a2_p1_lr_scaling as p1


OUTPUT_DIR = Path(__file__).parents[1] / "notes" / "a2" / "p2"
FIXED_LR = .003
TEST_TOKENS = 2.4576e9


def points_by_budget(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["tokens"]].append((row["learning_rate"], row["weight_decay"], row["final_val_loss"]))
    return {tokens: np.array(sorted(points)) for tokens, points in sorted(groups.items())}


def fit_quadratic(points):
    """Least-squares a+bx+cy+dx^2+exy+fy^2 with x=log(LR), y=log(WD); returns (a,b,c,d,e,f)."""
    x, y = np.log(points[:, 0]), np.log(points[:, 1])
    design = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    return np.linalg.lstsq(design, points[:, 2], rcond=None)[0]


def quadratic_value(coefficients, lr, wd):
    a, b, c, d, e, f = coefficients
    x, y = np.log(lr), np.log(wd)
    return a + b * x + c * y + d * x * x + e * x * y + f * y * y


def quadratic_optimum(coefficients):
    """Stationary point of the fitted surface as (LR, WD, loss); NaN unless it is a minimum."""
    a, b, c, d, e, f = coefficients
    hessian = np.array([[2 * d, e], [e, 2 * f]])
    if np.any(np.linalg.eigvalsh(hessian) <= 0):
        return np.nan, np.nan, np.nan
    x, y = np.linalg.solve(hessian, [-b, -c])
    return np.exp(x), np.exp(y), quadratic_value(coefficients, np.exp(x), np.exp(y))


def optimum_draws(points, rng):
    """Joint optima refit with seed noise added to every loss; draws without a minimum are dropped."""
    draws = []
    for _ in range(p1.NUM_NOISE_DRAWS):
        noisy = points.copy()
        noisy[:, 2] += rng.normal(0, p1.SEED_NOISE_SD, len(points))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            draws.append(quadratic_optimum(fit_quadratic(noisy))[:2])
    draws = np.array(draws)
    return draws[np.all(np.isfinite(draws), axis=1)]


def fit_power_law(tokens, values):
    exponent, log_coefficient = np.polyfit(np.log(tokens), np.log(values), 1)
    return exponent, log_coefficient


def lr_only_optima():
    """Problem 1's AdamW optima at WD .1 over the same four budgets."""
    budgets = {**p1.points_by_budget(load("P1a")), **p1.points_by_budget(load("P1b"))}
    return {tokens: p1.quadratic_optimum(*p1.fit_window(points).T)[0]
            for tokens, points in budgets.items() if tokens <= 1.2288e9}


def lr_only_best_losses():
    rows = load("P1a") + load("P1b")
    best = defaultdict(lambda: np.inf)
    for row in rows:
        if row["tokens"] <= 1.2288e9:
            best[row["tokens"]] = min(best[row["tokens"]], row["final_val_loss"])
    return dict(sorted(best.items()))


def save(fig, name, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def valley_slope(coefficients):
    """d log(WD) / d log(LR) along the fitted surface's flattest direction."""
    _, _, _, d, e, f = coefficients
    _, vectors = np.linalg.eigh(np.array([[2 * d, e], [e, 2 * f]]))
    x, y = vectors[:, 0]
    return y / x


def r_squared(tokens, values):
    """R^2 of a power law, measured in log-log space."""
    exponent, log_coefficient = fit_power_law(tokens, values)
    log_values = np.log(values)
    residuals = log_values - (log_coefficient + exponent * np.log(tokens))
    return 1 - np.sum(residuals**2) / np.sum((log_values - log_values.mean())**2)


def draw_product_lines(ax, lr_grid, optimum, slope):
    lr_star, wd_star, _ = optimum
    ax.plot(lr_grid, lr_star * wd_star / lr_grid, "--", color="white", linewidth=2,
            label="ηλ = η*λ* (slope −1)")
    ax.plot(lr_grid, wd_star * (lr_grid / lr_star)**slope, ":", color="red", linewidth=2,
            label="fitted valley direction")


def plot_contours(budgets, fits, optima, name, product_lines=False):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (tokens, points) in zip(axes.flat, budgets.items()):
        lrs, wds = np.unique(points[:, 0]), np.unique(points[:, 1])
        lr_grid = np.geomspace(lrs.min() / 1.6, lrs.max() * 1.6, 200)
        wd_grid = np.geomspace(wds.min() / 1.6, wds.max() * 1.6, 200)
        surface = quadratic_value(fits[tokens], *np.meshgrid(lr_grid, wd_grid))
        best = points[:, 2].min()
        levels = best + np.linspace(-0.01, 0.15, 17)
        filled = ax.contourf(lr_grid, wd_grid, surface, levels=levels, cmap="viridis_r", extend="both")
        ax.contour(lr_grid, wd_grid, surface, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
        fig.colorbar(filled, ax=ax, label="fitted final val loss")
        ax.scatter(points[:, 0], points[:, 1], c="white", edgecolors="k", zorder=3)
        for lr, wd, loss in points:
            ax.annotate(f"{loss:.4f}", (lr, wd), textcoords="offset points", xytext=(5, 5), fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        lr_star, wd_star, loss_star = optima[tokens]
        title = f"{p1.token_label(tokens)} tokens"
        optimum_label = f"fitted optimum ({lr_star:.5f}, {wd_star:.3f})"
        if product_lines:
            slope = valley_slope(fits[tokens])
            draw_product_lines(ax, lr_grid, optima[tokens], slope)
            title += f": η*λ* = {lr_star * wd_star:.2e}, valley slope {slope:.2f}"
            optimum_label = "fitted optimum"
        ax.plot(lr_star, wd_star, "*", color="red", markersize=16, markeredgecolor="k", label=optimum_label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lr_grid[0], lr_grid[-1])
        ax.set_ylim(wd_grid[0], wd_grid[-1])
        ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
        ax.set_yticks(wds, [f"{wd:g}" for wd in wds])
        ax.minorticks_off()
        ax.set_xlabel("peak learning rate")
        ax.set_ylabel("weight decay")
        ax.set_title(title, fontsize=10)
        if not product_lines:
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Local quadratic fits in (log LR, log WD); stars mark fitted optima")
    if product_lines:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                   facecolor="lightgray")
        save(fig, name, rect=(0, 0.04, 1, 1))
    else:
        save(fig, name)


def plot_errorbars(ax, tokens, values, draws, color, label):
    low, high = np.percentile(draws, [16, 84], axis=0)
    ax.errorbar(tokens, values, yerr=[values - low, high - values], fmt="o-", color=color, capsize=4,
                label=label)


def plot_power_law(ax, tokens, values, color, show_r2=False):
    exponent, log_coefficient = fit_power_law(tokens, values)
    line = np.geomspace(tokens.min() / 1.2, tokens.max() * 1.2, 100)
    label = rf"single power law $\propto D^{{{exponent:.2f}}}$"
    if show_r2:
        label += f", R² = {r_squared(tokens, values):.3f}"
    ax.plot(line, np.exp(log_coefficient) * line**exponent, "--", color=color, alpha=0.7, label=label)


def plot_scaling_comparison(tokens, lr_stars, wd_stars, draws):
    """Optimal LR, WD, and their product vs tokens, each with one power law."""
    series = (("optimal peak LR η*", lr_stars, draws[..., 0], "C0", [.0012, .0015, .002, .0025, .003, .004]),
              ("optimal weight decay λ*", wd_stars, draws[..., 1], "C2", [.1, .15, .2, .3, .5, .8, 1.2]),
              ("product η*λ*", lr_stars * wd_stars, draws[..., 0] * draws[..., 1], "C4",
               [3e-4, 4e-4, 6e-4, 8e-4, 1e-3, 1.5e-3, 2e-3]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (name, values, value_draws, color, ticks) in zip(axes, series):
        plot_errorbars(ax, tokens, values, value_draws, color, "fitted optimum (±1σ)")
        plot_power_law(ax, tokens, values, color, show_r2=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        p1.label_token_budgets(ax, tokens)
        ax.set_xlabel("training tokens D")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs token budget")
        ax.legend(fontsize=9)
    save(fig, "p2b_lr_wd_product_scaling.png")


def plot_optima_vs_tokens(tokens, lr_stars, wd_stars, draws, lr_only):
    fig, (lr_ax, wd_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_errorbars(lr_ax, tokens, lr_stars, draws[..., 0], "C0", "joint LR–WD optimum (±1σ)")
    plot_power_law(lr_ax, tokens, lr_stars, "C0")
    lr_ax.plot(tokens, [lr_only[t] for t in tokens], "s:", color="C1", label="Problem 1 optimum at WD .1")
    plot_errorbars(wd_ax, tokens, wd_stars, draws[..., 1], "C2", "joint LR–WD optimum (±1σ)")
    plot_power_law(wd_ax, tokens, wd_stars, "C2")
    for ax, name in ((lr_ax, "optimal peak LR"), (wd_ax, "optimal weight decay")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        p1.label_token_budgets(ax, tokens)
        ax.set_xlabel("training tokens D")
        ax.set_ylabel(name)
        ax.legend(fontsize=9)
    lr_ticks = [.0012, .0015, .002, .0025, .003, .004]
    lr_ax.set_yticks(lr_ticks, [f"{t:g}" for t in lr_ticks])
    wd_ticks = [.1, .15, .2, .3, .5, .8, 1.2]
    wd_ax.set_yticks(wd_ticks, [f"{t:g}" for t in wd_ticks])
    for ax in (lr_ax, wd_ax):
        ax.minorticks_off()
    lr_ax.set_title("Optimal LR vs token budget")
    wd_ax.set_title("Optimal WD vs token budget")
    save(fig, "p2a_optima_vs_tokens.png")


def plot_loss_comparison(tokens, joint_best, lr_only_best):
    fig, (loss_ax, gap_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
    loss_ax.plot(tokens, lr_only_best, "s-", color="C1", label="Problem 1: LR tuned, WD .1")
    loss_ax.plot(tokens, joint_best, "o-", color="C0", label="Problem 2: LR and WD tuned")
    loss_ax.set_ylabel("best measured final val loss")
    loss_ax.set_title("Best measured loss vs token budget")
    gap = lr_only_best - joint_best
    gap_ax.plot(tokens, gap, "o-", color="C3")
    for t, g in zip(tokens, gap):
        gap_ax.annotate(f"{g:.4f}", (t, g), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    gap_ax.set_ylim(-0.005, gap.max() * 1.15)
    gap_ax.axhline(0, color="k", linewidth=0.8)
    gap_ax.axhspan(-p1.SEED_NOISE_SD, p1.SEED_NOISE_SD, color="gray", alpha=0.2, label=r"$\pm1\sigma$ seed noise")
    gap_ax.set_ylabel("Problem 1 best − Problem 2 best")
    gap_ax.set_title("Benefit of also tuning WD")
    gap_ax.legend(fontsize=9)
    for ax in (loss_ax, gap_ax):
        ax.set_xscale("log")
        p1.label_token_budgets(ax, tokens)
        ax.set_xlabel("training tokens D")
    loss_ax.legend(fontsize=9)
    save(fig, "p2a_best_loss_vs_p1.png")


def main():
    rng = np.random.default_rng(0)
    budgets = points_by_budget(load("P2a"))
    fits = {tokens: fit_quadratic(points) for tokens, points in budgets.items()}
    optima = {tokens: quadratic_optimum(coefficients) for tokens, coefficients in fits.items()}
    draws = [optimum_draws(points, rng) for points in budgets.values()]
    count = min(len(d) for d in draws)
    draws = np.stack([d[:count] for d in draws], axis=1)
    low, high = np.percentile(draws, [16, 84], axis=0)
    for (tokens, (lr, wd, loss)), lo, hi in zip(optima.items(), low, high):
        print(f"{tokens / 1e6:7.1f}M  lr*={lr:.5f} [{lo[0]:.5f}, {hi[0]:.5f}]  "
              f"wd*={wd:.4f} [{lo[1]:.4f}, {hi[1]:.4f}]  lr*wd*={lr * wd:.3e}  L*={loss:.4f}")

    tokens = np.array(list(optima))
    lr_stars = np.array([lr for lr, _, _ in optima.values()])
    wd_stars = np.array([wd for _, wd, _ in optima.values()])
    for name, values in (("lr*", lr_stars), ("wd*", wd_stars)):
        print(f"single power law: {name} ∝ D^{fit_power_law(tokens, values)[0]:.3f}")
    products = lr_stars * wd_stars
    exponent, log_coefficient = fit_power_law(tokens, products)
    predicted_product = np.exp(log_coefficient + exponent * np.log(TEST_TOKENS))
    print(f"product law: lr*wd* ∝ D^{exponent:.3f}; at {TEST_TOKENS / 1e9:g}B -> {predicted_product:.3e}"
          f" -> WD {predicted_product / FIXED_LR:.4f} at LR {FIXED_LR}")

    smallest = budgets[min(budgets)]
    lr, wd, loss = smallest[smallest[:, 2].argmin()]
    print(f"best measured pair at {min(budgets) / 1e6:g}M: LR {lr:g}, WD {wd:g} (loss {loss:.4f})")

    lr_only = lr_only_optima()
    joint_best = np.array([points[:, 2].min() for points in budgets.values()])
    lr_only_best = np.array(list(lr_only_best_losses().values()))
    for t, j, s in zip(tokens, joint_best, lr_only_best):
        print(f"{t / 1e6:7.1f}M  best joint {j:.4f}  best P1 {s:.4f}  P1 - joint {s - j:+.4f}")

    print("\nfitted loss rise for a 2x change from the optimum: along ηλ = const vs one coordinate alone")
    for t, coefficients in fits.items():
        lr, wd, loss = optima[t]
        along = [quadratic_value(coefficients, lr * k, wd / k) - loss for k in (2, .5)]
        lr_only_moves = [quadratic_value(coefficients, lr * k, wd) - loss for k in (2, .5)]
        wd_only_moves = [quadratic_value(coefficients, lr, wd * k) - loss for k in (2, .5)]
        print(f"{t / 1e6:7.1f}M  valley slope {valley_slope(coefficients):+.2f}  "
              f"along product {along[0]:+.4f}/{along[1]:+.4f}  LR only {lr_only_moves[0]:+.4f}/{lr_only_moves[1]:+.4f}  "
              f"WD only {wd_only_moves[0]:+.4f}/{wd_only_moves[1]:+.4f}")

    print("\npower-law fits vs tokens (R² in log-log, ±1σ from seed-noise refits)")
    for name, values, value_draws in (("lr*", lr_stars, draws[..., 0]), ("wd*", wd_stars, draws[..., 1]),
                                      ("lr*wd*", products, draws[..., 0] * draws[..., 1])):
        r2_low, r2_high = np.percentile([r_squared(tokens, d) for d in value_draws], [16, 84])
        ratios = ", ".join(f"{r:.2f}" for r in values[1:] / values[:-1])
        print(f"{name:7s} exponent {fit_power_law(tokens, values)[0]:+.3f}  "
              f"R² {r_squared(tokens, values):.3f} [{r2_low:.3f}, {r2_high:.3f}]  per doubling {ratios}")

    plot_contours(budgets, fits, optima, "p2a_lr_wd_contours.png")
    plot_contours(budgets, fits, optima, "p2b_contours_product_line.png", product_lines=True)
    plot_optima_vs_tokens(tokens, lr_stars, wd_stars, draws, lr_only)
    plot_loss_comparison(tokens, joint_best, lr_only_best)
    plot_scaling_comparison(tokens, lr_stars, wd_stars, draws)


if __name__ == "__main__":
    main()
