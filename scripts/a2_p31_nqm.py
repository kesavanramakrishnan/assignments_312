"""A2 P3.1: noisy quadratic model, optimal LR vs batch size (CPU)."""
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).parents[1] / "notes" / "a2" / "p3"
CACHE_DIR = Path(__file__).parents[1] / "tmp" / "a2_p31_cache"
TOTAL_EXAMPLES = 8192
CURVATURES = np.array([1., 10.])
# N(0, 1) and N(0, .1), read as variances.
INIT_VARIANCES = np.array([1., .1])
NUM_SAMPLES = 4096
ADAM_BETA2 = .95
ADAM_EPS = 1e-8
SGD_LR_GRID = np.geomspace(1e-4, .4, 100)
ADAPTIVE_LR_GRID = np.geomspace(1e-5, 1, 120)
SOURCE_BATCHES = (1, 2, 4, 8, 16, 32, 64)
TEST_BATCHES = (128, 256, 512)
PREDICTED_BATCH = 256
SGD_STABILITY_LIMIT = 2 / CURVATURES.max()
OPTIMIZERS = {  # name -> (simulate kwargs, LR grid)
    "SGD": (dict(optimizer="sgd"), SGD_LR_GRID),
    "RMSProp": (dict(optimizer="adam", beta1=0.), ADAPTIVE_LR_GRID),
    "Adam": (dict(optimizer="adam", beta1=.9), ADAPTIVE_LR_GRID),
}
COLORS = {"SGD": "C0", "RMSProp": "C1", "Adam": "C2"}
SIGMAS = (1., 10., 100., 300.)
SCALAR_MODEL = dict(curvatures=np.array([1.]), init_variances=np.array([2.]))
# Adam-type updates are normalized by the gradient scale, so the best LR grows with sigma.
WIDE_LR_GRID = np.geomspace(1e-5, 10, 150)


def simulate(optimizer, batch, lrs, *, curvatures=CURVATURES, init_variances=INIT_VARIANCES, noise_scale=1.,
             beta1=.9, momentum=0., num_samples=NUM_SAMPLES, seed=0, record_curve=False):
    """Mean final loss at each LR; every LR sees the same initializations and gradient noise."""
    rng = np.random.default_rng(seed)
    lrs = np.asarray(lrs, float)[:, None, None]
    dim = len(curvatures)
    w = np.broadcast_to(rng.normal(size=(num_samples, dim)) * np.sqrt(init_variances),
                        (len(lrs), num_samples, dim)).copy()
    m, v = np.zeros_like(w), np.zeros_like(w)
    noise_sd = noise_scale / np.sqrt(batch)
    curve = []
    with np.errstate(all="ignore"):
        for t in range(1, TOTAL_EXAMPLES // batch + 1):
            g = curvatures * w + noise_sd * rng.normal(size=(num_samples, dim))
            if optimizer == "sgd":
                m = momentum * m + g
                w -= lrs * m
            else:
                m = beta1 * m + (1 - beta1) * g
                v = ADAM_BETA2 * v + (1 - ADAM_BETA2) * g * g
                w -= lrs * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - ADAM_BETA2**t)) + ADAM_EPS)
            if record_curve:
                curve.append(mean_loss(w, curvatures))
    final = mean_loss(w, curvatures)
    return (final, np.array(curve)) if record_curve else final


def mean_loss(w, curvatures):
    loss = (.5 * curvatures * w * w).sum(axis=-1).mean(axis=-1)
    return np.where(np.isfinite(loss), loss, np.inf)


def cached_simulate(optimizer, batch, lrs, **kwargs):
    """simulate() with results stored on disk, keyed by every input."""
    key = repr((optimizer, batch, np.round(lrs, 12).tolist(), sorted(kwargs.items()), TOTAL_EXAMPLES,
                CURVATURES.tolist(), INIT_VARIANCES.tolist(), NUM_SAMPLES))
    path = CACHE_DIR / (hashlib.md5(key.encode()).hexdigest() + ".npy")
    if path.exists():
        return np.load(path)
    result = simulate(optimizer, batch, lrs, **kwargs)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, result)
    return result


def exact_sgd_loss(batch, lrs, noise_scale=1.):
    """Closed-form E[f(w_T)] for plain SGD, used to check the simulator."""
    steps = TOTAL_EXAMPLES // batch
    lrs = np.asarray(lrs, float)[:, None]
    decay = (1 - lrs * CURVATURES)**2
    variance = decay**steps * INIT_VARIANCES + lrs**2 * noise_scale**2 / batch * (1 - decay**steps) / (1 - decay)
    return (.5 * CURVATURES * variance).sum(axis=-1)


def fitted_optimum(lrs, losses):
    """Lowest grid point, refined by a parabola in log LR through it and its neighbours."""
    i = int(np.argmin(losses))
    if 0 < i < len(lrs) - 1 and np.all(np.isfinite(losses[i - 1:i + 2])):
        a, b, c = np.polyfit(np.log(lrs[i - 1:i + 2]), losses[i - 1:i + 2], 2)
        if a > 0:
            return float(np.exp(-b / (2 * a))), float(losses[i])
    return float(lrs[i]), float(losses[i])


def fit_power_law(batches, lrs):
    exponent, log_coefficient = np.polyfit(np.log(batches), np.log(lrs), 1)
    return exponent, log_coefficient


@dataclass
class Sweep:
    name: str
    lr_grid: np.ndarray
    losses: dict
    optima: dict
    exponent: float
    log_coefficient: float
    predicted_lr: float
    loss_at_predicted: float

    def rule(self, batch):
        return float(np.exp(self.log_coefficient + self.exponent * np.log(batch)))


def run_sweep(name, lr_grid=None, **overrides):
    kwargs, default_grid = OPTIMIZERS[name]
    lr_grid = default_grid if lr_grid is None else lr_grid
    kwargs = {**kwargs, **overrides}
    optimizer = kwargs.pop("optimizer")
    losses = {b: cached_simulate(optimizer, b, lr_grid, **kwargs) for b in SOURCE_BATCHES + TEST_BATCHES}
    optima = {b: fitted_optimum(lr_grid, losses[b]) for b in losses}
    exponent, log_coefficient = fit_power_law(SOURCE_BATCHES, [optima[b][0] for b in SOURCE_BATCHES])
    predicted = float(np.exp(log_coefficient + exponent * np.log(PREDICTED_BATCH)))
    at_predicted = cached_simulate(optimizer, PREDICTED_BATCH, np.append(lr_grid, predicted), **kwargs)[-1]
    return Sweep(name, lr_grid, losses, optima, exponent, log_coefficient, predicted, float(at_predicted))


def report(sweep):
    print(f"\n{sweep.name}: optimal LR ∝ B^{sweep.exponent:.3f}; predicted at B={PREDICTED_BATCH}: "
          f"{sweep.predicted_lr:.5f}")
    for b, (lr, loss) in sweep.optima.items():
        print(f"B={b:4d}  steps={TOTAL_EXAMPLES // b:5d}  lr*={lr:.5f}  rule={sweep.rule(b):.5f}  "
              f"best loss={loss:.5f}")
    tuned = sweep.optima[PREDICTED_BATCH]
    print(f"B={PREDICTED_BATCH}: loss at predicted LR {sweep.loss_at_predicted:.5f}, "
          f"at tuned LR {tuned[1]:.5f} (η={tuned[0]:.5f}); predicted/tuned LR = {sweep.predicted_lr / tuned[0]:.2f}")


def save(fig, name):
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def plot_sweep(sweep, filename, stability_limit=None):
    batches = np.array(SOURCE_BATCHES + TEST_BATCHES)
    fig, (curve_ax, lr_ax, loss_ax) = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = plt.cm.viridis(np.linspace(0, .95, len(batches)))
    for color, b in zip(colors, batches):
        curve_ax.plot(sweep.lr_grid, sweep.losses[b], color=color, label=f"B = {b}")
        curve_ax.plot(*sweep.optima[b], "*", color=color, markersize=12, markeredgecolor="k")
    curve_ax.set_xscale("log")
    curve_ax.set_yscale("log")
    best = min(loss for _, loss in sweep.optima.values())
    curve_ax.set_ylim(best / 1.6, 3)
    curve_ax.set_xlabel("learning rate η")
    curve_ax.set_ylabel("expected final loss")
    curve_ax.set_title(f"{sweep.name}: loss vs LR at each batch size")
    curve_ax.legend(fontsize=8, ncol=2)

    source = np.array(SOURCE_BATCHES)
    line = np.geomspace(1, 512, 100)
    lr_ax.plot(line, [sweep.rule(b) for b in line], "--", color="C0",
               label=f"power law fit on B ≤ 64: ∝ B^{sweep.exponent:.2f}")
    lr_ax.plot(source, [sweep.optima[b][0] for b in source], "o", color="C0", label="fitted optimum (source)")
    lr_ax.plot(TEST_BATCHES, [sweep.optima[b][0] for b in TEST_BATCHES], "s", color="C3",
               label="fitted optimum (test)")
    lr_ax.plot(PREDICTED_BATCH, sweep.predicted_lr, "x", color="k", markersize=11, markeredgewidth=2,
               label="prediction at B = 256")
    if stability_limit:
        lr_ax.axhline(stability_limit, color="k", linestyle=":", linewidth=1, label="stability limit 2/10")
        curve_ax.axvline(stability_limit, color="k", linestyle=":", linewidth=1)
    lr_ax.set_xscale("log", base=2)
    lr_ax.set_yscale("log")
    lr_ax.set_xlabel("batch size B")
    lr_ax.set_ylabel("optimal learning rate")
    lr_ax.set_title(f"{sweep.name}: optimal LR vs batch size")
    lr_ax.legend(fontsize=8)

    loss_ax.plot(source, [sweep.optima[b][1] for b in source], "o-", color="C0", label="source")
    loss_ax.plot(TEST_BATCHES, [sweep.optima[b][1] for b in TEST_BATCHES], "s-", color="C3", label="test")
    loss_ax.set_xscale("log", base=2)
    loss_ax.set_yscale("log")
    loss_ax.set_xlabel("batch size B")
    loss_ax.set_ylabel("best expected final loss")
    loss_ax.set_title(f"{sweep.name}: best loss after tuning LR")
    loss_ax.legend(fontsize=8)
    save(fig, filename)


def plot_comparison(sweeps, filename):
    fig, (lr_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
    line = np.geomspace(1, 512, 100)
    for sweep in sweeps:
        color = COLORS[sweep.name]
        lr_ax.plot(line, [sweep.rule(b) for b in line], "--", color=color, alpha=.7,
                   label=f"{sweep.name} fit on B ≤ 64: ∝ B^{sweep.exponent:.2f}")
        lr_ax.plot(SOURCE_BATCHES, [sweep.optima[b][0] for b in SOURCE_BATCHES], "o", color=color)
        lr_ax.plot(TEST_BATCHES, [sweep.optima[b][0] for b in TEST_BATCHES], "s", color=color)
        loss_ax.plot(SOURCE_BATCHES + TEST_BATCHES, [sweep.optima[b][1] for b in SOURCE_BATCHES + TEST_BATCHES],
                     "o-", color=color, label=sweep.name)
    lr_ax.axhline(SGD_STABILITY_LIMIT, color="k", linestyle=":", linewidth=1, label="SGD stability limit")
    for ax, ylabel in ((lr_ax, "optimal learning rate"), (loss_ax, "best expected final loss")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.axvspan(96, 700, color="gray", alpha=.12)
        ax.set_xlabel("batch size B (shaded: test batches)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    lr_ax.set_title("Optimal LR vs batch size (circles: source, squares: test)")
    loss_ax.set_title("Best loss after tuning LR")
    save(fig, filename)


def part_a():
    sweep = run_sweep("SGD")
    check = exact_sgd_loss(4, SGD_LR_GRID)
    stable = SGD_LR_GRID < SGD_STABILITY_LIMIT
    print(f"simulator vs closed form, B=4, max relative error over stable LRs: "
          f"{np.max(np.abs(sweep.losses[4][stable] / check[stable] - 1)):.4f}")
    report(sweep)
    plot_sweep(sweep, "p31a_sgd.png", stability_limit=SGD_STABILITY_LIMIT)


def part_b():
    sweeps = [run_sweep(name) for name in ("SGD", "RMSProp", "Adam")]
    for sweep in sweeps[1:]:
        report(sweep)
        plot_sweep(sweep, f"p31b_{sweep.name.lower()}.png")
    plot_comparison(sweeps, "p31b_comparison.png")


def sigma_task(task):
    name, sigma, scalar = task
    overrides = dict(SCALAR_MODEL) if scalar else {}
    return run_sweep(name, lr_grid=WIDE_LR_GRID * sigma, noise_scale=sigma, **overrides)


def plot_sigma_summary(sweeps, tasks):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for col, (scalar, title) in enumerate(((False, "2D quadratic, H = diag(1, 10)"), (True, "scalar quadratic, f = w²/2"))):
        for name in ("RMSProp", "Adam"):
            mine = [(t, sweeps[t]) for t in tasks if t[0] == name and t[2] == scalar]
            sigmas = [t[1] for t, _ in mine]
            axes[0, col].plot(sigmas, [s.exponent for _, s in mine], "o-", color=COLORS[name], label=name)
            axes[1, col].plot(sigmas, [s.predicted_lr / s.optima[PREDICTED_BATCH][0] for _, s in mine], "o-",
                              color=COLORS[name], label=name)
        axes[0, col].axhline(.5, color="k", linestyle=":", linewidth=1, label="square-root rule")
        axes[0, col].axhline(1, color="k", linestyle="--", linewidth=1, label="linear rule")
        axes[1, col].axhline(1, color="k", linestyle="--", linewidth=1)
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("fitted exponent (B ≤ 64)")
        axes[1, col].set_ylabel("predicted ÷ tuned LR at B = 256")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("gradient-noise scale σ")
        axes[0, col].legend(fontsize=8)
    axes[0, 0].set_xscale("log")
    save(fig, "p31c_exponent_vs_sigma.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    colors = plt.cm.plasma(np.linspace(0, .85, len(SIGMAS)))
    for row, scalar in enumerate((False, True)):
        for col, name in enumerate(("RMSProp", "Adam")):
            ax = axes[row, col]
            for color, sigma in zip(colors, SIGMAS):
                sweep = sweeps[(name, sigma, scalar)]
                batches = SOURCE_BATCHES + TEST_BATCHES
                ax.plot(batches, [sweep.optima[b][0] for b in batches], "o-", color=color, label=f"σ = {sigma:g}")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{name}, {'scalar' if scalar else '2D'} quadratic")
            ax.set_ylabel("fitted optimal LR")
            ax.legend(fontsize=8)
            if row == 1:
                ax.set_xlabel("batch size B")
    save(fig, "p31c_optimal_lr_by_sigma.png")


def part_c():
    tasks = [(name, sigma, scalar) for scalar in (False, True) for sigma in SIGMAS for name in ("RMSProp", "Adam")]
    with ProcessPoolExecutor(max_workers=7) as pool:
        results = list(pool.map(sigma_task, tasks))
    sweeps = dict(zip(tasks, results))
    for (name, sigma, scalar), sweep in sweeps.items():
        sweep.name = f"{name}, σ={sigma:g}, {'scalar' if scalar else '2D'}"
        report(sweep)
        edge = [b for b, (lr, _) in sweep.optima.items() if lr in (sweep.lr_grid[0], sweep.lr_grid[-1])]
        if edge:
            print(f"  warning: optimum at the LR grid edge for B = {edge}")
    plot_sigma_summary(sweeps, tasks)


MOMENTUM_BATCHES = (16, 256)
ADAM_BETA1S = (0., .5, .8, .9, .95, .98, .99)


def best_lr(optimizer, batch, lr_grid, **kwargs):
    """(LR of the lowest grid point, its loss): the pair is measured, so curves at this LR end at this loss."""
    losses = cached_simulate(optimizer, batch, lr_grid, **kwargs)
    i = int(np.argmin(losses))
    return float(lr_grid[i]), float(losses[i])


def learning_curve(optimizer, batch, lr, **kwargs):
    """Mean loss after each update at one LR (same seed, so the same runs as the sweep)."""
    _, curve = simulate(optimizer, batch, [lr], record_curve=True, **kwargs)
    return np.arange(1, len(curve) + 1) * batch, curve[:, 0]


def part_d():
    sgd, adam = {}, {}
    for batch in MOMENTUM_BATCHES:
        for mu in (0., .9):
            sgd[(batch, mu)] = best_lr("sgd", batch, SGD_LR_GRID, momentum=mu)
        for beta1 in ADAM_BETA1S:
            adam[(batch, beta1)] = best_lr("adam", batch, ADAPTIVE_LR_GRID, beta1=beta1)
    print("SGD: batch, mu, tuned LR, best loss")
    for (batch, mu), (lr, loss) in sgd.items():
        print(f"  B={batch:3d} mu={mu:g}  lr*={lr:.5f}  loss={loss:.5f}")
    print("Adam: batch, beta1, tuned LR, best loss")
    for (batch, beta1), (lr, loss) in adam.items():
        print(f"  B={batch:3d} beta1={beta1:g}  lr*={lr:.5f}  loss={loss:.5f}")
    for batch in MOMENTUM_BATCHES:
        best_beta = min(ADAM_BETA1S, key=lambda b: adam[(batch, b)][1])
        print(f"B={batch}: SGD momentum benefit {sgd[(batch, 0.)][1] - sgd[(batch, .9)][1]:+.5f}; "
              f"Adam best beta1={best_beta:g}, benefit over beta1=0: {adam[(batch, 0.)][1] - adam[(batch, best_beta)][1]:+.5f} "
              f"(ratio {adam[(batch, 0.)][1] / adam[(batch, best_beta)][1]:.2f})")
    plot_momentum(sgd, adam)


def plot_momentum(sgd, adam):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    width = .35
    for k, mu in enumerate((0., .9)):
        axes[0, 0].bar(np.arange(len(MOMENTUM_BATCHES)) + (k - .5) * width, [sgd[(b, mu)][1] for b in MOMENTUM_BATCHES],
                       width, label=f"μ = {mu:g}")
    axes[0, 0].set_xticks(np.arange(len(MOMENTUM_BATCHES)), [f"B = {b}" for b in MOMENTUM_BATCHES])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("best expected final loss (LR tuned)")
    axes[0, 0].set_title("SGD with and without momentum")
    axes[0, 0].legend()
    for batch, color in zip(MOMENTUM_BATCHES, ("C0", "C3")):
        axes[0, 1].plot(ADAM_BETA1S, [adam[(batch, b)][1] for b in ADAM_BETA1S], "o-", color=color, label=f"B = {batch}")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("β₁")
    axes[0, 1].set_ylabel("best expected final loss (LR tuned)")
    axes[0, 1].set_title("Adam: best loss vs β₁ (β₂ = 0.95)")
    axes[0, 1].legend()
    for ax, batch in zip(axes[1], MOMENTUM_BATCHES):
        best_beta = min(ADAM_BETA1S, key=lambda b: adam[(batch, b)][1])
        for beta1, style in ((best_beta, "-"), (0., "--")):
            lr = adam[(batch, beta1)][0]
            x, y = learning_curve("adam", batch, lr, beta1=beta1)
            ax.plot(x, y, style, label=f"Adam β₁ = {beta1:g}, η = {lr:.4f}")
        for mu, style in ((.9, "-"), (0., "--")):
            lr = sgd[(batch, mu)][0]
            x, y = learning_curve("sgd", batch, lr, momentum=mu)
            ax.plot(x, y, style, label=f"SGD μ = {mu:g}, η = {lr:.4f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("examples seen")
        ax.set_ylabel("expected loss")
        ax.set_title(f"B = {batch}: learning curves at the tuned LRs")
        ax.legend(fontsize=8)
    save(fig, "p31d_momentum.png")


PARTS = {"a": part_a, "b": part_b, "c": part_c, "d": part_d}


def main():
    for part in sys.argv[1:] or PARTS:
        PARTS[part]()


if __name__ == "__main__":
    main()
