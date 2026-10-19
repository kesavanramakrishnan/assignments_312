"""A3 P1: mode-connectivity plots and tables from the measured (x, y) grid."""
import json
import re
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "experiments" / "a3_optimization" / "results" / "a3-mode-connectivity.json"
OUTPUT_DIR = ROOT / "notes" / "a3" / "p1"
X_VALUES = (0, 50, 100, 250, 500, 1000)
Y_VALUES = (25, 50, 100, 250, 500)
THRESHOLD = 0.01
REPRESENTATIVE = ((0, 100), (0, 500), (50, 250), (100, 500), (250, 500), (1000, 500))


def load():
    """cells[(x, y)] -> dict(endpoints, interpolation, ensemble, barrier)."""
    cells = {}
    for row in json.loads(RESULTS.read_text()):
        match = re.search(r"fork(\d+)", row["run_a"])
        x = int(match.group(1)) if match else 0
        y = row["step_a"] - x
        losses = row["interpolation"]
        endpoints = (losses[0], losses[-1])
        mean_endpoint = sum(endpoints) / 2
        cells[(x, y)] = dict(endpoints=endpoints, mean_endpoint=mean_endpoint, interpolation=losses,
                             alphas=row["alphas"], midpoint=losses[len(losses) // 2],
                             ensemble=row["ensemble"], barrier=losses[len(losses) // 2] - mean_endpoint)
    assert set(cells) == {(x, y) for x in X_VALUES for y in Y_VALUES}, sorted(cells)
    return cells


def grid(cells, key):
    return np.array([[cells[(x, y)][key] for y in Y_VALUES] for x in X_VALUES])


def heatmap(ax, values, title, fmt, cmap, norm=None, mark=None, ylabels=X_VALUES):
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(Y_VALUES)), Y_VALUES)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel("branch steps y")
    ax.set_ylabel("shared prefix steps x")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            r, g, b, _ = image.cmap(image.norm(values[i, j]))
            dark = .299 * r + .587 * g + .114 * b < .5
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=7,
                    color="white" if dark else "black",
                    fontweight="bold" if mark is not None and mark[i, j] else None)
            if mark is not None and mark[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.6))
    return image


def plot_endpoint_ensemble(cells):
    endpoint, ensemble = grid(cells, "mean_endpoint"), grid(cells, "ensemble")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    low, high = min(endpoint.min(), ensemble.min()), max(endpoint.max(), ensemble.max())
    for ax, values, title in ((axes[0], endpoint, "Endpoint loss (mean of two branches)"),
                              (axes[1], ensemble, "0.5/0.5 prediction-ensemble loss")):
        image = heatmap(ax, values, title, ".2f", "viridis_r")
        image.set_clim(low, high)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.03)
    gain = endpoint - ensemble
    image = heatmap(axes[2], gain, "Ensemble gain (endpoint − ensemble)", ".3f", "magma_r")
    fig.colorbar(image, ax=axes[2], fraction=.046, pad=.03)
    fig.savefig(OUTPUT_DIR / "p1a_endpoint_ensemble.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, .9, len(X_VALUES)))
    for color, x in zip(colors, X_VALUES):
        ys = np.array(Y_VALUES)
        axes[0].plot(x + ys, [cells[(x, y)]["mean_endpoint"] for y in Y_VALUES], "o-", color=color, label=f"x = {x}")
        axes[1].plot(x + ys, [cells[(x, y)]["ensemble"] for y in Y_VALUES], "o-", color=color, label=f"x = {x}")
    for ax, title in zip(axes, ("Endpoint loss", "Ensemble loss")):
        ax.set_xscale("log")
        ax.set_xlabel("total updates x + y")
        ax.set_title(title)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("validation loss")
    axes[0].legend(ncol=2)
    fig.savefig(OUTPUT_DIR / "p1a_loss_vs_updates.png", dpi=150)
    plt.close(fig)


def plot_barrier(cells):
    barrier = grid(cells, "barrier")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.12, vmax=0.25)
    image = heatmap(ax, barrier, f"Midpoint barrier B(x, y); boxed: B ≤ {THRESHOLD}", ".3f", "RdBu_r", norm,
                    mark=barrier <= THRESHOLD)
    fig.colorbar(image, ax=ax, label="B (validation loss)")
    fig.savefig(OUTPUT_DIR / "p1b_barrier_heatmap.png", dpi=150)
    plt.close(fig)


def plot_interpolation(cells):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, key in zip(axes.flat, REPRESENTATIVE):
        cell = cells[key]
        ax.plot(cell["alphas"], cell["interpolation"], "o-", color="C0", label="weight interpolation")
        ax.axhline(cell["ensemble"], color="C3", ls="--", label="0.5/0.5 ensemble")
        ax.axhline(cell["mean_endpoint"], color="gray", ls=":", label="mean endpoint")
        ax.set_title(f"x = {key[0]}, y = {key[1]}  (B = {cell['barrier']:+.3f})")
        ax.set_xlabel("α")
        ax.set_ylabel("validation loss")
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / "p1c_interpolation.png", dpi=150)
    plt.close(fig)


def print_tables(cells):
    print("| x | y | endpoint 1 | endpoint 2 | average | ensemble | B(x, y) | B ≤ 0.01 |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for x in X_VALUES:
        for y in Y_VALUES:
            c = cells[(x, y)]
            print(f"| {x} | {y} | {c['endpoints'][0]:.4f} | {c['endpoints'][1]:.4f} | {c['midpoint']:.4f} | "
                  f"{c['ensemble']:.4f} | {c['barrier']:+.4f} | {'yes' if c['barrier'] <= THRESHOLD else 'no'} |")
    print()
    print("| x | y | " + " | ".join(f"α = {a}" for a in cells[REPRESENTATIVE[0]]["alphas"]) + " | ensemble |")
    print("| ---: | ---: | " + " | ".join("---:" for _ in cells[REPRESENTATIVE[0]]["alphas"]) + " | ---: |")
    for key in REPRESENTATIVE:
        c = cells[key]
        print(f"| {key[0]} | {key[1]} | " + " | ".join(f"{v:.4f}" for v in c["interpolation"]) + f" | {c['ensemble']:.4f} |")


D_X_VALUES = (0, 50, 100, 500)
VARIANT_LABELS = {"baseline": "baseline (LR 3e-3, B 64, constant)", "lr1e-3": "LR 1e-3", "lr6e-3": "LR 6e-3",
                  "bs16": "batch size 16", "bs128": "batch size 128", "cos1000": "cosine over 1,000 updates"}


def load_variant(name):
    if name == "baseline":
        return {k: v for k, v in load().items() if k[0] in D_X_VALUES}
    path = RESULTS.with_name(f"a3-mode-connectivity-d-{name}.json")
    cells = {}
    for row in json.loads(path.read_text()):
        match = re.search(r"fork(\d+)", row["run_a"])
        x = int(match.group(1)) if match else 0
        losses = row["interpolation"]
        mean_endpoint = (losses[0] + losses[-1]) / 2
        cells[(x, row["step_a"] - x)] = dict(mean_endpoint=mean_endpoint, midpoint=losses[2], ensemble=row["ensemble"],
                                              interpolation=losses, alphas=row["alphas"],
                                              barrier=losses[2] - mean_endpoint)
    return cells


def plot_variants():
    variants = {name: load_variant(name) for name in VARIANT_LABELS}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.12, vmax=0.5)
    for ax, (name, cells) in zip(axes.flat, variants.items()):
        values = np.array([[cells[(x, y)]["barrier"] for y in Y_VALUES] for x in D_X_VALUES])
        heatmap(ax, values, VARIANT_LABELS[name], ".2f", "RdBu_r", norm, mark=values <= THRESHOLD, ylabels=D_X_VALUES)
    fig.suptitle("Midpoint barrier B(x, y); boxed: B ≤ 0.01")
    fig.savefig(OUTPUT_DIR / "p1d_barrier_variants.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = dict(zip(VARIANT_LABELS, ("k", "C0", "C3", "C2", "C4", "C1")))
    for ax, key in zip(axes, ((0, 500), (100, 500), (500, 500))):
        for name, cells in variants.items():
            cell = cells[key]
            ax.plot(cell["alphas"], np.array(cell["interpolation"]) - cell["mean_endpoint"], "o-", ms=3,
                    color=colors[name], label=VARIANT_LABELS[name] if key == (0, 500) else None)
        ax.axhline(0, color="gray", lw=.8)
        ax.set_title(f"x = {key[0]}, y = {key[1]}")
        ax.set_xlabel("α")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("loss − mean endpoint loss")
    axes[0].legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / "p1d_interpolation.png", dpi=150)
    plt.close(fig)

    print("| Variant | B(0, 500) | B(100, 500) | B(500, 500) | cells with B > 0.01 | most negative B | ensemble gain (x = 500, y = 500) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, cells in variants.items():
        failing = sorted(k for k, v in cells.items() if v["barrier"] > THRESHOLD)
        most = min(v["barrier"] for v in cells.values())
        e = cells[(500, 500)]
        print(f"| {VARIANT_LABELS[name]} | {cells[(0, 500)]['barrier']:+.3f} | {cells[(100, 500)]['barrier']:+.3f} | "
              f"{cells[(500, 500)]['barrier']:+.3f} | {len(failing)} {failing} | {most:+.3f} | {e['mean_endpoint'] - e['ensemble']:.3f} |")


def main():
    if "--variants" in __import__("sys").argv:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_variants()
        return
    cells = load()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_endpoint_ensemble(cells)
    plot_barrier(cells)
    plot_interpolation(cells)
    print_tables(cells)


if __name__ == "__main__":
    main()
