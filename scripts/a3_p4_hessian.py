"""A3 P4: Hessian analysis plots and tables from the fetched measurement results."""
import sys
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization import hessian as hs
from experiments.a3_optimization.hessian_checkpoints import STEPS, TOKENS_PER_STEP
from experiments.a3_optimization.hessian_jobs import RESULTS_DIR, load_result


ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "notes" / "a3" / "p4"
FAMILIES = ("embedding", "attention", "qk_norm", "mlp", "norm", "head")
FAMILY_COLORS = dict(zip(FAMILIES, ("C0", "C1", "C4", "C2", "C5", "C3")))
DENSITY_STEPS = (0, 153, 1526, 9375)


def tokens_m(step):
    return step * TOKENS_PER_STEP / 1e6


def have(name):
    return any((RESULTS_DIR / f"{name}{ext}").exists() for ext in (".json", ".jsonl"))


def style_token_axis(ax):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("training tokens (M)")
    ax.grid(alpha=.3)


def energy_matrix(result, part, key="energy"):
    """Mean over probes of the block-energy matrix, ||H_cb||_F^2, and its block names."""
    block = result[key][part]
    return block["blocks"], np.array(block["per_probe"]).mean(0)


def block_sizes(result, part):
    label = {"family": hs.family_of, "layer": hs.layer_of}[part]
    blocks = hs.partition(result["names"], result["sizes"], label)
    return {k: hs.block_size(v) for k, v in blocks.items()}


def part_a_spectrum():
    results = {step: load_result(f"a-adamw-{step}") for step in STEPS}
    tokens = [tokens_m(s) for s in STEPS]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for fam in FAMILIES:
        axes[0].plot(tokens, [results[s]["families"][fam]["lambda_max"] for s in STEPS], "o-", ms=3,
                     color=FAMILY_COLORS[fam], label=fam)
    axes[0].plot(tokens, [results[s]["lambda_max"] for s in STEPS], "ks--", ms=3, label="whole model")
    axes[0].set_ylabel("top eigenvalue λ_max")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    for fam in FAMILIES:
        values = []
        for s in STEPS:
            names, E = energy_matrix(results[s], "family")
            i = names.index(fam)
            values.append(E[i, i] / block_sizes(results[s], "family")[fam])
        axes[1].plot(tokens, values, "o-", ms=3, color=FAMILY_COLORS[fam], label=fam)
    axes[1].set_ylabel("tr(H_bb²) / size, per family block")
    axes[1].set_yscale("log")
    total = []
    for s in STEPS:
        names, E = energy_matrix(results[s], "family")
        total.append(E.sum() / results[s]["size"])
    axes[2].plot(tokens, total, "ks-", ms=3, label="tr(H²)/d")
    axes[2].plot(tokens, [results[s]["lambda_max"] ** 2 / results[s]["size"] for s in STEPS], "o--", ms=3, color="C3",
                 label="λ_max²/d")
    axes[2].plot(tokens, [abs(results[s]["lambda_min"]) for s in STEPS], "^:", ms=3, color="C0", label="|λ_min|")
    axes[2].set_ylabel("whole model")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=7)
    for ax in axes:
        style_token_axis(ax)
    fig.savefig(OUTPUT_DIR / "p4a_family_spectrum.png", dpi=150)
    plt.close(fig)

    print("| tokens | λ_max | λ_min | tr(H²)/d | sharpest family (λ_max) | λ_max² / tr(H²) |")
    for s in STEPS:
        r = results[s]
        fam = max(r["families"], key=lambda f: r["families"][f]["lambda_max"])
        names, E = energy_matrix(r, "family")
        trh2 = E.sum()
        print(f"| {tokens_m(s):.1f}M | {r['lambda_max']:.3f} | {r['lambda_min']:.3f} | {trh2 / r['size']:.3g} | "
              f"{fam} ({r['families'][fam]['lambda_max']:.3f}) | {r['lambda_max'] ** 2 / trh2:.3f} |")
    print()
    print("| tokens | " + " | ".join(FAMILIES) + " |")
    for s in STEPS:
        print(f"| {tokens_m(s):.1f}M | " + " | ".join(f"{results[s]['families'][f]['lambda_max']:.3f}" for f in FAMILIES) + " |")
    print()


def part_a_density():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    summary = []
    for ax, step in zip(axes, DENSITY_STEPS):
        r = load_result(f"a-adamw-{step}")
        runs = r["density_runs"]
        ritz = np.concatenate([d["ritz"] for d in runs])
        weights = np.concatenate([np.array(d["weights"]) / len(runs) for d in runs])
        lo, hi = min(ritz.min(), r["lambda_min"]), r["lambda_max"]
        edges = np.linspace(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo), 90)
        hist, edges = np.histogram(ritz, bins=edges, weights=weights)
        centers = (edges[1:] + edges[:-1]) / 2
        ax.step(centers, np.maximum(hist / np.diff(edges), 1e-12), where="mid", color="C0")
        ax.axvline(0, color="gray", lw=.8)
        ax.set_yscale("log")
        ax.set_xlabel("eigenvalue")
        ax.set_title(f"{tokens_m(step):.0f}M tokens")
        ax.grid(alpha=.3)
        negative = float(weights[ritz < 0].sum())
        bulk_edge = float(np.quantile(np.repeat(ritz, 1), 0.99))
        summary.append((tokens_m(step), r["lambda_min"], r["lambda_max"], negative, r["top"]["values"][:3]))
    axes[0].set_ylabel("spectral density (Lanczos quadrature)")
    fig.savefig(OUTPUT_DIR / "p4a_density.png", dpi=150)
    plt.close(fig)
    print("| tokens | λ_min | λ_max | mass at negative Ritz values | top-3 eigenvalues |")
    for t, lmin, lmax, neg, top in summary:
        print(f"| {t:.0f}M | {lmin:.3f} | {lmax:.3f} | {neg:.4f} | {', '.join(f'{v:.3f}' for v in top)} |")
    print()


def diag_fraction(result, part, key="energy"):
    names, E = energy_matrix(result, part, key)
    return float(np.trace(E) / E.sum())


def part_a_blocks():
    have_random = all(have(f"a-random-{s}") for s in STEPS)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    tokens = [tokens_m(s) for s in STEPS]
    for part, color in (("family", "C0"), ("layer", "C1")):
        real = [diag_fraction(load_result(f"a-adamw-{s}"), part) for s in STEPS]
        axes[0].plot(tokens, real, "o-", ms=3, color=color, label=f"{part} blocks")
        if have_random:
            rand = [diag_fraction(load_result(f"a-random-{s}"), part, "energy_random") for s in STEPS]
            axes[0].plot(tokens, rand, "x--", ms=4, color=color, label=f"random, same sizes ({part})")
    axes[0].set_ylabel("share of ||H||_F² inside blocks")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    style_token_axis(axes[0])
    for ax, part in zip(axes[1:], ("family", "layer")):
        r = load_result("a-adamw-9375")
        names, E = energy_matrix(r, part)
        image = ax.imshow(np.log10(E / E.sum() + 1e-12), cmap="viridis")
        ax.set_xticks(range(len(names)), names, rotation=60, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(f"log₁₀ share of ||H||_F² per {part} block pair (614M)")
        fig.colorbar(image, ax=ax, fraction=.046)
    fig.savefig(OUTPUT_DIR / "p4a_block_structure.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for step, color in zip((153, 1526, 4578, 9375), plt.cm.viridis(np.linspace(0, .9, 4))):
        r = load_result(f"a-adamw-{step}")
        names, E = energy_matrix(r, "layer")
        sizes = block_sizes(r, "layer")
        layers = [n for n in names if n.startswith("L")]
        ax.plot(range(len(layers)), [E[names.index(n), names.index(n)] / sizes[n] for n in layers], "o-", ms=3,
                color=color, label=f"{tokens_m(step):.0f}M tokens")
    ax.set_xticks(range(8), [f"L{i}" for i in range(8)])
    ax.set_ylabel("tr(H_ll²) / size, per layer block")
    ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / "p4a_depth.png", dpi=150)
    plt.close(fig)

    print("| tokens | family blocks: share inside | layer blocks: share inside | random family | random layer |")
    for s in STEPS:
        r = load_result(f"a-adamw-{s}")
        rf = rl = float("nan")
        if have_random:
            rr = load_result(f"a-random-{s}")
            rf, rl = diag_fraction(rr, "family", "energy_random"), diag_fraction(rr, "layer", "energy_random")
        print(f"| {tokens_m(s):.1f}M | {diag_fraction(r, 'family'):.3f} | {diag_fraction(r, 'layer'):.3f} | {rf:.3f} | {rl:.3f} |")
    print()


RESCALES = ("reference", "head-x0.25", "head-x0.5", "head-x4", "mlp-x0.25", "vo-x0.25", "qk-x0.25")
HEAD_RUNS = {"head0.25": 0.25, "head0.0625": 0.0625, "head4": 4.0}


def part_b(val_losses):
    print("Manual rescalings of the final AdamW checkpoint (probe loss, whole-model λ_max, per-family λ_max):")
    print("| Rescaling | Probe loss | λ_max | × reference | " + " | ".join(FAMILIES) + " |")
    ref = load_result("b-rescale-reference")
    names, values = [], []
    for name in RESCALES:
        r = load_result(f"b-rescale-{name}")
        print(f"| {name} | {r['probe_loss']:.4f} | {r['lambda_max']:.3f} | {r['lambda_max'] / ref['lambda_max']:.2f} | "
              + " | ".join(f"{r['families'][f]['lambda_max']:.3f}" for f in FAMILIES) + " |")
        names.append(name)
        values.append(r["lambda_max"])
    print()
    print("Training with head_init_scale (probe loss and λ_max at the final checkpoint):")
    print("| head_init_scale | Final val loss | Probe loss | λ_max | × reference | head λ_max |")
    rows = []
    for tag, scale in HEAD_RUNS.items():
        name = f"b-final-model-d8-lr0.003-tok614M-{tag}"
        if not have(name):
            continue
        r = load_result(name)
        val = val_losses.get(tag)
        rows.append((scale, r))
        print(f"| {scale:g} | {val if val is None else f'{val:.4f}'} | {r['probe_loss']:.4f} | {r['lambda_max']:.3f} | "
              f"{r['lambda_max'] / ref['lambda_max']:.2f} | {r['families']['head']['lambda_max']:.3f} |")
    print()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = ["gray"] + ["C0"] * (len(names) - 1)
    axes[0].bar(range(len(names)), values, color=colors)
    axes[0].set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("λ_max (probe loss unchanged at 2.9474)")
    axes[0].set_title("Function-preserving rescalings of the final checkpoint")
    axes[0].grid(alpha=.3, axis="y")
    if rows:
        scales = [s for s, _ in rows]
        axes[1].plot(scales, [r["lambda_max"] for _, r in rows], "o", color="C3", label="trained with head_init_scale")
        axes[1].axhline(ref["lambda_max"], color="k", ls="--", label="default recipe (final checkpoint)")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("head_init_scale")
        axes[1].set_ylabel("final λ_max")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=.3)
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(OUTPUT_DIR / "p4b_rescaling.png", dpi=150)
    plt.close(fig)


ARM_COLORS = {"full": "k", "top": "C3", "removed": "C0"}
VARIANTS = (
    ("Top-k only", ("c-top", "c-top-lr0.25", "c-top-k3", "c-top-k40", "c-top-start305")),
    ("Top-k removed", ("c-removed", "c-removed-lr2", "c-removed-k40", "c-removed-start305")),
)


def records(name):
    rec = load_result(name)
    return ([r for r in rec if r["kind"] == "start"][0], [r for r in rec if r["kind"] == "refresh"],
            [r for r in rec if r["kind"] == "step"])


def part_c():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for arm in ("full", "top", "removed"):
        start, refresh, steps = records(f"c-{arm}")
        color = ARM_COLORS[arm]
        val = [(r["step"], r["val_loss"]) for r in steps if r.get("val_loss") is not None]
        axes[0, 0].plot([start["step"]] + [s for s, _ in val], [start["val_loss"]] + [v for _, v in val], "o-", ms=3,
                        color=color, label=arm)
        axes[0, 1].plot([r["step"] for r in refresh], [r["lambda_max"] for r in refresh], "o-", ms=3, color=color, label=arm)
        axes[1, 0].plot([r["step"] for r in steps], [r["top_fraction"] for r in steps], color=color, label=arm, lw=.8)
        if arm == "full":
            axes[0, 1].plot([r["step"] for r in refresh], [r["heavy_ball_threshold"] for r in refresh], "k--",
                            label="heavy-ball threshold 2(1+β)/η_t")
        axes[1, 1].plot([r["step"] for r in steps], [r["gradient_norm"] for r in steps], color=color, label=arm, lw=.8)
    axes[0, 0].set_ylabel("validation loss")
    axes[0, 0].set_ylim(3.4, 5.0)
    axes[0, 1].set_ylabel("λ_max (refreshed every 25 steps)")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_ylabel("fraction of gradient in top-10 subspace")
    axes[1, 1].set_ylabel("gradient norm")
    axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("update")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / "p4c_arms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (title, names) in zip(axes, VARIANTS):
        for name, color in zip(names, plt.cm.plasma(np.linspace(0, .85, len(names)))):
            if not have(name):
                continue
            start, _, steps = records(name)
            val = [(r["step"] - start["step"], r["val_loss"]) for r in steps if r.get("val_loss") is not None]
            ax.plot([0] + [s for s, _ in val], [start["val_loss"]] + [v for _, v in val], "o-", ms=3, color=color,
                    label=name.removeprefix("c-"))
        start, _, steps = records("c-full")
        ax.plot([0] + [r["step"] - start["step"] for r in steps if r.get("val_loss")],
                [start["val_loss"]] + [r["val_loss"] for r in steps if r.get("val_loss")], "k--", label="full (control)")
        ax.set_title(title)
        ax.set_xlabel("updates after the start")
        ax.set_ylim(3.0, 5.5)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("validation loss")
    fig.savefig(OUTPUT_DIR / "p4c_variants.png", dpi=150)
    plt.close(fig)

    print("| Job | start step | start val | final val | Δ val | mean top-k gradient fraction | final λ_max / threshold |")
    for name in ["c-full", "c-top", "c-removed"] + [n for _, group in VARIANTS for n in group if n not in ("c-top", "c-removed")]:
        if not have(name):
            continue
        start, refresh, steps = records(name)
        val = [r["val_loss"] for r in steps if r.get("val_loss") is not None]
        ratio = refresh[-1]["lambda_max"] / refresh[-1]["heavy_ball_threshold"]
        print(f"| {name} | {start['step']} | {start['val_loss']:.3f} | {val[-1]:.3f} | {val[-1] - start['val_loss']:+.3f} | "
              f"{np.mean([r['top_fraction'] for r in steps]):.3f} | {ratio:.2f} |")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_losses = {"head0.25": 2.9257, "head0.0625": 2.9602, "head4": 2.9320}
    sections = sys.argv[1:] or ["a", "b", "c"]
    if "a" in sections:
        part_a_spectrum()
        part_a_density()
        part_a_blocks()
    if "b" in sections:
        part_b(val_losses)
    if "c" in sections:
        part_c()


if __name__ == "__main__":
    main()
