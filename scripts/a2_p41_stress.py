"""A2 P4.1: LR transfer across width and depth in the five-update stress test, plus the feature and alignment probes."""
import json
import sys
from collections import defaultdict
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import a2_p1_lr_scaling as p1


OUTPUT_DIR = Path(__file__).parents[1] / "notes" / "a2" / "p4"
JSON_DIR = Path("/private/tmp/claude-501/-Users-kesavanramakrishnan-assignments-312/50cb7fd1-dc35-47c7-9e01-445054bb84e5/scratchpad/p41_json/a2-stress")
WIDTHS = (640, 2560, 5120)
DEPTHS = (2, 100, 1000)
WIDTH_POLICIES = ("kaiming", "mup")
DEPTH_POLICIES = ("mup", "depth-mup", "completep")
NAMES = {"kaiming": "Kaiming", "mup": "µP", "depth-mup": "Depth-µP", "completep": "CompleteP"}
MATRICES = ("q", "k", "v", "o", "gate", "up", "down")


def load_runs():
    runs = []
    for path in sorted(JSON_DIR.glob("p41-*.json")):
        stem = path.stem.split("-")
        kind = stem[1]
        d = json.load(open(path))
        policy = path.stem[len(f"p41-{kind}-"):].rsplit("-" + ("w" if kind == "width" else "d"), 1)[0]
        runs.append(dict(kind=kind, policy=policy, width=d["config"]["width"], depth=d["config"]["depth"],
                         lr=d["base_lr"], loss=d["history"][-1]["val_loss"], data=d, path=path.name))
    return runs


def group(runs, kind, key):
    out = defaultdict(list)
    for r in runs:
        if r["kind"] == kind:
            out[(r["policy"], r[key])].append(r)
    return {k: sorted(v, key=lambda r: r["lr"]) for k, v in out.items()}


def fit_curve(rs):
    """Parabola in log2 LR through the lowest sampled loss and its neighbours: (optimal LR, fitted minimum loss)."""
    points = np.array([(r["lr"], r["loss"]) for r in rs])
    window = p1.fit_window(points)
    lr, loss, _ = p1.quadratic_optimum(window[:, 0], window[:, 1])
    return lr, loss


def best_run(rs):
    return min(rs, key=lambda r: r["loss"])


def save(fig, name):
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def transfer_plot(groups, policies, xs, xlabel, filename, xticks=None):
    fig, axes = plt.subplots(2, len(policies), figsize=(5.2 * len(policies), 8.5))
    colors = plt.cm.viridis(np.linspace(0, .85, len(xs)))
    fits = {}
    for col, policy in enumerate(policies):
        ax = axes[0, col]
        for color, x in zip(colors, xs):
            rs = groups[(policy, x)]
            ax.plot([r["lr"] for r in rs], [r["loss"] for r in rs], "o-", color=color, label=f"{xlabel} {x}")
            lr, loss = fit_curve(rs)
            fits[(policy, x)] = (lr, loss)
            ax.plot(lr, loss, "*", color=color, markersize=14, markeredgecolor="k")
        ax.set_xscale("log")
        ax.set_ylim(top=min(ax.get_ylim()[1], 12))
        ax.set_xlabel("base LR")
        ax.set_ylabel("validation loss after 5 updates")
        ax.set_title(NAMES[policy])
        ax.legend(fontsize=8)
    series = {p: (np.array([fits[(p, x)][0] for x in xs]), np.array([fits[(p, x)][1] for x in xs])) for p in policies}
    for ax, idx, ylabel in ((axes[1, 0], 0, "fitted optimal base LR"), (axes[1, 1], 1, "fitted minimum validation loss")):
        for i, p in enumerate(policies):
            ax.plot(xs, series[p][idx], "o-", color=f"C{i}", label=NAMES[p])
        ax.set_xscale("log")
        if idx == 0:
            ax.set_yscale("log")
            values = np.concatenate([series[p][0] for p in policies])
            ticks = [t for t in (3e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2)
                     if values.min() / 1.4 <= t <= values.max() * 1.4]
            ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
            ax.set_ylim(values.min() / 1.3, values.max() * 1.3)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
    if len(policies) > 2:
        for i, p in enumerate(policies):
            axes[1, 2].axis("off") if i == 0 else None
    save(fig, filename)
    return fits


def part_a(runs):
    groups = group(runs, "width", "width")
    print("\n4.1(a) fitted optimal LR and minimum loss (and best sampled)")
    fits = transfer_plot(groups, WIDTH_POLICIES, WIDTHS, "width", "p41a_width_transfer.png")
    for policy in WIDTH_POLICIES:
        for w in WIDTHS:
            lr, loss = fits[(policy, w)]
            b = best_run(groups[(policy, w)])
            print(f"  {NAMES[policy]:9s} w={w:4d}: fitted LR {lr:.5f}, fitted min loss {loss:.3f}; best sampled LR {b['lr']:g} loss {b['loss']:.3f}")
    return groups


def part_c(runs):
    groups = group(runs, "depth", "depth")
    # depth 2: every depth rule coincides (r = 1), so the single sweep serves all three prescriptions
    for policy in DEPTH_POLICIES:
        groups[(policy, 2)] = groups[("mup", 2)]
    print("\n4.1(c) fitted optimal LR and minimum loss (and best sampled)")
    fits = transfer_plot(groups, DEPTH_POLICIES, DEPTHS, "depth", "p41c_depth_transfer.png")
    for policy in DEPTH_POLICIES:
        for d in DEPTHS:
            lr, loss = fits[(policy, d)]
            b = best_run(groups[(policy, d)])
            edge = "  [best at grid edge]" if b["lr"] in (groups[(policy, d)][0]["lr"], groups[(policy, d)][-1]["lr"]) else ""
            print(f"  {NAMES[policy]:9s} d={d:4d}: fitted LR {lr:.5f}, fitted min loss {loss:.3f}; best sampled LR {b['lr']:g} loss {b['loss']:.3f}{edge}")
    return groups


def step_values(run, key):
    return [h[key] for h in run["data"]["history"]]


def alpha_by_step(run, matrix):
    """alpha_upd of one matrix type averaged over the probed blocks, per update 1..5 (None if undefined)."""
    values = defaultdict(list)
    for rec in run["data"]["alignment"]:
        if rec["parameter"].endswith(f".{matrix}.weight") and rec.get("alpha") is not None:
            values[rec["step"]].append(rec["alpha"])
    return [np.mean(values[s]) if values[s] else np.nan for s in range(1, 6)]


def readout_alpha(run):
    values = {}
    for rec in run["data"]["alignment"]:
        if rec["parameter"].startswith("head") and rec.get("alpha") is not None:
            values[rec["step"]] = rec["alpha"]
    return [values.get(s, np.nan) for s in range(1, 6)]


def omega_by_step(run):
    out = []
    for h in run["data"]["history"]:
        o = h["readout_alignment"].get("movement", {}).get("omega")
        out.append(np.nan if o is None else o)
    return out


def part_b(groups):
    best = {(p, w): best_run(groups[(p, w)]) for p in WIDTH_POLICIES for w in WIDTHS}
    styles = {("kaiming", w): ("C3", ls) for w, ls in zip(WIDTHS, ("-", "--", ":"))}
    styles.update({("mup", w): ("C0", ls) for w, ls in zip(WIDTHS, ("-", "--", ":"))})
    steps = np.arange(6)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for key, run in best.items():
        label = f"{NAMES[key[0]]} w={key[1]} (LR {run['lr']:g})"
        color, ls = styles[key]
        axes[0].plot(steps, step_values(run, "logit_rms"), ls, color=color, marker="o", label=label)
        axes[1].plot(steps, [h["features"]["final_norm"]["rms"] for h in run["data"]["history"]], ls, color=color, marker="o")
        axes[2].plot(steps, [h["features"]["final_norm"]["movement"] for h in run["data"]["history"]], ls, color=color, marker="o")
        axes[3].plot(steps, omega_by_step(run), ls, color=color, marker="o")
    for ax, title in zip(axes, ("logit RMS", "final-norm output RMS", "final-norm feature movement M_t",
                                "readout alignment ω_move")):
        ax.set_xlabel("optimizer updates")
        ax.set_title(title, fontsize=10)
    axes[3].axhline(.5, color="k", linestyle=":", linewidth=1, label="no alignment (1/2)")
    axes[3].axhline(1., color="k", linestyle="--", linewidth=1, label="full alignment (1)")
    axes[0].legend(fontsize=7)
    axes[3].legend(fontsize=7)
    save(fig, "p41b_width_features.png")

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharex=True)
    for ax, matrix in zip(axes.flat, (*MATRICES, "readout")):
        for key, run in best.items():
            color, ls = styles[key]
            alpha = readout_alpha(run) if matrix == "readout" else alpha_by_step(run, matrix)
            ax.plot(np.arange(1, 6), alpha, ls, color=color, marker="o")
        ax.axhline(.5, color="k", linestyle=":", linewidth=1)
        ax.axhline(1., color="k", linestyle="--", linewidth=1)
        ax.set_title(f"α_upd: {matrix}", fontsize=10)
        ax.set_ylim(0, 1.15)
    for ax in axes[1]:
        ax.set_xlabel("optimizer update")
    save(fig, "p41b_width_alignment.png")

    print("\n4.1(b) at the best sampled LR: logit RMS, final-norm RMS, movement M_t, omega_move per step 0..5")
    for key, run in best.items():
        print(f"  {NAMES[key[0]]:8s} w={key[1]} LR {run['lr']:g}")
        print("    logit_rms ", np.round(step_values(run, "logit_rms"), 2))
        print("    norm rms  ", np.round([h['features']['final_norm']['rms'] for h in run['data']['history']], 2))
        print("    movement  ", np.round([h['features']['final_norm']['movement'] for h in run['data']['history']], 3))
        print("    omega_move", np.round(omega_by_step(run), 3))
        print("    alpha (mean over matrices, steps 1..5)", np.round(np.nanmean([alpha_by_step(run, m) for m in MATRICES], axis=0), 3),
              " readout", np.round(readout_alpha(run), 3))
        print("    alpha by matrix @ step 5:", {m: round(alpha_by_step(run, m)[-1], 2) for m in MATRICES})


def part_d(groups):
    best = {(p, d): best_run(groups[(p, d)]) for p in DEPTH_POLICIES for d in DEPTHS}
    colors = {"mup": "C0", "depth-mup": "C1", "completep": "C2"}
    styles = {2: "-", 100: "--", 1000: ":"}
    steps = np.arange(6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for (p, d), run in best.items():
        if d == 2 and p != "mup":
            continue
        color, ls = ("k" if d == 2 else colors[p]), styles[d]
        label = "all rules, depth 2" if d == 2 else f"{NAMES[p]}, depth {d}"
        h = run["data"]["history"]
        last = run["data"]["probe_block_indices"][-1]
        mid = run["data"]["probe_block_indices"][len(run["data"]["probe_block_indices"]) // 2]
        axes[0, 0].plot(steps, [x["residual_rms"]["final_norm"] for x in h], ls, color=color, marker="o", label=label)
        axes[0, 1].plot(steps, [x["unscaled_branch_rms"][f"blocks.{last}.down"] for x in h], ls, color=color, marker="o")
        axes[0, 2].plot(steps, [x["features"][f"blocks.{mid}.norm1"]["movement"] for x in h], ls, color=color, marker="o")
        axes[1, 0].plot(steps, [x["features"]["final_norm"]["movement"] for x in h], ls, color=color, marker="o")
        axes[1, 1].plot(steps, [x["features"]["final_norm"]["rms"] for x in h], ls, color=color, marker="o")
        axes[1, 2].plot(steps, omega_by_step(run), ls, color=color, marker="o")
    axes[1, 2].axhline(.5, color="k", linestyle=":", linewidth=1)
    axes[1, 2].axhline(1., color="k", linestyle="--", linewidth=1)
    for ax, title in zip(axes.flat, ("residual RMS before final norm", "last FFN branch RMS (before c_L)",
                                     "midpoint feature movement", "final-norm movement M_t",
                                     "final-norm output RMS", "ω_move")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("optimizer updates")
        ax.set_yscale("log") if "before final" in title or "branch" in title else None
    axes[0, 0].legend(fontsize=8)
    save(fig, "p41d_depth_probes.png")

    print("\n4.1(d) at the best sampled LR: residual RMS (final), branch RMS, M_t(final norm), M_t(mid), omega at step 5 and step 0 RMS")
    for (p, d), run in best.items():
        h = run["data"]["history"]
        last = run["data"]["probe_block_indices"][-1]
        mid = run["data"]["probe_block_indices"][len(run["data"]["probe_block_indices"]) // 2]
        print(f"  {NAMES[p]:9s} d={d:4d} LR {run['lr']:g}: resid RMS step0 {h[0]['residual_rms']['final_norm']:.3g} -> step5 {h[5]['residual_rms']['final_norm']:.3g}; "
              f"branch(last down) {h[5]['unscaled_branch_rms'][f'blocks.{last}.down']:.3g}; M_t final {h[5]['features']['final_norm']['movement']:.3f}; "
              f"M_t mid {h[5]['features'][f'blocks.{mid}.norm1']['movement']:.3f}; norm RMS {h[5]['features']['final_norm']['rms']:.2f}; "
              f"omega {np.round(omega_by_step(run)[1:], 2)}; logit RMS {h[5]['logit_rms']:.2f}; c_L {run['data']['residual_multipliers'][0]:.4f}")


def main():
    runs = load_runs()
    print(len(runs), "runs loaded")
    parts = sys.argv[1:] or ["a", "b", "c", "d"]
    width_groups = depth_groups = None
    if "a" in parts or "b" in parts:
        width_groups = part_a(runs)
    if "b" in parts:
        part_b(width_groups)
    if "c" in parts or "d" in parts:
        depth_groups = part_c(runs)
    if "d" in parts:
        part_d(depth_groups)


if __name__ == "__main__":
    main()
