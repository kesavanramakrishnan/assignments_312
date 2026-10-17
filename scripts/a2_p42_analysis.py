"""A2 P4.2: width/depth LR transfer at 153.6M tokens, held-out width 1024, and the update/readout alignment diagnostics."""
import json
import sys
from collections import defaultdict
from pathlib import Path

from scripts import _figstyle  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.a2.provided_sweeps import load, reference_diagnostics
from scripts import a2_p1_lr_scaling as p1
from scripts import a2_p42_transfer as t


OUTPUT_DIR = Path(__file__).parents[1] / "notes" / "a2" / "p4"
CACHE = Path("/private/tmp/claude-501/-Users-kesavanramakrishnan/assignments_312/50cb7fd1-dc35-47c7-9e01-445054bb84e5/scratchpad/p42".replace("kesavanramakrishnan/assignments_312", "kesavanramakrishnan-assignments-312"))
TOKENS = 153_600_000
SOURCE_LR = .003
NAMES = {"base": "Course baseline", "mup": "µP", "depth-mup": "Depth-µP", "completep": "CompleteP"}
MATRICES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def fetch_runs():
    """Every finished 4.2 run: prescription, width, depth, LR, final validation loss and W&B id."""
    runs = []
    for tag in ("a2-p42a", "a2-p42c", "a2-p42d"):
        for r in wandb.Api().runs(p1.WANDB_PROJECT, filters={"tags": tag, "state": "finished"}):
            m = r.config["model_config"]
            prescription = next(x for x in r.tags if x.startswith("p42-"))[4:]
            runs.append(dict(id=r.id, name=r.name, part=tag[-1], prescription=prescription, width=m["hidden_size"],
                             depth=m["num_hidden_layers"], lr=r.config["learning_rate"], loss=r.summary["val_loss"]))
    return runs


def run_files(run, names=("features.jsonl", "alignment.jsonl")):
    """Download (once) and parse the JSON-lines diagnostics of one run."""
    folder = CACHE / run["id"]
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    api_run = None
    for name in names:
        path = folder / name
        if not path.exists():
            api_run = api_run or wandb.Api().run(f"{p1.WANDB_PROJECT}/{run['id']}")
            api_run.file(name).download(root=str(folder), replace=True)
        out[name] = [json.loads(line) for line in open(path)]
    return out


def gradient_summary(run):
    """Pre-clip gradient norm and clip coefficient at updates 1 and 5 (the norm is logged at nine checkpoints)."""
    path = CACHE / run["id"] / "gradient_history.json"
    if not path.exists():
        rows = wandb.Api().run(f"{p1.WANDB_PROJECT}/{run['id']}").history(
            keys=["logging/gradient/pre_clip_norm", "logging/gradient/clip_coefficient"], samples=100000, pandas=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(path, "w"))
    rows = [r for r in json.load(open(path)) if r.get("logging/gradient/pre_clip_norm") is not None]
    first, fifth = rows[0], rows[4]
    return dict(norm1=first["logging/gradient/pre_clip_norm"], coef1=first["logging/gradient/clip_coefficient"],
                norm5=fifth["logging/gradient/pre_clip_norm"], coef5=fifth["logging/gradient/clip_coefficient"])


def sweeps_by(runs, prescription, key, value, parts):
    pts = sorted((r["lr"], r["loss"]) for r in runs if r["prescription"] == prescription and r[key] == value and r["part"] in parts)
    return np.array(pts)


def vertex(points):
    lr, loss, _ = p1.quadratic_optimum(*p1.fit_window(points).T)
    return lr, loss


def part_b(runs):
    """Fitted optimal LR and minimum loss vs width (128, 256, 512), and the loss at the transferred LR."""
    sweeps = t.load_sweeps()
    fits = {(p, w): vertex(sweeps[(p, w)]) for p in ("base", "mup") for w in (128, 256, 512)}
    transferred = {(p, w): float(dict(sweeps[(p, w)])[SOURCE_LR]) for p in ("base", "mup") for w in (128, 256, 512)}
    print("\n4.2(b): fitted optimal LR, fitted min loss, loss at transferred LR .003")
    for p in ("base", "mup"):
        for w in (128, 256, 512):
            print(f"  {NAMES[p]:15s} w={w:3d}: LR* {fits[(p, w)][0]:.5f}  min loss {fits[(p, w)][1]:.4f}  loss@.003 {transferred[(p, w)]:.4f}  "
                  f"(gap {transferred[(p, w)] - fits[(p, w)][1]:+.4f})")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    widths = (128, 256, 512)
    for i, p in enumerate(("base", "mup")):
        axes[0].plot(widths, [fits[(p, w)][0] for w in widths], "o-", color=f"C{i}", label=NAMES[p])
        axes[1].plot(widths, [fits[(p, w)][1] for w in widths], "o-", color=f"C{i}", label=f"{NAMES[p]}: fitted minimum")
        axes[1].plot(widths, [transferred[(p, w)] for w in widths], "s--", color=f"C{i}", alpha=.7, label=f"{NAMES[p]}: at LR 0.003")
        axes[2].plot(widths, [transferred[(p, w)] - fits[(p, w)][1] for w in widths], "o-", color=f"C{i}", label=NAMES[p])
    axes[0].axhline(SOURCE_LR, color="gray", linestyle=":", label="transferred source LR")
    axes[0].set_yscale("log")
    ticks = [.002, .003, .004, .006, .008]
    axes[0].set_yticks(ticks, [f"{x:g}" for x in ticks])
    for ax, ylabel in zip(axes, ("fitted optimal base LR", "validation loss", "loss at LR 0.003 − fitted minimum")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, [str(w) for w in widths])
        ax.minorticks_off()
        ax.set_xlabel("width")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    t.save(fig, "p42b_width_transfer.png")
    return fits, transferred


def part_c(runs, fits_small):
    """Held-out width 1024: predicted LR, transferred LR and the local sweep."""
    print("\n4.2(c): width 1024")
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, p in zip(axes, ("base", "mup")):
        pts = sweeps_by(runs, p, "width", 1024, ("c",))
        predicted = {"base": .00132, "mup": .00194}[p]
        best = pts[pts[:, 1].argmin()]
        lr_fit, loss_fit = vertex(pts)
        loss_pred = float(dict(pts)[predicted])
        loss_src = float(dict(pts)[SOURCE_LR])
        results[p] = dict(points=pts, predicted=predicted, lr_fit=lr_fit, loss_fit=loss_fit, best=best, loss_pred=loss_pred, loss_src=loss_src)
        print(f"  {NAMES[p]:15s} sweep " + ", ".join(f"{lr:g}:{l:.4f}" for lr, l in pts))
        print(f"     fitted optimum {lr_fit:.5f} (min {loss_fit:.4f}); best sampled LR {best[0]:g} loss {best[1]:.4f}; "
              f"predicted {predicted:g}: loss {loss_pred:.4f} (gap {loss_pred - best[1]:+.4f}); transferred .003: loss {loss_src:.4f} (gap {loss_src - best[1]:+.4f})")
        ax.plot(pts[:, 0], pts[:, 1], "o-", color="C0", label="sweep (including the predicted-LR run)")
        ax.plot(predicted, loss_pred, "x", color="k", markersize=12, markeredgewidth=2.5, label=f"predicted LR {predicted:g}")
        ax.plot(SOURCE_LR, loss_src, "s", color="C3", markersize=9, label="transferred LR 0.003")
        ax.plot(lr_fit, loss_fit, "*", color="gold", markersize=16, markeredgecolor="k", label=f"fitted optimum {lr_fit:.4f}")
        ax.set_xscale("log")
        ax.set_xlabel("base LR")
        ax.set_ylabel("final val loss")
        ax.set_title(f"{NAMES[p]}, width 1,024")
        ax.legend(fontsize=8)
    t.save(fig, "p42c_width1024.png")
    return results


def part_d(runs):
    """Depth transfer at width 512: depths 4, 8 (supplied) and 16 for standard µP, Depth-µP and CompleteP."""
    print("\n4.2(d): depth")
    supplied = np.array(sorted((r["learning_rate"], r["final_val_loss"]) for r in load("P1a") if r["tokens"] == TOKENS))
    depths = (4, 8, 16)
    prescriptions = ("mup", "depth-mup", "completep")
    fits, transferred, sweeps = {}, {}, {}
    for p in prescriptions:
        for d in depths:
            pts = supplied if d == 8 else sweeps_by(runs, p, "depth", d, ("d",))
            sweeps[(p, d)] = pts
            fits[(p, d)] = vertex(pts)
            transferred[(p, d)] = float(dict(pts)[SOURCE_LR])
            print(f"  {NAMES.get(p, p):10s} d={d:2d}: sweep " + ", ".join(f"{lr:g}:{l:.4f}" for lr, l in pts) +
                  f"  | fitted LR* {fits[(p, d)][0]:.5f}, min loss {fits[(p, d)][1]:.4f}, loss@.003 {transferred[(p, d)]:.4f}")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, p in enumerate(prescriptions):
        axes[0].plot(depths, [fits[(p, d)][0] for d in depths], "o-", color=f"C{i}", label=NAMES.get(p, p) if p != "mup" else "Standard µP")
        axes[1].plot(depths, [fits[(p, d)][1] for d in depths], "o-", color=f"C{i}", label=f"{NAMES.get(p, p) if p != 'mup' else 'Standard µP'}: fitted minimum")
        axes[1].plot(depths, [transferred[(p, d)] for d in depths], "s--", color=f"C{i}", alpha=.7)
        axes[2].plot(depths, [transferred[(p, d)] - fits[(p, d)][1] for d in depths], "o-", color=f"C{i}", label=NAMES.get(p, p) if p != "mup" else "Standard µP")
    axes[0].axhline(SOURCE_LR, color="gray", linestyle=":", label="transferred source LR")
    for ax, ylabel in zip(axes, ("fitted optimal base LR", "validation loss (dashed: at LR 0.003)", "loss at LR 0.003 − fitted minimum")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(depths, [str(d) for d in depths])
        ax.minorticks_off()
        ax.set_xlabel("depth (width 512)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    t.save(fig, "p42d_depth_transfer.png")
    return fits, transferred, sweeps


def alpha_by_step(records, matrix):
    out = defaultdict(list)
    for r in records:
        if r["parameter"].endswith(f".{matrix}.weight") and r.get("alpha") is not None:
            out[r["step"]].append(r["alpha"])
    return {s: float(np.mean(v)) for s, v in sorted(out.items())}


def readout_alpha(records):
    return {r["step"]: r["alpha"] for r in records if r["parameter"] == "lm_head.weight" and r.get("alpha") is not None}


def part_a(runs):
    """alpha_upd through training and omega_move at the source LR, widths 128, 256 and the supplied 512."""
    ref = reference_diagnostics()
    series = {}
    for p in ("base", "mup"):
        for w in (128, 256):
            run = next(r for r in runs if r["part"] == "a" and r["prescription"] == p and r["width"] == w and r["lr"] == SOURCE_LR)
            files = run_files(run)
            series[(p, w)] = dict(alignment=files["alignment.jsonl"], features=files["features.jsonl"])
    for p in ("base", "mup"):
        series[(p, 512)] = dict(alignment=ref["alignment"], features=ref["diagnostics"],
                                omega=[(r["step"], (r["movement"] or {}).get("omega")) for r in ref["readout_alignment"]])
    colors = {"base": "C3", "mup": "C0"}
    styles = {128: ":", 256: "--", 512: "-"}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharex=True)
    for ax, matrix in zip(axes.flat, (*MATRICES, "readout")):
        for (p, w), s in series.items():
            alpha = readout_alpha(s["alignment"]) if matrix == "readout" else alpha_by_step(s["alignment"], matrix)
            ax.plot(list(alpha), list(alpha.values()), styles[w], color=colors[p], marker=".", label=f"{NAMES[p]} w={w}")
        ax.axhline(.5, color="k", linestyle=":", linewidth=1)
        ax.axhline(1., color="k", linestyle="--", linewidth=1)
        ax.set_xscale("symlog", linthresh=5)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"α_upd: {matrix.replace('_proj', '')}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("optimizer update")
    axes[0, 0].legend(fontsize=7)
    t.save(fig, "p42a_alignment.png")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    print("\n4.2(a): alpha_upd averaged over block matrices and layers, and omega_move, at source LR .003")
    for (p, w), s in series.items():
        pts = {}
        for matrix in MATRICES:
            for step, v in alpha_by_step(s["alignment"], matrix).items():
                pts.setdefault(step, []).append(v)
        mean_alpha = {step: np.mean(v) for step, v in pts.items()}
        if "omega" in s:
            omega = {step: o for step, o in s["omega"] if o is not None}
        else:
            omega = {f["step"]: f["readout_alignment"]["movement"]["omega"] for f in s["features"]
                     if f["readout_alignment"]["movement"]["omega"] is not None}
        ax.plot(list(omega), list(omega.values()), styles[w], color=colors[p], marker="o", label=f"{NAMES[p]} w={w}")
        steps = sorted(mean_alpha)
        pick = [x for x in (1, 5, 23, 234, 1172, 2344) if x in mean_alpha]
        print(f"  {NAMES[p]:15s} w={w:3d}: mean alpha " + ", ".join(f"step {x}: {mean_alpha[x]:.2f}" for x in pick) +
              f" | readout alpha " + ", ".join(f"{x}: {readout_alpha(s['alignment']).get(x, float('nan')):.2f}" for x in pick) +
              " | omega " + ", ".join(f"{k}: {v:.3f}" for k, v in omega.items()))
    ax.axhline(.5, color="k", linestyle=":", linewidth=1)
    ax.axhline(1., color="k", linestyle="--", linewidth=1)
    ax.set_xscale("symlog", linthresh=5)
    ax.set_ylim(.4, 1.05)
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("ω_move")
    ax.set_title("Readout–feature-change alignment during training (LR 0.003)")
    ax.legend(fontsize=7)
    t.save(fig, "p42a_omega.png")


def final_features(files):
    rows = {f["step"]: f for f in files["features.jsonl"]}
    return rows


def part_e(runs):
    """Logit RMS, normalized feature movement and omega_move at update 5 and at the end, at the transferred and best LRs."""
    print("\n4.2(e): configuration, LR, loss | logit RMS (step 5 / end) | final-norm movement (step 5 / end) | omega (step 5 / end) | grad norm step1 / median, fraction clipped")
    rows = []
    groups = defaultdict(list)
    for r in runs:
        groups[(r["prescription"], r["width"], r["depth"])].append(r)
    wanted = [(p, w, 8) for p in ("base", "mup") for w in (128, 256, 1024)] + \
             [(p, 512, d) for p in ("mup", "depth-mup", "completep") for d in (4, 16)]
    for key in wanted:
        rs = sorted(groups[key], key=lambda r: r["lr"])
        best = min(rs, key=lambda r: r["loss"])
        for r in ([best] if best["lr"] == SOURCE_LR else [next(x for x in rs if x["lr"] == SOURCE_LR), best]):
            files = run_files(r, ("features.jsonl",))
            feats = {f["step"]: f for f in files["features.jsonl"]}
            end = max(feats)
            norm = lambda f: f["features"]["model.norm"]["movement"] / max(f["features"]["model.norm"]["rms"], 1e-9)
            omega = lambda f: (f["readout_alignment"].get("movement") or {}).get("omega")
            grad = gradient_summary(r)
            tag = "best" if r is best else "transferred"
            label = f"{NAMES[key[0]]} w={key[1]} d={key[2]}"
            row = dict(label=label, lr=r["lr"], tag=tag, loss=r["loss"], logit5=feats[5]["logit_rms"], logit_end=feats[end]["logit_rms"],
                       move5=norm(feats[5]), move_end=norm(feats[end]), omega5=omega(feats[5]), omega_end=omega(feats[end]), **{f"grad_{k}": v for k, v in grad.items()})
            rows.append(row)
            print(f"  {label:26s} LR {r['lr']:g} ({tag}) loss {r['loss']:.4f} | logit {row['logit5']:.2f}/{row['logit_end']:.2f} | "
                  f"move {row['move5']:.3f}/{row['move_end']:.3f} | omega {row['omega5']:.3f}/{row['omega_end']:.3f} | grad norm {grad['norm1']:.1f} (coef {grad['coef1']:.2f}) at update 1, {grad['norm5']:.1f} (coef {grad['coef5']:.2f}) at update 5")
    return rows


def main():
    runs = fetch_runs()
    print(len(runs), "finished 4.2 runs")
    parts = sys.argv[1:] or ["b", "c", "d", "a", "e"]
    if "b" in parts:
        fits, transferred = part_b(runs)
    if "c" in parts:
        part_c(runs, None)
    if "d" in parts:
        part_d(runs)
    if "a" in parts:
        part_a(runs)
    if "e" in parts:
        part_e(runs)


if __name__ == "__main__":
    main()
