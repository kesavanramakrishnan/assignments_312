"""Problem 4: Inside the Hessian.

Each stage launches its jobs on Modal and returns; re-running a stage launches only the
jobs whose results are not on your volume yet. `fetch` downloads every finished result.

    uv run python -m experiments.a3_optimization.p4_inside_the_hessian a          # part (a)
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian b-rescale  # part (b)
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian b-train
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian b-measure  # after b-train
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian c          # part (c)
    uv run python -m experiments.a3_optimization.p4_inside_the_hessian fetch

Add --local to run measurement stages on this machine's GPU instead. Read results with

    from experiments.a3_optimization.hessian_jobs import load_result
    r = load_result("a-adamw-9375")
    r["lambda_max"], r["top"]["residuals"][0], r["families"]["head"]["lambda_max"]
    r["energy"]["layer"]["blocks"], r["energy"]["layer"]["per_probe"]   # [probe, block, block]
"""

import argparse

from experiments.a3_optimization.hessian_checkpoints import FINAL_STEP, STEPS
from experiments.a3_optimization.hessian_jobs import EXPERIMENT_KEY, fetch, launch
from train import TrainConfig, training_run_name


# Part (a): the shipped AdamW checkpoints (no training). Each checkpoint costs about 13
# H100-minutes with these options; a density adds about 13 more.
CHECKPOINTS = STEPS  # 0, 3, 15, ..., 9375
DENSITY_AT = (0, 153, 1526, 9375)  # initialization, 10M, 100M, 614M tokens

LOOK_JOBS = [
    {"name": f"a-adamw-{step}",
     "measure": {"run": "adamw", "step": step,
                 "extremes": True,  # lambda_max, lambda_min, top-5 eigenvectors by family
                 "families": True,  # top eigenvalue of each family's block
                 "energy": ("family", "layer"),  # ||H_cb||_F^2 between blocks
                 "density_runs": 4 if step in DENSITY_AT else 0}}
    for step in CHECKPOINTS
]

# Part (a), block structure baseline: the same block sizes, but each block a random subset of
# the coordinates. Compared with the real family/layer energies, it shows how much of ||H||_F^2
# sits in the blocks because of structure rather than because blocks are big.
LOOK_JOBS += [
    {"name": f"a-random-{step}",
     "measure": {"run": "adamw", "step": step, "extremes": False, "families": False,
                 "energy_random": ("family", "layer")}}
    for step in CHECKPOINTS
]

# Part (b): hand rescaling of the shipped final checkpoint (no training). `scale`
# multiplies named parameters by constants before measuring; `tensors` reports the top
# eigenvalue of each named tensor's own block, with its RMS and gradient norm.
FINAL = {"extremes": True, "families": True}

RESCALE_JOBS = [
    {"name": "b-rescale-reference",
     "measure": {"run": "adamw", "step": FINAL_STEP, **FINAL, "tensors": []}},
]

# Part (b): function-preserving rescalings of the final checkpoint. Scaling a parameter
# `a` by c and its partner `b` by 1/c leaves the predictions unchanged (RMSNorm erases the
# scale of whatever feeds it, and linear maps compose), but H_aa scales by 1/c^2 and H_bb by c^2.
L = range(8)


def _rescale(label, scale, tensors):
    return {"name": f"b-rescale-{label}",
            "measure": {"run": "adamw", "step": FINAL_STEP, **FINAL, "scale": scale,
                        "tensors": tensors}}


def _layers(part, factor):
    return {f"model.layers.{i}.{part}": factor for i in L}


for _label, _c in (("head-x0.25", 0.25), ("head-x0.5", 0.5), ("head-x4", 4.0)):
    # logits = lm_head(norm(x)): lm_head * c and final-norm gain / c
    RESCALE_JOBS.append(_rescale(_label, {"lm_head.weight": _c, "model.norm.weight": 1 / _c},
                                 ["lm_head.weight", "model.norm.weight"]))
RESCALE_JOBS.append(_rescale(
    "mlp-x0.25", {**_layers("mlp.up_proj.weight", 0.25), **_layers("mlp.down_proj.weight", 4.0)},
    ["model.layers.3.mlp.up_proj.weight", "model.layers.3.mlp.down_proj.weight"]))
RESCALE_JOBS.append(_rescale(
    "vo-x0.25", {**_layers("self_attn.v_proj.weight", 0.25), **_layers("self_attn.o_proj.weight", 4.0)},
    ["model.layers.3.self_attn.v_proj.weight", "model.layers.3.self_attn.o_proj.weight"]))
RESCALE_JOBS.append(_rescale(
    "qk-x0.25", {**_layers("self_attn.q_proj.weight", 0.25), **_layers("self_attn.k_proj.weight", 0.25)},
    ["model.layers.3.self_attn.q_proj.weight", "model.layers.3.self_attn.k_proj.weight"]))

# Part (b): training changes. A run takes about 11 minutes on an H200 (a little more on
# an H100); measuring its final checkpoint about 12.
# head_init_scale = c multiplies the output head at initialization by c and the final norm
# gain by 1/c (function-preserving), starting training with a sharper (c < 1) or flatter
# (c > 1) head block than the default recipe reaches by itself.
RUNS = [
    TrainConfig(head_init_scale=c, wandb_tags=("a3", EXPERIMENT_KEY)) for c in (0.25, 0.0625, 4.0)
]

FINAL_JOBS = [
    {"name": f"b-final-{training_run_name(config)}",
     "measure": {"run": training_run_name(config), "step": FINAL_STEP, **FINAL}}
    for config in RUNS
]

# Part (c): continue the shipped SGDM run from 100M tokens with the full update, the
# update projected onto the top-k eigenvectors, and the update with them removed. An arm
# costs 25-60 H100-minutes, mostly in its 12 basis refreshes.
START = {"run": "sgdm", "step": 1526, "k": 10, "steps": 300, "refresh": 25,
         "reset_momentum": True}

SUBSPACE_JOBS = [
    {"name": f"c-{arm}", "continue": {**START, "transform": arm}}
    for arm in ("full", "top", "removed")
]


# Part (c) variants of the three arms: lower LR for the top-subspace arm, a larger LR for the
# arm with the top subspace removed, different subspace sizes, and an earlier start.
SUBSPACE_JOBS += [
    {"name": "c-top-lr0.25", "continue": {**START, "transform": "top", "lr_scale": 0.25}},
    {"name": "c-removed-lr2", "continue": {**START, "transform": "removed", "lr_scale": 2.0}},
    {"name": "c-top-k3", "continue": {**START, "transform": "top", "k": 3}},
    {"name": "c-top-k40", "continue": {**START, "transform": "top", "k": 40, "lanczos_steps": 100}},
    {"name": "c-removed-k40", "continue": {**START, "transform": "removed", "k": 40, "lanczos_steps": 100}},
    {"name": "c-top-start305", "continue": {**START, "transform": "top", "step": 305}},
    {"name": "c-removed-start305", "continue": {**START, "transform": "removed", "step": 305}},
]

# TODO: Part (a): anything else you need, e.g. the baseline for the block-structure
# question. H.hvp (hessian.ProbeHessian) applies H to any vector you construct.
# TODO: Part (b): one rescaling job per transformation you test, e.g.
# RESCALE_JOBS.append({"name": "b-rescale-<label>",
#     "measure": {"run": "adamw", "step": FINAL_STEP, **FINAL,
#                 "scale": {"<parameter name>": 0.25, ...}, "tensors": ["<parameter name>", ...]}})
# and the training changes in RUNS.
# TODO: Part (c): vary what you think decides the outcome, e.g.
# SUBSPACE_JOBS.append({"name": "c-removed-<label>", "continue": {**START, "transform": "removed", ...}})
# Options: k, refresh, steps, lr_scale (multiplies the run's schedule), reset_momentum,
# and the starting step (any of STEPS; the run has optimizer state at each).

STAGES = {"a": LOOK_JOBS, "b-rescale": RESCALE_JOBS, "b-measure": FINAL_JOBS, "c": SUBSPACE_JOBS}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=(*STAGES, "b-train", "fetch"))
    parser.add_argument("--local", action="store_true", help="run on this machine's GPU")
    args = parser.parse_args()
    if args.stage == "b-train":
        from modal_train import launch_training_jobs

        launch_training_jobs(RUNS)
    elif args.stage == "fetch":
        print(fetch([j["name"] for jobs in STAGES.values() for j in jobs]))
    else:
        launch(STAGES[args.stage], local=args.local)


if __name__ == "__main__":
    main()
