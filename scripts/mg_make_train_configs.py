"""Emit one robomimic training config per task for the 12-task ID90 study (84px, 500 epochs).

Derived from the Hammer yaw45_spawn12 config that produced this project's reference result, with
only what the new study changes:

  epochs            1000 -> 500
  save.every_n_epochs 50 -> 10   (50 checkpoints; the eval grid uses a subset and can fill in)
  save.on_best_rollout_success_rate  true -> FALSE
        Left on, checkpoints would be written on rollout success too, so the saved set would no
        longer be the even 10-epoch grid the evaluation assumes.
  rollout           n=50 rate=20 -> n=1 rate=100
        Training-time rollouts are a LIVENESS check here, not a measurement: numbers from 48
        separate jobs on mixed GPUs are not comparable across arms (~+-10pt, see the project's
        cross-job noise finding), so every reported number comes from the single unified eval job.
        One episode at epochs 100..500 costs 5 episodes per run. Note what that buys and what it
        does not: it catches a crashing rollout path or a policy that never moves, but with n=1
        the success rate is only ever 0 or 1, so it cannot tell 0% apart from 20%.
  crop              202 -> 76   (84px images; 224px used 202)
  rollout.horizon   500 -> per task, from the generated data's own episode lengths

Usage: python mg_make_train_configs.py            # after datagen, reads each task's real stats
       python mg_make_train_configs.py --smoke_ok # allow falling back to the 5-demo smoke stats
"""
import argparse
import json
import math
import os

TEMPLATE = "robomimic/exps/templates/bc_transformer_hammer_yaw45_spawn12_gmm_224.json"
OUT_DIR = "robomimic/exps/templates/id90"
RESULTS = "/scratch1/hyeonhoo/results"

# Start offsets, within obs/object, of the objects the policy MANIPULATES -- recovered per task
# by mg_probe_object_layout.py (each observable matched against object-state by value, pos+quat as
# one contiguous 7-wide pair). Every block is read as [pos(3) | quat(4)] from its offset.
#
# The default the aux code shipped with -- "block 0 is the object" -- is right for only 8 of the
# 12: three_piece_assembly leads with the static assembly `base` and kitchen with the static
# `Stove1`, so training aux on block 0 there would ask the model to predict something that never
# moves; pick_place and nut_assembly move several objects, and stack_three moves two of its three
# cubes (cubeB is the base being stacked ONTO, so it is a target, not a manipulated object).
OBJ_BLOCKS = {
    "stack_d1":                (0,),           # cubeA; cubeB (7) is the base
    "stack_three_d1":          (0, 23),        # cubeA, cubeC; cubeB (7) is the base
    "square_d2":               (0,),           # SquareNut
    "threading_d0":            (0,),           # needle
    "three_piece_assembly_d0": (14, 28),       # piece_1, piece_2; base (0) is static
    "hammer_cleanup_d1":       (0,),           # hammer
    "mug_cleanup_d1":          (0,),           # object (the mug)
    "coffee_d2":               (0,),           # coffee_pod
    "kitchen_d1":              (14, 28),       # cube_bread, PotObject; Stove1 (0) is static
    "pick_place_d0":           (0, 14, 28, 42),  # Milk, Bread, Cereal, Can
    "coffee_preparation_d1":   (0, 70),        # coffee_pod, mug
    "nut_assembly_d0":         (0, 14),        # SquareNut, RoundNut
}

# Canonical geometry per manipulated object, positionally aligned with OBJ_BLOCKS, for the
# point-cloud family (pc / voxel / tsdf / embed). All 12 tasks are covered.
#   pick_place_d0          mesh objects, sampled from the assets robosuite loads
#   coffee_preparation_d1  same ShapeNet mug as mug_cleanup but at shapenet_scale 1.0, not 0.8
#   hammer_cleanup_d1      robosuite randomizes the hammer's head size per instantiation, so no
#                          single canonical cloud exists -- hammer_geometry.py reads each
#                          episode's own MJCF instead (pass --aux_pc_object hammer).
PC_OBJECTS = {
    "stack_d1":                ("cube",),
    "stack_three_d1":          ("cube", "cube"),
    "square_d2":               ("nut",),
    "threading_d0":            ("needle",),
    "three_piece_assembly_d0": ("piece_1", "piece_2"),
    "mug_cleanup_d1":          ("mug",),
    "coffee_d2":               ("coffee_pod",),
    "kitchen_d1":              ("bread", "pot"),
    "nut_assembly_d0":         ("nut", "round_nut"),
    "hammer_cleanup_d1":       ("hammer",),   # per-demo geometry, see hammer_geometry.py
    # pick_place's bread is a mesh LOAF, a different object from kitchen's box `bread`
    "pick_place_d0":           ("milk", "loaf", "cereal", "can"),
    "coffee_preparation_d1":   ("coffee_pod", "coffee_mug"),
}

# task -> the directory datagen wrote (task key doubles as the experiment stem)
TASKS = [
    "stack_d1", "stack_three_d1", "square_d2", "threading_d0", "three_piece_assembly_d0",
    "hammer_cleanup_d1", "mug_cleanup_d1", "coffee_d2", "kitchen_d1", "pick_place_d0",
    "coffee_preparation_d1", "nut_assembly_d0",
]


def stats_path(task, suffixes):
    for suf in suffixes:
        p = f"{RESULTS}/mg_{task}_id90{suf}/{task}_id90/important_stats.json"
        if os.path.exists(p):
            return p
    return None


def horizon_for(mean_len):
    """2x the mean demo length, rounded up to 100, floored at 400.

    A rollout that fails runs to the horizon, so this is the dominant cost knob at evaluation
    time; too short silently truncates slow-but-successful episodes into failures.
    """
    return max(400, int(math.ceil(2.0 * mean_len / 100.0) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--rollout_n", type=int, default=1)
    ap.add_argument("--rollout_rate", type=int, default=100)
    ap.add_argument("--crop", type=int, default=76)
    ap.add_argument("--wandb", action="store_true",
                    help="log to wandb (expects credentials in ~/.netrc, not in the job file)")
    ap.add_argument("--smoke_ok", action="store_true",
                    help="fall back to the 5-demo smoke stats when a task has no full datagen yet")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    suffixes = ["", "_smoke2", "_smoke"] if args.smoke_ok else [""]
    missing = []
    for task in TASKS:
        sp = stats_path(task, suffixes)
        if sp is None:
            missing.append(task)
            continue
        stats = json.load(open(sp))
        horizon = horizon_for(stats["ep_length_mean"])
        dataset = f"{RESULTS}/mg_{task}_id90/{task}_id90/demo.hdf5"

        cfg = json.load(open(TEMPLATE))
        exp, train = cfg["experiment"], cfg["train"]
        exp["name"] = f"{task}_id90"
        exp["env"] = None          # rollout env comes from the dataset's own env_meta
        exp["save"]["every_n_epochs"] = args.save_every
        exp["save"]["on_best_rollout_success_rate"] = False
        exp["save"]["on_best_validation"] = False
        exp["rollout"].update(enabled=True, n=args.rollout_n, rate=args.rollout_rate,
                              horizon=horizon, warmstart=0)
        _aux = cfg.setdefault("algo", {}).setdefault("aux_pose", {})
        _aux["obj_blocks"] = list(OBJ_BLOCKS[task])
        if task in PC_OBJECTS:
            _aux["pc_object"] = ",".join(PC_OBJECTS[task])
        train["num_epochs"] = args.epochs
        train["data"] = [{"path": dataset}]
        train["output_dir"] = f"{RESULTS}/id90_train/{task}"
        cfg["observation"]["encoder"]["rgb"]["obs_randomizer_kwargs"].update(
            crop_height=args.crop, crop_width=args.crop)
        # wandb off by default: the old job files carried a hardcoded API key, which is a
        # credential sitting in a file that gets committed. TensorBoard covers the same need
        # locally; to re-enable, put the key in ~/.netrc and pass --wandb.
        exp["logging"]["log_wandb"] = args.wandb
        exp["logging"]["log_tb"] = True

        out = os.path.join(OUT_DIR, f"{task}_id90_gmm_84.json")
        with open(out, "w") as f:
            json.dump(cfg, f, indent=4)
        src = "FULL" if "_smoke" not in sp else "smoke"
        print(f"{task:24s} horizon {horizon:5d}  obj_blocks {str(OBJ_BLOCKS[task]):16s}"
              f" (mean ep {stats['ep_length_mean']:6.1f}, {src})")

    if missing:
        print("\nNO DATAGEN STATS YET (config not written): " + ", ".join(missing))
        print("re-run after those datagen jobs finish, or pass --smoke_ok to size them off the "
              "5-demo smoke run")


if __name__ == "__main__":
    main()
