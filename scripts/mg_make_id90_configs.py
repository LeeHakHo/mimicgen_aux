"""Emit one MimicGen datagen config per task for the ID90 training distribution.

Each config is the upstream template from mimicgen/exps/templates/robosuite/<task>.json with only
the fields our earlier Hammer run changed (verified by diffing hammer_yaw45_spawn12.json against
its template): source dataset, generation path/num_trials/guarantee, task name+interface, camera
names, and the render counts. Everything else -- the per-task task_spec, which is the part that
actually drives subtask segmentation -- is left exactly as MimicGen ships it.

  demos  : 200 per task
  render : 84x84, agentview + robot0_eye_in_hand (collect_obs is already true in the templates)
  env    : <Base>_ID90 from mimicgen/envs/robosuite/ood_ladder.py

Usage: python mg_make_id90_configs.py [--demos 200]
"""
import argparse
import json
import os

# CARC defaults; the Anvil side of the study overrides all four through the environment rather
# than editing them, so the two clusters share one script (same pattern as MG_RESULTS in
# mg_make_train_configs.py).
TEMPLATE_DIR = os.environ.get(
    "MG_TEMPLATE_DIR", "/scratch1/hyeonhoo/code/mimicgen/mimicgen/exps/templates/robosuite")
SOURCE_DIR = os.environ.get("MG_SOURCE_DIR", "/scratch1/hyeonhoo/code/mimicgen/datasets/source")
OUT_DIR = os.environ.get("MG_CONFIG_DIR", "/scratch1/hyeonhoo/code/Robomimic_Async/mg_configs")
RESULTS = os.environ.get("MG_RESULTS", "/scratch1/hyeonhoo/results")

# task key -> (template/source basename, MimicGen env interface, ID90 env name)
TASKS = {
    "stack_d1":                ("stack",                "MG_Stack",              "Stack_D1_ID90"),
    "stack_three_d1":          ("stack_three",          "MG_StackThree",         "StackThree_D1_ID90"),
    "square_d2":               ("square",               "MG_Square",             "Square_D2_ID90"),
    "threading_d0":            ("threading",            "MG_Threading",          "Threading_D0_ID90"),
    "three_piece_assembly_d0": ("three_piece_assembly", "MG_ThreePieceAssembly", "ThreePieceAssembly_D0_ID90"),
    # hammer does NOT use the generic D1-based ID90 env. HammerCleanup_Yaw45_Spawn12 is the
    # distribution that produced this project's only above-noise baseline-vs-aux gap (0.57 ->
    # 0.77 on OOD20), and it is already an ID90-shaped window: drawer pinned to its D0 pose,
    # hammer yaw +-45 deg with the head PINNED and xy at 2.5x the narrow D0 box. Yaw45_Spawn12's
    # window was really two lobes 180 deg apart (init_quat is a coin flip), and its 0.0080 m2
    # spawn was 27x smaller in area than square_d2's, which left the task with almost no position
    # axis and made it the easiest of the twelve. The generic
    # HammerCleanup_D1_ID90 keeps D1's wide spawn (x 0.400 vs 0.096) and a moving drawer, which
    # dropped datagen success to 4% in the smoke run against ~32% here.
    "hammer_cleanup_d1":       ("hammer_cleanup",       "MG_HammerCleanup",      "HammerCleanup_FixedHead_Yaw45_Spawn25"),
    "mug_cleanup_d1":          ("mug_cleanup",          "MG_MugCleanup",         "MugCleanup_D1_ID90"),
    "coffee_d2":               ("coffee",               "MG_Coffee",             "Coffee_D2_ID90"),
    "kitchen_d1":              ("kitchen",              "MG_Kitchen",            "Kitchen_D1_ID90"),
    "pick_place_d0":           ("pick_place",           "MG_PickPlace",          "PickPlace_D0_ID90"),
    "coffee_preparation_d1":   ("coffee_preparation",   "MG_CoffeePreparation",  "CoffeePreparation_D1_ID90"),
    "nut_assembly_d0":         ("nut_assembly",         "MG_NutAssembly",        "NutAssembly_D0_ID90"),
}
CAMERAS = ["agentview", "robot0_eye_in_hand"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=200)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for task, (base, iface, env_name) in TASKS.items():
        with open(os.path.join(TEMPLATE_DIR, f"{base}.json")) as f:
            cfg = json.load(f)

        exp = cfg["experiment"]
        exp["name"] = f"{task}_id90"
        exp["source"]["dataset_path"] = os.path.join(SOURCE_DIR, f"{base}.hdf5")
        exp["generation"]["path"] = os.path.join(RESULTS, f"mg_{task}_id90")
        exp["generation"]["num_trials"] = args.demos
        exp["generation"]["guarantee"] = True   # keep going until num_trials SUCCEED
        exp["task"]["name"] = env_name
        exp["task"]["interface"] = iface
        exp["task"]["interface_type"] = "robosuite"
        exp["num_demo_to_render"] = 5
        exp["num_fail_demo_to_render"] = 5

        cfg["obs"]["camera_names"] = list(CAMERAS)
        assert cfg["obs"]["collect_obs"], f"{task}: template has collect_obs off"
        assert cfg["obs"]["camera_height"] == 84 and cfg["obs"]["camera_width"] == 84, \
            f'{task}: template renders {cfg["obs"]["camera_height"]}x{cfg["obs"]["camera_width"]}'

        out = os.path.join(args.out_dir, f"{task}_id90.json")
        with open(out, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"{task:24s} -> {out}  (env {env_name}, {args.demos} demos)")


if __name__ == "__main__":
    main()
