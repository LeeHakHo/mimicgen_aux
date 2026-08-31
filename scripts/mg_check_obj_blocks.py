"""Check, against the generated data, that a task's aux target blocks point at objects that MOVE.

The aux offsets come from the env's observable layout (mg_probe_object_layout.py). They should
therefore be right -- but the failure they guard against is silent and expensive: if a block index
is wrong, three_piece_assembly and kitchen train their aux head to predict a fixture that never
moves, and nothing in training reports an error. Half a GPU-day per arm would be spent learning a
constant. This reads the data itself and refuses to let that through.

Checks, per task:
  * obs/object is wide enough for every configured block
  * every SELECTED block's position actually varies within an episode
  * reports the motion of every block so a mis-selection is visible, not just a pass/fail

Usage: python mg_check_obj_blocks.py <task> [--min_motion 0.02] [--n_demos 20]
Exit code 0 = pass, 1 = fail.
"""
import argparse
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, "/scratch1/hyeonhoo/code/Robomimic_Async")
from mg_make_train_configs import OBJ_BLOCKS, RESULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--min_motion", type=float, default=0.02,
                    help="metres a manipulated object's position must span within an episode "
                         "(median over demos) for the block to count as moving")
    ap.add_argument("--n_demos", type=int, default=20)
    args = ap.parse_args()

    task = args.task
    blocks = OBJ_BLOCKS[task]
    path = f"{RESULTS}/mg_{task}_id90/{task}_id90/demo.hdf5"

    with h5py.File(path, "r") as f:
        data = f["data"]
        demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))[: args.n_demos]
        width = data[demos[0]]["obs"]["object"].shape[1]
        # motion of every block the layout could hold, so a wrong pick is visible next to the right one.
        # The stride-7 sweep is a display aid only -- it does NOT enumerate the valid offsets. A task
        # whose object-state interleaves relative vectors puts a real object off the stride (stack_three
        # keeps cubeC at 23), so the CONFIGURED blocks are always measured, or they read as motionless.
        starts = sorted(set(range(0, width - 6, 7)) | {b for b in blocks if b + 7 <= width})
        motion = {s: [] for s in starts}
        for d in demos:
            obj = data[d]["obs"]["object"][()]
            for s in starts:
                p = obj[:, s:s + 3]
                motion[s].append(float(np.linalg.norm(p.max(0) - p.min(0))))

    med = {s: float(np.median(v)) for s, v in motion.items()}
    print(f"task {task}   obs/object width {width}   configured blocks {tuple(blocks)}")
    print(f"{'offset':>7s} {'median motion (m)':>18s}   {'':s}")
    for s in starts:
        tag = "  <== SELECTED" if s in blocks else ""
        print(f"{s:7d} {med[s]:18.4f}{tag}")

    problems = []
    for b in blocks:
        if b + 7 > width:
            problems.append(f"block {b} needs columns {b}:{b + 7} but obs/object is only {width} wide")
        elif med.get(b, 0.0) < args.min_motion:
            problems.append(
                f"block {b} moves only {med.get(b, 0.0):.4f} m (median over {len(demos)} demos), "
                f"below {args.min_motion} m -- this looks like a STATIC object, not a manipulated one")

    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  " + p)
        moving = [s for s in starts if med[s] >= args.min_motion]
        print(f"  blocks that DO move: {moving}")
        return 1
    print(f"\nPASS: all {len(blocks)} selected block(s) move at least {args.min_motion} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
