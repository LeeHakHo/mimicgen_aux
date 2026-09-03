"""Verify that the OOD ladders reach the sampler that actually places objects.

`mg_verify_ood_ladder.py` reads `_get_initial_placement_bounds()` -- the dict the ladder mixins
rewrite. That checks the INTENT. It cannot see whether anything consumes the dict, and on
`NutAssembly_D0` nothing did: robosuite's `NutAssembly._load_model()` built its own sampler with a
hardcoded x/y and `rotation=None`, so ID90, OOD_POS, OOD_YAW and OOD_BOTH all sampled identically
while the declared windows said otherwise, and the whole task was generated and evaluated against
a ladder that was never applied.

This script closes that gap from the other side: it draws real placements from
`env.placement_initializer` -- the object the env places from, whatever its type -- and checks
what comes out.

  --mode rungs   (default) the ID90 / OOD_POS / OOD_YAW / OOD_BOTH windows, each measured against
                 the width the bounds dict declares. Spans, not endpoints, because a sampler's
                 `reference_pos` and an object's `init_quat` shift the absolute values while
                 leaving the width alone, and the width is what the ladder manipulates.
  --mode graded  the five-level position and rotation ladders, measured against the ID env rather
                 than against a declared number: position levels must scale by 1.2 .. 2.0, and
                 rotation levels must be disjoint from the ID window, disjoint from each other,
                 ordered outward, and reach the far side of the orbit at L5.

Usage:
  python mg_verify_sampler_ranges.py                                  # rungs, all 12 tasks
  python mg_verify_sampler_ranges.py --mode graded stack_d1
Exit code is 0 only when every check passes.
"""

import argparse
import sys

import numpy as np
import robosuite as suite
from robosuite.utils import RandomizationError

import mimicgen  # noqa: F401  -- registers the ladder envs with EnvMeta
from mimicgen.envs.robosuite.ood_ladder import POS_LADDER_SCALES, YAW_LADDER_LEVELS

# task key -> the ladder's env stem; the rungs are <stem>_ID90 / _OOD_POS / _OOD_YAW / _OOD_BOTH
# and the graded levels <stem>_OOD_POS_L1..L5 / _OOD_YAW_L1..L5, exactly as the datagen and
# eval-scene jobs name them.
STEMS = {
    "stack_d1": "Stack_D1",
    "stack_three_d1": "StackThree_D1",
    "square_d2": "Square_D2",
    "threading_d0": "Threading_D0",
    "three_piece_assembly_d0": "ThreePieceAssembly_D0",
    "hammer_cleanup_d1": "HammerCleanup_FixedHead_Yaw45_Spawn25",
    "mug_cleanup_d1": "MugCleanup_D1",
    "coffee_d2": "Coffee_D2",
    "kitchen_d1": "Kitchen_D1",
    "pick_place_d0": "PickPlace_D0",
    "coffee_preparation_d1": "CoffeePreparation_D1",
    "nut_assembly_d0": "NutAssembly_D0",
}

# hammer's ID env is the training distribution itself, not a `_ID90` alias
def id_env_of(stem):
    return stem if stem.startswith("HammerCleanup") else f"{stem}_ID90"


# PickPlace builds its sampler ranges from bin geometry inside _get_placement_initializer rather
# than from the bounds dict, so the dict is not the source of truth there and the `rungs`
# comparison does not apply. mg_verify_ood_ladder.py checks that task against its sampler directly.
SKIP_RUNGS = {"pick_place_d0"}

# Levels that are degenerate by construction, and are expected to equal ID rather than to move.
# See GRADED_DEGENERATE in ood_ladder.py.
DEGENERATE_GRADED = {("coffee_d2", "YAW"), ("pick_place_d0", "POS")}

RUNGS = ("ID90", "OOD_POS", "OOD_YAW", "OOD_BOTH")
N_DRAWS = 200
POS_TOL = 1e-4      # metres
ROT_TOL = np.radians(0.5)
COVER = 0.85        # a fair sampler covers ~(N-1)/(N+1) of its window; leave room for rejections


def make(env_name):
    return suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
    )


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def draw(env_name, n=N_DRAWS):
    """-> {object_name: (xs, ys, yaws)} from real placements, arrays of length <= n."""
    env = make(env_name)
    try:
        xs, ys, yaws = {}, {}, {}
        for _ in range(n):
            try:
                placements = env.placement_initializer.sample()
            except RandomizationError:
                continue  # a rejected draw says nothing about the window
            for name, (pos, quat, _obj) in placements.items():
                xs.setdefault(name, []).append(pos[0])
                ys.setdefault(name, []).append(pos[1])
                yaws.setdefault(name, []).append(yaw_of(quat))
        return {k: (np.asarray(xs[k]), np.asarray(ys[k]), np.asarray(yaws[k])) for k in xs}
    finally:
        env.close()


def declared(env_name):
    """-> [(object, x width, y width, yaw width)] for every object the bounds dict declares."""
    env = make(env_name)
    try:
        spec = env._get_initial_placement_bounds()
        return [(obj,
                 abs(b["x"][1] - b["x"][0]) if "x" in b else None,
                 abs(b["y"][1] - b["y"][0]) if "y" in b else None,
                 abs(b["z_rot"][1] - b["z_rot"][0]) if "z_rot" in b else None)
                for obj, b in spec.items()]
    finally:
        env.close()


def yaw_span(yaws):
    """Width of the sampled yaw interval, unwrapped so a window straddling +-pi reads as one."""
    return float(np.ptp(np.unwrap(np.sort(np.asarray(yaws)))))


def fold_of(env_name):
    env = make(env_name)
    try:
        return int(getattr(env, "yaw_fold", 1))
    finally:
        env.close()


# ---------------------------------------------------------------- mode: rungs


def check_axis(measured, width, tol, label, obj, failures):
    """Both directions: not wider than declared, and not ignoring the window by staying narrow."""
    if width is None:
        return f"{label} --"
    if measured > width + tol:
        failures.append(f"{obj}: {label} sampled {measured:.4f} > declared {width:.4f}")
        return f"{label} {measured:.4f}>{width:.4f} FAIL"
    if width > tol and measured < COVER * width - tol:
        failures.append(f"{obj}: {label} sampled {measured:.4f} < {COVER:.2f} x declared {width:.4f}")
        return f"{label} {measured:.4f}<{width:.4f} FAIL"
    return f"{label} {measured:.4f}/{width:.4f}"


def check_rungs(task, n_draws, failures):
    stem = STEMS[task]
    for rung in RUNGS:
        env_name = id_env_of(stem) if rung == "ID90" else f"{stem}_{rung}"
        got = {k: (np.ptp(v[0]), np.ptp(v[1]), yaw_span(v[2]))
               for k, v in draw(env_name, n_draws).items()}
        want = declared(env_name)

        # The bounds dict and the sampler name objects differently (`square_nut` vs `SquareNut`,
        # and a per-object sampler may split one entry), so pair them by matching each declared
        # window to the measured object it best fits rather than by name -- what matters is that
        # every declared window is the one being sampled from.
        unclaimed = dict(got)
        for obj, wx, wy, wr in want:
            best, best_cost = None, None
            for name, (mx, my, mr) in unclaimed.items():
                cost = sum(abs(m - w) for m, w in ((mx, wx), (my, wy), (mr, wr)) if w is not None)
                if best_cost is None or cost < best_cost:
                    best, best_cost = name, cost
            if best is None:
                failures.append(f"{env_name}/{obj}: declared but nothing sampled it")
                continue
            mx, my, mr = unclaimed.pop(best)
            cols = [
                check_axis(mx, wx, POS_TOL, "x", f"{env_name}/{obj}", failures),
                check_axis(my, wy, POS_TOL, "y", f"{env_name}/{obj}", failures),
                check_axis(np.degrees(mr), None if wr is None else np.degrees(wr),
                           np.degrees(ROT_TOL), "yaw_deg", f"{env_name}/{obj}", failures),
            ]
            print(f"  {rung:9s} {obj:18s} ~ {best:18s} " + "  ".join(cols))


# ---------------------------------------------------------------- mode: graded


def sampler_ranges(env_name):
    """-> {label: (x_range, y_range, rotation)} straight off the env's placement initializer.

    The graded ladder is checked against these rather than against 200 drawn placements, because
    the two failure modes it has to separate are geometric -- do the bands overlap, do they walk
    outward, does the last one reach the far side of the orbit -- and a 200-draw span estimate is
    a noisy measurement of a window whose exact endpoints are sitting right here. It also avoids
    reading a rotation back out of a quaternion, which does not work on `rotation_axis="y"` objects
    like the hammer: the sampler's angle and the z-yaw of the resulting quat are not the same
    number once init_quat lays the object flat.

    That the sampler is CONSULTED at all -- the NutAssembly_D0 failure -- is what `--mode rungs`
    establishes empirically, on the same machinery these levels inherit.
    """
    env = make(env_name)
    try:
        init = env.placement_initializer
        samplers = getattr(init, "samplers", None)
        samplers = dict(samplers) if samplers else {getattr(init, "name", "sampler"): init}
        out = {}
        for sname, s in samplers.items():
            per_object = getattr(s, "per_object", None)
            if per_object:
                # a per-object sampler carries one window per cube; report each separately
                for obj, ov in per_object.items():
                    out[f"{sname}/{obj}"] = (
                        tuple(ov.get("x_range", s.x_range)),
                        tuple(ov.get("y_range", s.y_range)),
                        tuple(ov["rotation"]) if isinstance(ov.get("rotation", s.rotation),
                                                            (tuple, list)) else None,
                    )
                continue
            out[sname] = (
                tuple(s.x_range), tuple(s.y_range),
                tuple(s.rotation) if isinstance(s.rotation, (tuple, list)) else None,
            )
        return out
    finally:
        env.close()


def width(rng):
    return None if rng is None else abs(rng[1] - rng[0])


def fold_delta(angle, center, fold):
    """Distance from `center` on the orientation ORBIT, in DEGREES, in [0, 180/fold].

    Folding first means a C4 cube's yaw and yaw+90 deg count as the same scene, which is what makes
    "distance from the training window" mean the same thing for a cube as for a nut.
    """
    return abs(float(np.degrees(np.angle(np.exp(1j * fold * (angle - center))) / fold)))


def check_graded(task, n_draws, failures):
    stem = STEMS[task]
    fold = fold_of(f"{stem}_OOD_BOTH")
    base = sampler_ranges(id_env_of(stem))
    far = np.degrees(np.pi / fold)
    print(f"  fold {fold}, far side of the orbit {far:.1f} deg")

    # --- position ladder. Clamping to the table can hold a level short of its nominal scale; that
    # is a real property of the env, so it is reported rather than failed.
    print("  -- position ladder: x/y width vs the ID env, rotation must not move")
    for level, scale in enumerate(POS_LADDER_SCALES, start=1):
        got = sampler_ranges(f"{stem}_OOD_POS_L{level}")
        for label in sorted(got):
            ref = base.get(label, base.get(label.split("/")[0]))
            if ref is None:
                failures.append(f"{stem}_OOD_POS_L{level}: sampler {label} absent from the ID env")
                continue
            (bx, by, brot), (mx, my, mrot) = ref, got[label]
            held = (mx == bx and my == by)
            rx = width(mx) / width(bx) if width(bx) > POS_TOL else float("nan")
            ry = width(my) / width(by) if width(by) > POS_TOL else float("nan")
            tag = ""
            if held:
                tag = " held at ID (target or degenerate)"
            elif rx == rx and rx > scale + 1e-6:
                failures.append(f"{stem}_OOD_POS_L{level}/{label}: x scaled {rx:.3f} > "
                                f"nominal {scale}")
                tag = " FAIL"
            elif rx == rx and rx < scale - 1e-6:
                tag = " clamped-to-table"
            if mrot != brot:
                failures.append(f"{stem}_OOD_POS_L{level}/{label}: rotation moved from "
                                f"{brot} to {mrot} -- the position ladder must hold it")
                tag += " ROT-MOVED"
            print(f"     L{level} x{scale:<4} {label:34s} x {width(mx):.4f} ({rx:.2f}x)  "
                  f"y {width(my):.4f} ({ry:.2f}x){tag}")

    # --- rotation ladder: disjoint bands walking outward, position untouched.
    print("  -- rotation ladder: |dyaw| from the ID window center, deg")
    reach = {}
    for label, (_bx, _by, brot) in base.items():
        if brot is None:
            continue
        c = (brot[0] + brot[1]) / 2.0
        reach[label] = [c, np.degrees(width(brot) / 2.0)]  # center, how far ID itself reaches
    for level in range(1, YAW_LADDER_LEVELS + 1):
        got = sampler_ranges(f"{stem}_OOD_YAW_L{level}")
        for label in sorted(got):
            ref_key = label if label in base else label.split("/")[0]
            if ref_key not in base:
                continue
            (bx, by, brot), (mx, my, mrot) = base[ref_key], got[label]
            tag = ""
            if mx != bx or my != by:
                failures.append(f"{stem}_OOD_YAW_L{level}/{label}: x/y moved from {bx},{by} to "
                                f"{mx},{my} -- the rotation ladder must hold position at ID")
                tag += " POS-MOVED"
            if brot is None or mrot is None or mrot == brot:
                print(f"     L{level}       {label:34s} rotation held at ID{tag}")
                continue
            c, hi_so_far = reach.setdefault(label, list(reach[ref_key]))
            lo = min(fold_delta(mrot[0], c, fold), fold_delta(mrot[1], c, fold))
            hi = max(fold_delta(mrot[0], c, fold), fold_delta(mrot[1], c, fold))
            if lo < hi_so_far - 1e-6:
                failures.append(f"{stem}_OOD_YAW_L{level}/{label}: |dyaw| starts at {lo:.2f} deg, "
                                f"inside level {level - 1}'s reach {hi_so_far:.2f} -- bands overlap")
                tag += " OVERLAP"
            if level == YAW_LADDER_LEVELS and hi < far - 1e-3:
                failures.append(f"{stem}_OOD_YAW_L{level}/{label}: last band reaches {hi:.2f} deg, "
                                f"short of the far side of the orbit {far:.2f}")
                tag += " SHORT"
            reach[label][1] = hi
            print(f"     L{level}       {label:34s} |dyaw| {lo:6.2f}..{hi:6.2f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=sorted(STEMS))
    ap.add_argument("--mode", choices=("rungs", "graded"), default="rungs")
    ap.add_argument("--n_draws", type=int, default=N_DRAWS)
    args = ap.parse_args()

    failures = []
    for task in args.tasks:
        if task not in STEMS:
            failures.append(f"{task}: unknown task")
            continue
        if args.mode == "rungs" and task in SKIP_RUNGS:
            print(f"\n### {task}: SKIPPED (sampler ranges come from bin geometry, not the dict)")
            continue
        print(f"\n### {task}")
        if args.mode == "rungs":
            check_rungs(task, args.n_draws, failures)
        else:
            check_graded(task, args.n_draws, failures)

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print("  " + f)
        return 1
    print("OK: every window checked is the one the env actually samples from")
    return 0


if __name__ == "__main__":
    sys.exit(main())
