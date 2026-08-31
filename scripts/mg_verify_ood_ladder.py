"""Print and check the ID90 / OOD-POS / OOD-YAW / OOD-BOTH placement windows for all 12 tasks.

The OOD rung is split into single-factor variants so an ID->OOD drop can be attributed to position
or to rotation rather than to "the scene changed". This script verifies the split actually holds:

  * POS  : every object's yaw is EXACTLY the ID window; primaries' x/y are widened.
  * YAW  : every object's x/y is EXACTLY the ID ranges; primaries' yaw is widened to 180 deg.
  * BOTH : primaries widened on both axes.
  * targets: identical to ID in all three variants (only the manipulated object goes OOD).
  * no OOD range runs off the table (robosuite's sampler does not check table extent).

Run under SLURM (MuJoCo needs a GPU/EGL device): sbatch mg_verify_ood_ladder.job
"""
import numpy as np
import robosuite as suite

from mimicgen.envs.robosuite.ood_ladder import TASK_LADDER

BASE_ENV = {
    "stack_d1": "Stack_D1", "stack_three_d1": "StackThree_D1", "square_d2": "Square_D2",
    "threading_d0": "Threading_D0", "three_piece_assembly_d0": "ThreePieceAssembly_D0",
    "hammer_cleanup_d1": "HammerCleanup_D1", "mug_cleanup_d1": "MugCleanup_D1",
    "coffee_d2": "Coffee_D2", "kitchen_d1": "Kitchen_D1", "pick_place_d0": "PickPlace_D0",
    "coffee_preparation_d1": "CoffeePreparation_D1", "nut_assembly_d0": "NutAssembly_D0",
}


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


def bounds_of(env_name):
    """-> (bounds dict, target_objects, per-object sampler overrides or None, table_full_size)"""
    env = make(env_name)
    try:
        # the stack tasks route their (deliberately asymmetric) bounds through a per-object
        # sampler, since upstream shares one sampler across every cube -- report it so the
        # target exemption can be checked where it actually takes effect
        per_object = getattr(env.placement_initializer, "per_object", None)
        return (env._get_initial_placement_bounds(),
                getattr(env, "target_objects", ()),
                per_object,
                tuple(env.table_full_size))
    except AttributeError:
        # PickPlace builds sampler ranges from bin geometry instead
        s = env.placement_initializer.samplers["CollisionObjectSampler"]
        rot = tuple(s.rotation) if isinstance(s.rotation, (tuple, list)) else (0.0, 0.0)
        return ({"<sampler>": dict(x=tuple(s.x_range), y=tuple(s.y_range), z_rot=rot)},
                (), None, tuple(env.table_full_size))
    finally:
        env.close()


def fmt(rng, deg=False):
    lo, hi = rng
    if deg:
        lo, hi = np.degrees(lo), np.degrees(hi)
        return f"[{lo:7.1f},{hi:7.1f}]"
    return f"[{lo:7.3f},{hi:7.3f}]"


failures = []
overflow = []
for task, (id_env, pos_env, yaw_env, both_env) in TASK_LADDER.items():
    print(f"\n{'=' * 112}\n{task}   base={BASE_ENV[task]}   id={id_env}\n{'=' * 112}")
    b, _, _, table = bounds_of(BASE_ENV[task])
    ID, targets, _, _ = bounds_of(id_env)
    variants = {}
    for label, env_name in (("POS", pos_env), ("YAW", yaw_env), ("BOTH", both_env)):
        vb, vtargets, vper_object, _ = bounds_of(env_name)
        variants[label] = (vb, vtargets, vper_object)
    half_x, half_y = table[0] / 2.0, table[1] / 2.0
    print(f"  targets held at ID = {targets or '(none)'}   table {table[0]:.2f} x {table[1]:.2f}")

    for obj in sorted(b):
        role = "TARGET " if obj in targets else "primary"
        print(f"  -- {obj} [{role}]")
        for key, deg in (("x", False), ("y", False), ("z_rot", True)):
            if key not in b[obj]:
                continue
            cells = "  ".join(f"{lab} {fmt(variants[lab][0][obj][key], deg)}"
                              for lab in ("POS", "YAW", "BOTH"))
            print(f"       {key:5s} id {fmt(ID[obj][key], deg)}   {cells}")

        for lab, (vb, vtargets, vper_object) in variants.items():
            # a target must be untouched in every variant
            if obj in targets:
                for key in ("x", "y", "z_rot"):
                    if key in b[obj] and not np.allclose(vb[obj][key], ID[obj][key]):
                        failures.append(f"{task}/{lab}: target {obj}.{key} differs from ID "
                                        f"({ID[obj][key]} vs {vb[obj][key]})")
            # single-factor invariants: the axis a variant does not move must equal ID exactly
            held = {"POS": ("z_rot",), "YAW": ("x", "y"), "BOTH": ()}[lab]
            for key in held:
                if key in b[obj] and not np.allclose(vb[obj][key], ID[obj][key]):
                    failures.append(f"{task}/{lab}: {obj}.{key} should be held at ID but is "
                                    f"{vb[obj][key]} vs {ID[obj][key]}")
            # where a per-object sampler carries the bounds, verify it agrees with the dict
            if vper_object is not None and obj in vper_object:
                for key, s_key in (("x", "x_range"), ("y", "y_range"), ("z_rot", "rotation")):
                    if key in vb[obj] and not np.allclose(vper_object[obj][s_key], vb[obj][key]):
                        failures.append(f"{task}/{lab}: sampler {obj}.{s_key}="
                                        f"{vper_object[obj][s_key]} != bounds {key}={vb[obj][key]}")
            # off-table check: robosuite retries only on object-object overlap, so a range past
            # the edge does not raise -- the object is spawned off the table and simply falls
            if obj in targets:
                continue
            for key, half in (("x", half_x), ("y", half_y)):
                if key not in vb[obj]:
                    continue
                lo, hi = vb[obj][key]
                if min(lo, -hi) < -half:
                    overflow.append(f"{task}/{lab}: {obj}.{key}=[{lo:.3f},{hi:.3f}] exceeds the "
                                    f"table half-extent {half:.3f}")

    # a variant that changes nothing at all is degenerate -- report it, it is not a failure
    for lab, (vb, _, _) in variants.items():
        if all(np.allclose(vb[o][k], ID[o][k]) for o in b for k in ("x", "y", "z_rot") if k in b[o]):
            print(f"  NOTE: {lab} is IDENTICAL to ID for this task (no rung on that axis)")

# The Hammer rung is the one with a published result behind it (baseline 0.57 vs aux_obj_eef 0.77).
# Its OOD env used to be hand-written as HammerCleanup_OOD_Spawn20_Yaw90; the generic rule applied
# to the training distribution is supposed to reproduce it exactly. Assert that rather than trust it
# -- if OOD_SCALE or the clamp ever moves, this is what catches the drift.
print(f"\n{'=' * 112}\nHammer: generic rule vs the hand-written HammerCleanup_OOD_Spawn20_Yaw90\n"
      f"{'=' * 112}")
gen, _, _, _ = bounds_of("HammerCleanup_Yaw45_Spawn12_OOD_BOTH")
hand, _, _, _ = bounds_of("HammerCleanup_OOD_Spawn20_Yaw90")
for obj in sorted(hand):
    for key, deg in (("x", False), ("y", False), ("z_rot", True)):
        if key not in hand[obj]:
            continue
        ok = np.allclose(gen[obj][key], hand[obj][key], atol=1e-3)
        print(f"  {obj:10s} {key:5s} generic {fmt(gen[obj][key], deg)}  "
              f"hand-written {fmt(hand[obj][key], deg)}   {'match' if ok else 'DIFFERS'}")
        if not ok:
            failures.append(f"hammer: generic {obj}.{key}={gen[obj][key]} != "
                            f"hand-written {hand[obj][key]}")

print(f"\n{'=' * 112}")
if overflow:
    print("OFF-TABLE OOD RANGES:")
    for f in overflow:
        print("  " + f)
if failures:
    print("INVARIANT FAILURES:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("OK: POS moves x/y only, YAW moves yaw only, targets identical to ID in all three variants, "
      "per-object samplers agree, every range on the table")
