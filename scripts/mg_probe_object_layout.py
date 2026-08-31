"""Recover the layout of `obs/object` for each task's ID env, empirically.

Why this exists
---------------
The aux targets are cut out of `obs/object` with a hardcoded position slice [:, :3] and quaternion
slice [3:7] -- i.e. "the first block is the manipulated object". That is true for Square and Hammer,
which is where the aux code grew up, but it is NOT true across the 12 tasks: three_piece_assembly's
first block is the static `base` fixture and kitchen's is the static `Stove1`, while pick_place and
nut_assembly have several objects that all get manipulated. Training aux on the wrong slice asks the
model to predict something that never moves.

The offsets cannot be copied from the equi_diffpo study either: that used abs-converted robomimic
datasets whose object obs is 14 columns per object, while these datasets carry robosuite's native
`object-state` straight from each env's `_setup_observables` (stack_d1 is 23 wide, hammer 28).

Method: reset each env, then match every individual observable against slices of `object-state` by
VALUE. That recovers the layout without trusting any assumption about ordering, and the match is
its own verification -- a component that cannot be located is reported rather than guessed at.

Run under SLURM (MuJoCo needs an EGL device): sbatch mg_probe_object_layout.job
"""
import numpy as np
import robosuite as suite

from mimicgen.envs.robosuite.ood_ladder import TASK_LADDER

TOL = 1e-9


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


def layout_of(env_name, n_resets=3):
    """-> (list of (offset, name, dim), object-state width, list of unmatched offsets)

    Several resets are matched jointly: a single reset can pair a component with the wrong slice by
    coincidence (a zero-valued quaternion component, two objects sharing a coordinate), and a
    candidate has to line up in EVERY reset to be accepted.
    """
    env = make(env_name)
    try:
        obs_list = [env.reset() for _ in range(n_resets)]
    finally:
        env.close()

    width = obs_list[0]["object-state"].shape[0]
    # Robot observables are excluded, not just deprioritised: `robot0_gripper_qvel` is [0, 0] at
    # reset and matched the first two entries of an object quaternion (also [0, 0]) in all three
    # resets, which silently split a 4-wide quat into a bogus 2-wide hit plus two unmatched slots.
    # object-state only ever contains object observables anyway.
    names = [k for k, v in obs_list[0].items()
             if k != "object-state" and not k.startswith("robot0_")
             and isinstance(v, np.ndarray) and v.ndim == 1]
    # longest first, so a 4-wide quat is preferred over any 3-wide prefix that happens to agree
    names.sort(key=lambda k: -obs_list[0][k].shape[0])

    # Two passes. FIRST match each object's [pos(3) | quat(4)] as one contiguous 7-wide pair:
    # robosuite lays every object out that way, and matching the halves independently let one
    # object's quat be claimed at another object's offset (Stove1 came out with "quat 45", 45
    # slots away from its own position). Only then fall back to matching single observables.
    pairs = {}
    for name in names:
        for suf in ("_pos", "_quat"):
            if name.endswith(suf):
                pairs.setdefault(name[: -len(suf)], {})[suf] = name
    pairs = {k: v for k, v in pairs.items() if "_pos" in v and "_quat" in v}

    layout, unmatched, off, taken = [], [], 0, set()
    while off < width:
        hit = None
        for obj, nm in pairs.items():
            if obj in taken:
                continue
            pos_n, quat_n = nm["_pos"], nm["_quat"]
            dp, dq = obs_list[0][pos_n].shape[0], obs_list[0][quat_n].shape[0]
            if off + dp + dq > width:
                continue
            if all(np.allclose(o["object-state"][off:off + dp], o[pos_n], atol=TOL, rtol=0)
                   and np.allclose(o["object-state"][off + dp:off + dp + dq], o[quat_n],
                                   atol=TOL, rtol=0)
                   for o in obs_list):
                hit = (pos_n, dp, quat_n, dq, obj)
                break
        if hit is not None:
            pos_n, dp, quat_n, dq, obj = hit
            layout.append((off, pos_n, dp))
            layout.append((off + dp, quat_n, dq))
            taken.add(obj)
            off += dp + dq
            continue
        # no object pair starts here -- fall back to any single observable
        single = None
        for name in names:
            dim = obs_list[0][name].shape[0]
            if off + dim > width:
                continue
            if all(np.allclose(o["object-state"][off:off + dim], o[name], atol=TOL, rtol=0)
                   for o in obs_list):
                single = (name, dim)
                break
        if single is None:
            unmatched.append(off)
            off += 1
            continue
        layout.append((off, single[0], single[1]))
        off += single[1]
    return layout, width, unmatched


print(f"{'task':26s} {'env':34s} width")
print("=" * 100)
summary = {}
for task, (id_env, *_) in TASK_LADDER.items():
    layout, width, unmatched = layout_of(id_env)
    print(f"\n{task:26s} {id_env:34s} {width}")
    for off, name, dim in layout:
        mark = ""
        if name.endswith("_pos"):
            mark = "  <- position"
        elif name.endswith("_quat"):
            mark = "  <- quaternion"
        print(f"    [{off:3d}:{off + dim:3d}] {name:32s} dim {dim}{mark}")
    if unmatched:
        print(f"    UNMATCHED offsets: {unmatched}")
    # a manipulated object is one that has BOTH a pos and a quat block
    objs = {}
    for off, name, dim in layout:
        for suf, key in (("_pos", "pos"), ("_quat", "quat")):
            if name.endswith(suf):
                objs.setdefault(name[: -len(suf)], {})[key] = (off, dim)
    full = {k: v for k, v in objs.items() if "pos" in v and "quat" in v}
    print(f"    objects with pos+quat: {sorted(full)}")
    summary[task] = full

print("\n" + "=" * 100)
print("candidate aux slices (pos_offset, quat_offset) per object -- pick the MANIPULATED ones:")
for task, full in summary.items():
    parts = [f"{k}: pos {v['pos'][0]}, quat {v['quat'][0]}" for k, v in sorted(full.items())]
    print(f"  {task:26s} " + " | ".join(parts))
