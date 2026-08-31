"""Prove that every OOD env differs from its ID env ONLY in object placement.

Runs over all three single-factor rungs per task (OOD-POS, OOD-YAW, OOD-BOTH).

The study's claim is that an ID->OOD drop is caused by where the manipulated object starts. That
only holds if nothing else moved: no recoloured object, no swapped mesh, no different camera pose,
no changed lighting, mass or friction. This script checks that directly rather than by reading the
class definitions.

Two independent comparisons per task:

  1. MODEL XML. The compiled scene description of both envs, with every `pos=` / `quat=` /
     `axisangle=` / `euler=` attribute stripped out (that is exactly the placement channel the OOD
     env is supposed to change) plus absolute asset paths normalised. Anything left that differs --
     an rgba, a mesh file, a texture, a light, a camera, a geom size -- is an unintended OOD factor.

  2. MUJOCO MODEL ARRAYS after reset(): appearance (geom/material rgba, texture ids), geometry
     (geom types and sizes), physics (masses, inertias, frictions, solver options) and sensing
     (camera pose/fov, light pose/colour). Pose arrays (body_pos/body_quat/qpos) are reported
     separately as the expected placement channel, never as a failure.

Run under SLURM (MuJoCo needs an EGL device): sbatch mg_audit_ood_nonplacement.job
"""
import re

import numpy as np
import robosuite as suite

from mimicgen.envs.robosuite.ood_ladder import TASK_LADDER

# compared and REQUIRED to match: everything that is not where an object starts
CHECKED = [
    # appearance
    "geom_rgba", "geom_matid", "mat_rgba", "mat_texid", "mat_texrepeat", "tex_type",
    # geometry / identity
    "geom_type", "geom_size", "geom_dataid", "mesh_vertnum", "mesh_facenum",
    # physics
    "body_mass", "body_inertia", "geom_friction", "geom_solimp", "geom_solref",
    "dof_damping", "jnt_range", "opt.timestep", "opt.gravity",
    # sensing
    "cam_pos", "cam_quat", "cam_fovy", "light_pos", "light_dir",
    "light_diffuse", "light_ambient", "light_specular",
]
# reported only: this IS the channel the OOD env is meant to change
PLACEMENT = ["body_pos", "body_quat"]

POSE_ATTR = re.compile(r'\s(?:pos|quat|axisangle|euler|xyaxes|zaxis)="[^"]*"')
ABS_PATH = re.compile(r'(file|meshdir|texturedir)="[^"]*/([^/"]+)"')


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


def snapshot(env_name, seed=0):
    env = make(env_name)
    np.random.seed(seed)
    env.reset()
    model = env.sim.model
    out = {}
    for key in CHECKED + PLACEMENT:
        if key.startswith("opt."):
            val = getattr(model.opt, key.split(".", 1)[1], None)
        else:
            val = getattr(model, key, None)
        out[key] = None if val is None else np.array(val, dtype=float, copy=True)
    xml = env.sim.model.get_xml()
    xml = POSE_ATTR.sub("", xml)
    xml = ABS_PATH.sub(r'\1="\2"', xml)
    out["_xml"] = xml
    env.close()
    return out


failures = []
for task, (id_env, *ood_envs) in TASK_LADDER.items():
    a = snapshot(id_env)
    for ood_env in ood_envs:
        print(f"\n{'=' * 92}\n{task}:  {id_env}  vs  {ood_env}\n{'=' * 92}")
        b = snapshot(ood_env)

        same_xml = a["_xml"] == b["_xml"]
        print(f"  model xml (pose attributes stripped): {'IDENTICAL' if same_xml else 'DIFFERS'}"
              f"   [{len(a['_xml'])} vs {len(b['_xml'])} chars]")
        if not same_xml:
            la, lb = a["_xml"].splitlines(), b["_xml"].splitlines()
            shown = 0
            for i, (x, y) in enumerate(zip(la, lb)):
                if x != y and shown < 5:
                    print(f"     line {i}:\n       ID : {x.strip()[:140]}\n       OOD: {y.strip()[:140]}")
                    shown += 1
            if len(la) != len(lb):
                print(f"     (line counts differ: {len(la)} vs {len(lb)})")
            failures.append(f"{task}: model xml differs outside pose attributes")

        bad = []
        for key in CHECKED:
            x, y = a[key], b[key]
            if x is None and y is None:
                continue
            if x is None or y is None or x.shape != y.shape or not np.allclose(x, y, atol=0, rtol=0):
                d = "shape" if (x is None or y is None or x.shape != y.shape) else f"max|d|={np.abs(x - y).max():.3g}"
                bad.append(f"{key}({d})")
        print(f"  non-placement model arrays ({len(CHECKED)} checked): "
              f"{'ALL IDENTICAL' if not bad else 'DIFFER -> ' + ', '.join(bad)}")
        if bad:
            failures.append(f"{task}: {', '.join(bad)}")

        moved = []
        for key in PLACEMENT:
            x, y = a[key], b[key]
            if x is not None and y is not None and x.shape == y.shape and not np.allclose(x, y):
                moved.append(f"{key}(max|d|={np.abs(x - y).max():.3f})")
        print(f"  placement arrays (expected to differ): {', '.join(moved) if moved else 'identical this draw'}")

print(f"\n{'=' * 92}")
if failures:
    print("UNINTENDED OOD FACTORS FOUND:")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("OK: ID and OOD envs differ in object placement ONLY "
      "(colour, texture, mesh, size, mass, friction, camera and lighting all identical)")
