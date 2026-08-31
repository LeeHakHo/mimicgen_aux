# ID90 / OOD aux study — 12 MimicGen tasks

This repo carries the MimicGen side of a study that trains one policy per
`(task, arm)` on a rotation-restricted **ID90** distribution and scores it on three
**single-factor OOD** rungs, to find where an auxiliary object-state head buys
generalization and where it does not.

It generalizes the one regime in the predecessor work that produced a gap above the
noise floor: `HammerCleanup_Yaw45_Spawn12` → `HammerCleanup_OOD_Spawn20_Yaw90`,
baseline 0.57 vs `aux_obj_eef` 0.77, with obj_eef ahead at every evaluated epoch
rather than only at its best one.

## The ladder

`mimicgen/envs/robosuite/ood_ladder.py` defines, for each of the 12 tasks:

| env | role | placement |
| --- | --- | --- |
| `<Base>_ID90` | training | primary object's yaw restricted to a 90° window on its own declared center; x/y untouched |
| `<Base>_OOD_POS` | eval | x/y widened 1.67× about their own centers, yaw held at ID |
| `<Base>_OOD_YAW` | eval | yaw widened to a 180° window, x/y held at ID |
| `<Base>_OOD_BOTH` | eval | both |

**The axis a rung does not move equals ID exactly** — every object starts from
`_id_spec()` — which is what makes POS and YAW separately attributable. Only the
primary object leaves the distribution; target objects keep the ID distribution, so
the ID→OOD drop is attributable to the manipulated object alone.

Two rungs are degenerate by construction and equal ID: `coffee_d2`'s `_OOD_YAW`
(coffee_pod is a body of revolution) and `pick_place`'s `_OOD_POS` (its x/y range is
the bin interior).

The ladder lives inside the `mimicgen` package, not in the downstream study, because
data generation runs `mimicgen/scripts/generate_dataset.py`, which imports `mimicgen`
and never touches the policy code. `Square_D2_ID90` resolves only because
`mimicgen/__init__.py` imports `ood_ladder`, and robosuite's `EnvMeta` metaclass
registers each subclass into `REGISTERED_ENVS` at class-definition time. Generating
demos *in* the window (rather than filtering a pre-collected set down to it) is also
why the `z_rot_center_override` the equi_diffpo original needed on `square_d2` and
`mug_cleanup_d1` is dropped here: a filter cannot create demos where MimicGen's
rejection sampling left the window sparse, but generation can.

## Pipeline (`scripts/`)

```
mg_download_source_datasets.sh    source demos from HuggingFace
mg_prepare_9tasks.job             convert / prepare the source sets
mg_make_id90_configs.py           MimicGen datagen configs, one per task
mg_gen_id90_12task.job            200 demos per task at 84x84
mg_make_train_configs.py          training configs from the REAL episode lengths
mg_train_id90_12task.job          48 arms = 12 tasks x {baseline, world, eef, obj_eef}
mg_gen_eval_scenes_id90.job       fixed scenes: ID 60 + 20 per OOD rung
mg_eval_id90.job                  4 rungs x 3 seeds x 10 checkpoints per arm
mg_pipeline_supervisor.{sh,job}   drives the above per task as each becomes ready
```

There is no separate render step: at 84px datagen's own `demo.hdf5` already carries
`agentview`, `eye_in_hand` and `obs/object`, structurally identical to a
`dataset_states_to_obs` output bar `camera_info`.

### Correctness scaffolding

The failure modes here are silent, so each is checked rather than assumed:

- `mg_probe_object_layout.py` recovers each task's `obs/object` layout empirically,
  matching observables against object-state by value. Block 0 is *not* the
  manipulated object everywhere: `three_piece_assembly` and `kitchen` lead with a
  static fixture, and `pick_place` / `nut_assembly` move several objects.
- `mg_check_obj_blocks.py` refuses to submit training if a configured aux block does
  not actually move in the data. It measures the *configured* offsets: a stride-7
  sweep alone misses `stack_three`'s cubeC at offset 23 and calls a moving object static.
- `mg_verify_ood_ladder.py` asserts the window arithmetic, the single-factor
  invariants, the target exemption, on-table containment, and that the generic rule
  reproduces the hand-written hammer ladder.
- `mg_audit_ood_nonplacement.py` compares ID against OOD over 36 model arrays —
  colour, texture, mesh, size, mass, friction, camera, lighting — so a rung differs
  in placement and in nothing else.

## Reproducing

- **`setup_env/ENVIRONMENT.md`** — the exact stack: four editable source checkouts with
  their commits, the version pins (numpy must stay on 1.x — 2.x collapsed evaluation in
  this line of work), the required env vars, and a robosuite texture patch that
  `pick_place_d0`'s observations depend on.
- **Data** — https://huggingface.co/datasets/LeeHakHo/mimicgen_aux_data carries the
  generated demonstrations and the fixed evaluation scenes, so training and evaluation
  can be reproduced without re-running datagen.

## Dependency

The policy-side code (the aux heads, the multi-object target blocks, the point-cloud
geometries, `eval_fixed_scenes`) is a robomimic fork and stays there:

```
LeeHakHo/Robomimic_Async   branch feature/id90-12task
```

`scripts/` here is a copy of what runs in that checkout; the job files carry absolute
CARC paths (`/scratch1/$USER/...`, `--account=`, `--constraint=`) and are meant to be
read as the exact recipe that produced the results, not as portable scripts. Point
them at your own paths before running.

## Settings that matter

- Training: 500 epochs, checkpoint every 10, crop 76, GMM head. Rollouts during
  training are a liveness check only (`n=1`); all reported numbers come from the
  fixed-scene evaluation.
- Evaluation: fixed scenes, ID 60 + 20 per OOD rung, 3 rollout seeds, epochs 50…500.
  One `eval_fixed_scenes` call per rung carries all 10 checkpoints of an arm — the env
  is built once and `--agents` is looped over.
- **GPU architecture is a real nuisance variable here**: the same checkpoint has been
  measured at 46 vs 36 across jobs on mixed architectures. Training is pinned to one
  arch so a task's four arms are comparable, and the GPU name is recorded into every
  result JSON so an arm/arch correlation stays detectable.
