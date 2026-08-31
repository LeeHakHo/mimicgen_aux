# Reproducing the ID90 / OOD runs

Everything below was captured from the environment that actually produced the
results (`robot_mimic_mg`, CARC Discovery), not reconstructed from memory.

## Source checkouts

Four packages are **editable installs from local checkouts**, not PyPI wheels. All
four must be present and at these commits.

| package | source | commit / branch |
| --- | --- | --- |
| `robomimic` 0.5.0 | https://github.com/LeeHakHo/Robomimic_Async | `feature/id90-12task` |
| `mimicgen` 1.0.0 | https://github.com/LeeHakHo/mimicgen_aux | `main` (this repo) |
| `robosuite` 1.4.1 | https://github.com/ARISE-Initiative/robosuite | `b9d8d3de` (`v1.3-45-gb9d8d3de`) **+ texture patch, see below** |
| `robosuite_task_zoo` 0.1 | https://github.com/ARISE-Initiative/robosuite-task-zoo | `74eab7f` |

`robosuite_task_zoo` is not optional: `hammer_cleanup` and `kitchen` come from it, and
`mimicgen/__init__.py` degrades with a warning rather than an error when it is absent,
so those two tasks would silently be missing from `REGISTERED_ENVS`.

```bash
conda env create -f setup_env/environment_id90.yml     # or: -n <name> and install the pins below
conda activate robot_mimic_mg
pip install -e /path/to/robosuite_mg
pip install -e /path/to/robosuite_task_zoo
pip install -e /path/to/mimicgen_aux
pip install -e /path/to/Robomimic_Async
```

## The robosuite texture patch

The runs used two texture files that differ from robosuite `b9d8d3de`:

| file | upstream md5 | **used here** md5 |
| --- | --- | --- |
| `robosuite/models/assets/textures/cereal.png` | `e95ced814cb69b6fa9060452b13a3d70` | `94d62bdde54befd0bd8d5e4eeb32fbc8` |
| `robosuite/models/assets/textures/soda.png` | `df3a6db7ddc265d7209ffe420d1aabb1` | `1a2adfdadc322538fee12d1a9b206675` |

They are referenced by `cereal.xml` and `can.xml`, i.e. **`pick_place_d0`'s objects**.
The training data is 84×84 camera images, so a different texture is a different
observation: reproducing `pick_place_d0` against stock robosuite will not reproduce
these pixels. Copies are shipped in `setup_env/patches/robosuite_textures/`; apply them
after checking out robosuite:

```bash
cp setup_env/patches/robosuite_textures/*.png <robosuite>/robosuite/models/assets/textures/
```

## Pins that matter

```
python 3.10.20
numpy 1.26.4          <- see below, this one is load-bearing
torch 2.4.0+cu121, torchvision 0.19.0+cu121
mujoco 2.3.2
h5py 3.16.0, scipy 1.15.3, imageio 2.37.3, imageio-ffmpeg 0.6.0
opencv-python-headless 4.10.0.84, egl_probe 1.0.2
```

**numpy must stay on 1.x.** Upgrading this environment to numpy 2.2.6 previously
collapsed evaluation success rates in this line of work (a baseline went 0.29 → 0.02)
while training looked normal, and the environment had to be pinned back to 1.26.4. If
your numbers are near zero but training loss looks fine, check numpy first.

`setup_env/requirements-frozen.txt` is the full `pip freeze` (editable installs stripped;
they are the table above). `setup_env/environment_id90.yml` is the conda export.

## Runtime environment

```bash
export MUJOCO_GL=egl          # headless rendering; the jobs all set this
export OPENBLAS_NUM_THREADS=1 # OpenBLAS spawns 64 threads by default and fails
export OMP_NUM_THREADS=1      # numpy import outright on a busy shared node
```

`robosuite` prints a "No private macro file found" warning on import; it is harmless.
Running `setup_macros.py` writes a private macro file that can change rendering
defaults, so the runs here deliberately left it unset.

## Data

Generated demonstrations and fixed evaluation scenes:
**https://huggingface.co/datasets/LeeHakHo/mimicgen_aux_data**

```
id90/<task>/demo.hdf5                200 demos, 84x84, already includes obs/object
eval_scenes/<task>/scenes_<rung>.hdf5   ID 60 + 20 per OOD rung
```

Use the shipped eval scenes rather than regenerating them. Regeneration is only
bit-identical under the same robosuite/MuJoCo/numpy stack, and the numpy episode above
is exactly the kind of thing that silently breaks it.
