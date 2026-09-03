"""ID-restricted (training) and OOD (evaluation) placement variants for the 12 MimicGen tasks.

Ported from the equi_diffpo study (`equi_diffpo/env/mimicgen_id_restricted.py` +
`mimicgen_ood.py`) into the MimicGen fork so that *data generation* -- not just evaluation --
can run against these distributions. Generalizes the HammerCleanup_Yaw45_Spawn12 ->
HammerCleanup_OOD_Spawn20_Yaw90 ladder (the one regime that produced a baseline-vs-aux gap
above the noise floor: 0.57 -> 0.77) to every task.

The ladder
----------
  <Base>_ID90 : TRAINING distribution. The primary object's z_rot is restricted to a 90-degree
                window centered on its own declared center. x/y untouched. Demos are GENERATED
                inside this window by generate_dataset.py, so every training demo is in-window
                by construction.
  <Base>_OOD  : EVALUATION distribution. The primary object's x/y ranges are widened 2x about
                their own centers, and its z_rot is SET to a 180-degree window on the same
                center (scaling a full circle is a no-op mod 2pi, so rotation OOD has to be
                built by widening the *restricted* window instead).

Only the PRIMARY (manipulated) object is pushed out of distribution. The target object -- the
receptacle or base the primary is placed into/onto -- keeps the ID distribution, so the ID->OOD
drop is attributable to the manipulated object alone. `target_objects` names them per task.

Mechanism: MimicGen tasks sample initial poses via `_get_initial_placement_bounds()`, a dict of
`{obj_name: {x: (lo, hi), y: (lo, hi), z_rot: (lo, hi), reference: ...}}` consumed by
`_get_placement_initializer()`. These mixins rewrite that dict. Subclassing an env whose
metaclass is robosuite's EnvMeta auto-registers the class in REGISTERED_ENVS, so
`robosuite.make("MugCleanup_D1_ID90", ...)` and `env_meta['env_name'] = ...` work unchanged.

Difference from the equi_diffpo original
----------------------------------------
That study FILTERED pre-collected 1000-demo datasets down to the 90-degree window, so it needed
`z_rot_center_override = 15deg` on square_d2 / mug_cleanup_d1: MimicGen's rejection sampling had
left the window around the declared center sparse, and a filter cannot create demos where there
are none. Here the demos are generated, not filtered, so any window is equally cheap and the
override is dropped -- every ID90 window sits on its object's own declared center.
"""
import numpy as np

from robosuite.environments.manipulation.stack import Stack
from robosuite.utils.placement_samplers import UniformRandomSampler

from mimicgen.envs.robosuite.stack import Stack_D1, StackThree_D1
from mimicgen.envs.robosuite.nut_assembly import Square_D2, NutAssembly_D0
from mimicgen.envs.robosuite.threading import Threading_D0
from mimicgen.envs.robosuite.three_piece_assembly import ThreePieceAssembly_D0
from mimicgen.envs.robosuite.hammer_cleanup import (
    HammerCleanup_D1, HammerCleanup_Yaw45_Spawn12,
    HammerCleanup_FixedHead_Yaw45_Spawn25)
from mimicgen.envs.robosuite.mug_cleanup import MugCleanup_D1
from mimicgen.envs.robosuite.coffee import Coffee_D2, CoffeePreparation_D1
from mimicgen.envs.robosuite.kitchen import Kitchen_D1
from mimicgen.envs.robosuite.pick_place import PickPlace_D0

EPS = 1e-6
ID_ROT_HALF_WIDTH = np.pi / 4.0   # -> 90 deg training window
OOD_ROT_HALF_WIDTH = np.pi / 2.0  # -> 180 deg evaluation window
#   Both are divided by the primary object's `yaw_fold` (its N-fold symmetry about z). An object
#   with N-fold symmetry has an orientation ORBIT of 360/N degrees, not 360: yaw and yaw+360/N are
#   the same physical scene. A window as wide as the orbit therefore covers every distinguishable
#   orientation, and widening it is a no-op -- which is exactly what happened to the two stack
#   tasks, whose cubes are exact cubes (C4, a 90 degree orbit) against a 90 degree ID window: all
#   four arms scored ood_yaw >= id there while ood_pos dropped hard, because the yaw rung was not
#   a distribution shift at all. Dividing by the fold keeps the rung at the SAME FRACTION of the
#   orbit for every task -- ID a quarter of it, OOD a half -- so a cube gets 22.5 / 45 deg where an
#   asymmetric object gets 90 / 180, and the tasks stay comparable. The aux head's rotation target
#   should carry the matching `rotation_fold` (robomimic dataset.py `_quat_to_yaw_sincos`), or its
#   label is not a function of the image.
OOD_SCALE = 1.67                  # x/y half-width multiplier, center held fixed
#   Matches the Hammer study that produced this project's only above-noise gap (baseline 0.57 vs
#   aux_obj_eef 0.77 on HammerCleanup_OOD_Spawn20_Yaw90): its training box was 1.2x the D0
#   half-width and its OOD box 2.0x, i.e. a 1.67x train->OOD x/y rung, alongside a 2x rotation
#   rung (+-45 -> +-90 deg). OOD_ROT_HALF_WIDTH below reproduces that rotation rung.
OOD_TABLE_MARGIN = 0.05           # metres kept between an OOD x/y range and the table edge
#   This bounds the sampled object CENTRE -- the sampler has no notion of object extent here --
#   so the margin has to cover the object's own half-width or the object can still overhang the
#   edge. 0.05 keeps every object in this set fully on the table. It costs x/y range on the three
#   tasks whose ID box already sits near the edge: against the nominal 1.67x, square_d2 clamps to
#   1.40x, three_piece to 1.59x and mug_cleanup's x to 1.50x.


def fixed_width_range(lo, hi, half, never_widen=False):
    """Re-center (lo, hi) to exactly 2*half wide about its own center.

    A fixed (zero-width) range is returned unchanged -- there is no rotation there to restrict or
    widen. `never_widen` leaves a range alone when it is ALREADY no wider than the target window;
    the ID side needs that guard (an object whose native z_rot spans 60 degrees must not come out
    of an "ID restriction" at 90), the OOD side must not have it, since widening is the point.
    """
    if hi - lo <= EPS:
        return (lo, hi)
    if never_widen and (hi - lo) <= 2.0 * half + EPS:
        return (lo, hi)
    center = (lo + hi) / 2.0
    return (center - half, center + half)


def scale_range(lo, hi, scale):
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * scale
    return (center - half, center + half)


def clamp_to_table(lo, hi, table_half, margin=OOD_TABLE_MARGIN):
    """Keep an OOD x/y range on the table.

    robosuite's sampler does NOT check table extent -- `ensure_valid_placement` only retries on
    object-object overlap -- so a range that runs off the edge does not raise. The object is simply
    spawned past the edge and falls, which is a different (and unintended) kind of OOD than the
    x/y + yaw shift being studied. Measured before this cap: three_piece's pieces and mug_cleanup's
    mug reached +-0.44 / -0.45 on an 0.8 m table (half-extent 0.40).

    The cap is applied symmetrically around the range's own center so the OOD window stays centered
    on the ID one; a range already inside the table is returned untouched.
    """
    limit = table_half - margin
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    # the ID center itself can sit near the edge (mug_cleanup's mug is off-center); cap the
    # half-width by the distance from the center to the nearer edge, never past it
    half = min(half, max(0.0, min(limit - center, limit + center)))
    return (center - half, center + half)


class SquarePegPerResetMixin:
    """Re-randomize square_d2's peg on every reset instead of once per env process.

    The peg has no free joint, so its pose exists only in the compiled model and `Square_D1
    ._load_model` places it by editing the arena xml before compilation. robosuite only calls
    `_load_model()` on a HARD reset, so with `hard_reset=False` (what the rollout runners set to
    keep memory flat) the peg is sampled once per env process and never moves again -- an
    evaluation of 156 episodes over 12 envs would see 12 peg poses, not 156. Every other fixture
    in this task set is placed in `_reset_internal`, which runs every episode; this gives the peg
    the same treatment, sampling from whatever bounds the MRO provides.
    """

    peg_body_name = "peg1"

    def _reset_internal(self):
        super()._reset_internal()
        # robosuite sets `deterministic_reset` while rebuilding a scene from a recorded xml
        # (reset_from_xml_string -> _initialize_sim -> reset). Without this guard a demo-seeded
        # reset -- which is exactly what fixed-scene eval and datagen replay do -- lands on a peg
        # this mixin sampled instead of the one the demo recorded.
        if self.deterministic_reset:
            return
        bounds = self._get_initial_placement_bounds()["peg"]
        sample_x = np.random.uniform(low=bounds["x"][0], high=bounds["x"][1])
        sample_y = np.random.uniform(low=bounds["y"][0], high=bounds["y"][1])
        sample_z_rot = np.random.uniform(low=bounds["z_rot"][0], high=bounds["z_rot"][1])

        body_id = self.sim.model.body_name2id(self.peg_body_name)
        pos = np.array(self.sim.model.body_pos[body_id])
        pos[0] = bounds["reference"][0] + sample_x
        pos[1] = bounds["reference"][1] + sample_y
        self.sim.model.body_pos[body_id] = pos
        self.sim.model.body_quat[body_id] = np.array(
            [np.cos(sample_z_rot / 2), 0., 0., np.sin(sample_z_rot / 2)])
        self.sim.forward()


class ThreePieceAssembly_D0_Yaw(ThreePieceAssembly_D0):
    """Gives the two carried pieces a real yaw band, which D0 does not have.

    Upstream fixes both pieces at exactly 90 deg in D0 and D1, so the ID/OOD ladder built on D0
    would have NO rotation axis for this task at all -- restricting a zero-width range is a no-op
    and so is widening it. `ThreePieceAssembly_D2` shows what MimicGen considers generatable from
    the same source demos: pieces at 90 +- 90 deg. That D2 band is adopted here as the task's
    native range, so the standard ladder falls out of it -- ID restricts it to 90 +- 45, OOD sets
    it back to 90 +- 90.

    D2 also rotates the `base` (+- 45 deg), which is deliberately NOT copied: base is this task's
    target object and must keep the distribution the policy trained on.

    The other two rotation-free primaries in the study are left alone on purpose: coffee_d2's and
    coffee_preparation_d1's `coffee_pod` is a body of revolution (coffee_pod.stl: identical x/y
    extent, radius flat in theta up to facet discretization), so a yaw band on it would change
    neither the image nor the grasp. coffee_preparation gets its rotation axis from `mug` instead,
    and coffee_d2 is x/y-OOD only.
    """

    piece_yaw_half = np.pi / 2.0  # matches ThreePieceAssembly_D2
    yaw_objects = ("piece_1", "piece_2")

    def _get_initial_placement_bounds(self):
        base = super()._get_initial_placement_bounds()
        out = {}
        for obj_name, spec in base.items():
            new_spec = dict(spec)
            if obj_name in self.yaw_objects:
                lo, hi = spec["z_rot"]
                center = (lo + hi) / 2.0
                new_spec["z_rot"] = (center - self.piece_yaw_half, center + self.piece_yaw_half)
            out[obj_name] = new_spec
        return out


class SquarePegFixedMixin:
    """Pins square_d2's peg to the pose the original task gives it, in BOTH the ID and the OOD env.

    Square_D0 does not randomize the peg at all -- it has no `peg` entry in the bounds dict, and the
    peg sits wherever `pegs_arena.xml` puts it: world (0.23, 0.1, 0.85), i.e. offset (0.23, 0.1)
    from this task's table_offset (0, 0, 0.82), unrotated. D1 then randomizes its x/y and D2 its
    z_rot as well.

    Freezing it here removes a moving target from the one task where it is most expensive: the peg
    has no free joint, so it is placed by editing the arena xml at model-compile time, which only
    happens on a hard reset -- and it is precisely the object whose motion added variance to the
    square numbers without being part of what the OOD rung is testing. With it pinned, an ID and an
    OOD episode differ in the nut's pose and nothing else.

    Placed FIRST in the MRO so it runs LAST: each mixin transforms what super() returns, so the
    earliest mixin has the final say. Whatever the ID or OOD rule did to `peg` is overwritten here.
    """

    peg_fixed_xy = (0.23, 0.1)
    peg_fixed_z_rot = 0.0

    def _get_initial_placement_bounds(self):
        bounds = super()._get_initial_placement_bounds()
        out = {name: dict(spec) for name, spec in bounds.items()}
        if "peg" in out:
            x, y = self.peg_fixed_xy
            out["peg"]["x"] = (x, x)
            out["peg"]["y"] = (y, y)
            out["peg"]["z_rot"] = (self.peg_fixed_z_rot, self.peg_fixed_z_rot)
        return out


# ---------------------------------------------------------------- ID (training) distribution


class IDRestrictMixin:
    """Restricts every object's z_rot to a 90-degree window around its own declared center.
    Fixed (zero-width) z_rot dims, and all x/y bounds, are left untouched.

    Applied to every object rather than only the primary: `never_widen=True` makes it a no-op on
    anything whose native range is already at or below 90 degrees, and in this task set only the
    primary objects declare a full circle.
    """

    id_rot_half_width = ID_ROT_HALF_WIDTH
    yaw_fold = 1

    def _get_initial_placement_bounds(self):
        base = super()._get_initial_placement_bounds()
        out = {}
        for obj_name, spec in base.items():
            new_spec = dict(spec)
            if "z_rot" in spec:
                lo, hi = spec["z_rot"]
                new_spec["z_rot"] = fixed_width_range(
                    lo, hi, self.id_rot_half_width / self.yaw_fold, never_widen=True)
            out[obj_name] = new_spec
        return out


class Stack_D1_ID90(IDRestrictMixin, Stack_D1):
    # cubeA/cubeB are exact cubes (BoxObject size_min == size_max), so C4 about z: see yaw_fold
    yaw_fold = 4


class StackThree_D1_ID90(IDRestrictMixin, StackThree_D1):
    # cubeA, cubeB and cubeC are all exact cubes
    yaw_fold = 4


class Square_D2_ID90(SquarePegPerResetMixin, SquarePegFixedMixin, IDRestrictMixin, Square_D2):
    # SquarePegPerResetMixin first: its _reset_internal must win over Square_D2's, and its call to
    # _get_initial_placement_bounds then resolves to the restricted window rather than the native
    # full circle. With the peg pinned by SquarePegFixedMixin that per-reset resampling is a no-op
    # (a zero-width range redraws the same pose); it is kept so the peg is re-asserted every
    # episode rather than trusting a single model compile.
    pass


class NutAssembly_D0_ID90(IDRestrictMixin, NutAssembly_D0):
    pass


class Threading_D0_ID90(IDRestrictMixin, Threading_D0):
    pass


class HammerCleanup_D1_ID90(IDRestrictMixin, HammerCleanup_D1):
    pass


class MugCleanup_D1_ID90(IDRestrictMixin, MugCleanup_D1):
    pass


# The tasks below have no rotation variance for the ID restriction to bite on -- the mixin is a
# no-op there. They are still declared so every task's training set carries a distinct,
# self-describing env_name, and so the OOD side has a symmetric ID counterpart to widen against.
#   pick_place_d0 : upstream already randomizes z_rot over exactly 0-90 degrees
#   coffee_d2     : coffee_pod is a body of revolution -- see ThreePieceAssembly_D0_Yaw
#   kitchen_d1 / coffee_preparation_d1 : the ID window bites on one primary (bread / mug) and not
#                   the other (pot's native 60 deg and coffee_pod's fixed pose are left alone)


class ThreePieceAssembly_D0_ID90(IDRestrictMixin, ThreePieceAssembly_D0_Yaw):
    # off the yaw-enabled base: 90 +- 90 native -> 90 +- 45 training window
    pass


class Coffee_D2_ID90(IDRestrictMixin, Coffee_D2):
    pass


class Kitchen_D1_ID90(IDRestrictMixin, Kitchen_D1):
    pass


class CoffeePreparation_D1_ID90(IDRestrictMixin, CoffeePreparation_D1):
    pass


class PickPlace_D0_ID90(PickPlace_D0):
    # PickPlace does not use _get_initial_placement_bounds() at all (its sampler ranges come from
    # bin geometry in _get_placement_initializer), and its native rotation range is already the
    # 90-degree window. Nothing to restrict.
    pass


# ------------------------------------------------------------- OOD (evaluation) distribution


class TargetObjectsMixin:
    """Holds `target_objects` -- the receptacle or base the primary object is placed into/onto --
    and reproduces the ID window for them.

    Targets are NOT pushed out of distribution: an OOD episode must differ from an ID episode only
    in where the MANIPULATED object starts, so that the ID->OOD drop is attributable to that object
    alone. Their bounds are reproduced exactly as the ID env would sample them (which, since the ID
    restriction only bites on full-circle z_rot and no target declares one, is their native range).

    They are not frozen either: the targets genuinely move in the training data (both drawers span
    +-25 deg, kitchen's stove/button/serving_region all vary), so pinning them would throw away
    variation the policy trained on.
    """

    target_objects = ()
    yaw_fold = 1

    def _check_target_names(self, base):
        unknown = [n for n in self.target_objects if n not in base]
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.target_objects names {unknown}, which are not in "
                f"_get_initial_placement_bounds() ({sorted(base)}) -- a rename upstream would "
                f"otherwise silently turn a target back into an OOD object")

    def _id_spec(self, spec):
        """Exactly what IDRestrictMixin would produce for this object."""
        new_spec = dict(spec)
        if "z_rot" in spec:
            lo, hi = spec["z_rot"]
            new_spec["z_rot"] = fixed_width_range(
                lo, hi, ID_ROT_HALF_WIDTH / self.yaw_fold, never_widen=True)
        return new_spec


class OODPlacementMixin(TargetObjectsMixin):
    """Widens the PRIMARY objects only: x/y by `ood_scale`x (1.67) about their own centers, and
    z_rot SET to a 180-degree window on the same center -- the same 1.67x / 2x pair the Hammer
    study used. Objects named in `target_objects` keep the ID window.

    z_rot is set rather than scaled because the primary object in most of these tasks declares a
    full circle upstream, and scaling a full circle is a no-op after wrapping mod 2pi (uniform on a
    4pi interval reduces to uniform on 2pi) -- it would produce no rotation OOD at all, and on
    coffee_preparation's mug it produced a nonsensical 720-degree window. Setting 180 degrees
    doubles the ID env's 90-degree window on the same center for every task, so the rotation rung
    is uniform across the study. An object whose z_rot is a single fixed point upstream keeps it
    (zero-width ranges are returned unchanged), so those tasks get x/y OOD only.
    """

    ood_scale = OOD_SCALE
    widen_xy = True    # position rung
    widen_yaw = True   # rotation rung

    def _get_initial_placement_bounds(self):
        base = super()._get_initial_placement_bounds()
        self._check_target_names(base)
        out = {}
        for obj_name, spec in base.items():
            # every object starts at exactly what the ID env would sample; only the enabled rungs
            # move. That is what makes the POS / YAW variants clean single-factor comparisons --
            # an axis that is not widened is identical to ID, not merely "close to" it.
            new_spec = self._id_spec(spec)
            if obj_name in self.target_objects:
                out[obj_name] = new_spec
                continue
            if self.widen_xy:
                # table_full_size is unset while a subclass calls this from its own __init__ (the
                # stack tasks do); those ranges are inside the table anyway, so skip the cap
                table = getattr(self, "table_full_size", None)
                for axis, key in enumerate(("x", "y")):
                    if key in spec:
                        lo, hi = scale_range(*spec[key], self.ood_scale)
                        if table is not None:
                            lo, hi = clamp_to_table(lo, hi, table[axis] / 2.0)
                        new_spec[key] = (lo, hi)
            if self.widen_yaw and "z_rot" in spec:
                lo, hi = spec["z_rot"]
                new_spec["z_rot"] = fixed_width_range(lo, hi, OOD_ROT_HALF_WIDTH / self.yaw_fold)
            out[obj_name] = new_spec
        return out


class PerObjectUniformRandomSampler(UniformRandomSampler):
    """UniformRandomSampler that can give individual objects their own x/y/rotation ranges.

    Upstream `Stack_D0.__init__` builds ONE sampler for every cube and asserts their bounds are
    identical, so a stack task cannot hold its base cube at the ID window through the bounds dict
    alone. robosuite's own answer to per-object ranges, SequentialCompositeSampler, is not usable
    here either: `Stack._load_model` registers the cubes with `placement_initializer.add_objects()`,
    which that class raises on.

    So this samples one object at a time through the upstream implementation, swapping the ranges
    in between and feeding already-placed objects back in as `fixtures` -- which keeps the upstream
    collision/validity checking intact rather than reimplementing it.
    """

    def __init__(self, *args, per_object=None, per_object_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        # obj name -> dict with any of x_range / y_range / rotation
        self._per_object = dict(per_object or {})
        self._per_object_fn = per_object_fn

    @property
    def per_object(self):
        """Resolved lazily when a callback is given, so the ranges are always the CURRENT ones.

        The stack envs build this sampler inside `__init__`, before `self.table_full_size` exists,
        which is exactly when the OOD mixin has to skip its table clamp. Ranges frozen at that
        moment then disagreed with what `_get_initial_placement_bounds()` returns afterwards
        (cubeA came out +-0.40 in the sampler against +-0.35 in the bounds, i.e. the sampler was
        still placing cubes past the table edge). Re-reading per sample removes that window.
        """
        if self._per_object_fn is not None:
            return self._per_object_fn()
        return self._per_object

    def sample(self, fixtures=None, reference=None, on_top=True):
        objects = list(self.mujoco_objects)
        defaults = (self.x_range, self.y_range, self.rotation)
        placed = dict(fixtures) if fixtures else {}
        ranges = self.per_object  # resolve the callback once per sample, not once per object
        try:
            for obj in objects:
                override = ranges.get(obj.name, {})
                self.x_range = override.get("x_range", defaults[0])
                self.y_range = override.get("y_range", defaults[1])
                self.rotation = override.get("rotation", defaults[2])
                self.mujoco_objects = [obj]
                # upstream copies `fixtures` into its return value, so `placed` accumulates
                placed = super().sample(fixtures=placed, reference=reference, on_top=on_top)
        finally:
            self.mujoco_objects = objects
            self.x_range, self.y_range, self.rotation = defaults
        return placed


class PerCubeSamplerMixin:
    """Replaces the single shared cube sampler with a per-cube one, so `target_objects` is
    honoured on the stack tasks the same way it is everywhere else.

    Bypasses `Stack_D0.__init__` / `StackThree.__init__` (whose all-cubes-identical assert would
    fire on the deliberately asymmetric OOD bounds) and calls `Stack.__init__` directly, exactly as
    those two do themselves.
    """

    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"
        bounds = self._get_initial_placement_bounds()
        ref = bounds["cubeA"]
        sampler = PerObjectUniformRandomSampler(
            name="ObjectSampler",
            x_range=ref["x"],
            y_range=ref["y"],
            rotation=ref["z_rot"],
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=ref["reference"],
            z_offset=0.01,
            per_object_fn=lambda: {
                name: dict(x_range=b["x"], y_range=b["y"], rotation=b["z_rot"])
                for name, b in self._get_initial_placement_bounds().items()
            },
        )
        Stack.__init__(self, placement_initializer=sampler, **kwargs)


class Stack_D1_OOD_BOTH(PerCubeSamplerMixin, OODPlacementMixin, Stack_D1):
    # cubeA is carried onto cubeB, so cubeB is the target and keeps its ID window.
    target_objects = ("cubeB",)
    yaw_fold = 4  # C4 cubes -- must match Stack_D1_ID90 or the rungs stop being concentric


class StackThree_D1_OOD_BOTH(PerCubeSamplerMixin, OODPlacementMixin, StackThree_D1):
    # cubeA goes onto cubeB and cubeC onto cubeA: cubeB is the base, cubeA and cubeC are carried.
    target_objects = ("cubeB",)
    yaw_fold = 4


class Square_D2_OOD_BOTH(SquarePegPerResetMixin, SquarePegFixedMixin, OODPlacementMixin, Square_D2):
    # peg is pinned to the same pose the ID env uses, so it stays declared as a target: the
    # exemption check still guards against an upstream rename, and _id_spec leaves a zero-width
    # range untouched.
    target_objects = ("peg",)


class Threading_D0_OOD_BOTH(OODPlacementMixin, Threading_D0):
    target_objects = ("tripod",)  # already zero-width; declared so the intent is explicit


class ThreePieceAssembly_D0_OOD_BOTH(OODPlacementMixin, ThreePieceAssembly_D0_Yaw):
    # off the yaw-enabled base: 90 +- 90 evaluation window, twice the ID one
    target_objects = ("base",)


class HammerCleanup_D1_OOD_BOTH(OODPlacementMixin, HammerCleanup_D1):
    # kept for completeness; the study uses the Yaw45_Spawn12 rooting below instead
    target_objects = ("drawer",)


class HammerCleanup_FixedHead_Yaw45_Spawn25_OOD_BOTH(
        OODPlacementMixin, HammerCleanup_FixedHead_Yaw45_Spawn25):
    """The study's hammer rung, rooted on the fixed-head training distribution.

    Same construction as the Yaw45_Spawn12 rung below, which it replaces: that env's declared yaw
    window was really two lobes 180 degrees apart, so a band walked outward from it was walking
    away from one lobe and toward the other.
    """


class HammerCleanup_Yaw45_Spawn12_OOD_BOTH(OODPlacementMixin, HammerCleanup_Yaw45_Spawn12):
    """OOD rung for the Hammer training distribution this project actually uses.

    HammerCleanup_Yaw45_Spawn12 is already ID90-shaped -- drawer pinned to its D0 pose, hammer yaw
    +-45 deg (a 90 degree window), xy at 1.2x the narrow D0 box -- so it is used directly as the ID
    env rather than being run through IDRestrictMixin, which would be a no-op on it.

    Applying the generic rule to it REPRODUCES the hand-written HammerCleanup_OOD_Spawn20_Yaw90
    that produced the 0.57 -> 0.77 gap: x/y at 1.67x the training width lands on 2.0x the D0
    half-width ([0.060, 0.220] / [-0.235, -0.095]) and the yaw window doubles to +-90 deg. That is
    where OOD_SCALE=1.67 comes from, and mg_verify_ood_ladder.py asserts the two envs agree.

    The drawer is fixed by inheritance here, so the target exemption is trivially satisfied; it is
    still declared so a rename upstream trips _check_target_names.
    """

    target_objects = ("drawer",)


class MugCleanup_D1_OOD_BOTH(OODPlacementMixin, MugCleanup_D1):
    target_objects = ("drawer",)


class Coffee_D2_OOD_BOTH(OODPlacementMixin, Coffee_D2):
    target_objects = ("coffee_machine",)


class Kitchen_D1_OOD_BOTH(OODPlacementMixin, Kitchen_D1):
    # bread and pot are carried; the button is pressed in place, so it counts as a target.
    target_objects = ("stove", "button", "serving_region")


class CoffeePreparation_D1_OOD_BOTH(OODPlacementMixin, CoffeePreparation_D1):
    target_objects = ("coffee_machine", "drawer")


class NutAssembly_D0_OOD_BOTH(OODPlacementMixin, NutAssembly_D0):
    # Both nuts are manipulated; the two pegs they go onto never randomize and are absent from the
    # bounds dict, so there is no target entry to hold back.
    pass


class PickPlace_D0_OOD_BOTH(PickPlace_D0):
    # Different mechanism: the sampler's x/y/rotation ranges are built from bin geometry inside
    # _get_placement_initializer, so they are widened post-hoc. Every object in the bin is
    # manipulated; the bins themselves are not sampled.
    ood_scale = OOD_SCALE
    widen_xy = True    # accepted and ignored: see below
    widen_yaw = True

    def _get_placement_initializer(self):
        super()._get_placement_initializer()
        sampler = self.placement_initializer.samplers["CollisionObjectSampler"]
        # x/y are deliberately NOT widened here. This sampler's range is the interior of the bin
        # the objects start in (upstream derives it from bin geometry), so 2x would spawn objects
        # on the bin walls or outside it -- objects falling off a bin is a different failure than
        # the placement shift under study. pick_place therefore gets a rotation-only OOD rung.
        if self.widen_yaw and isinstance(sampler.rotation, (tuple, list)):
            lo, hi = sampler.rotation
            sampler.rotation = fixed_width_range(lo, hi, OOD_ROT_HALF_WIDTH)


# --- single-factor variants -----------------------------------------------------------------
#
# Each task's OOD rung is split three ways so an ID->OOD drop can be attributed to a factor rather
# than to "the scene changed". The Hammer rung that produced this project's only above-noise gap
# moved yaw and spawn together and its own docstring flags that the gap "CANNOT be attributed to
# either factor alone"; these variants remove that limitation.
#
#   _OOD_POS   x/y widened 1.67x, yaw held at the ID window
#   _OOD_YAW   yaw widened to 180 deg, x/y held at the ID ranges
#   _OOD_BOTH  both rungs at once (what the Hammer study did)
#
# Two variants are degenerate, and are still generated so every task has the same three names:
#   coffee_d2._OOD_YAW    == ID: its only primary is the coffee pod, a body of revolution whose
#                           yaw upstream fixes (see ThreePieceAssembly_D0_Yaw).
#   pick_place_d0._OOD_POS == ID: its x/y range is the interior of the bin the objects start in,
#                           so widening it would spawn them outside the bin, not further out on a
#                           table.

def _make_axis_variants(both_cls):
    """Derive the POS-only and YAW-only envs from a task's _OOD_BOTH class.

    Built with `type()` rather than written out: the per-task content -- target_objects, the peg
    pin, the per-cube sampler, pick_place's bin handling -- already lives in the _BOTH class, and
    36 hand-written subclasses differing only by two booleans would be 36 places for those to drift
    apart. robosuite's EnvMeta registers a class when it is created, so these land in
    REGISTERED_ENVS exactly like the written-out ones.
    """
    stem = both_cls.__name__[: -len("_OOD_BOTH")]
    made = []
    for suffix, flags in (("POS", dict(widen_yaw=False)), ("YAW", dict(widen_xy=False))):
        name = f"{stem}_OOD_{suffix}"
        doc = (f"{suffix}-only OOD rung for {stem}: "
               + ("x/y widened, yaw held at the ID window."
                  if suffix == "POS" else
                  "yaw widened to 180 deg, x/y held at the ID ranges."))
        made.append(type(name, (both_cls,), dict(__doc__=doc, **flags)))
    return made


_BOTH_CLASSES = [
    Stack_D1_OOD_BOTH, StackThree_D1_OOD_BOTH, Square_D2_OOD_BOTH, Threading_D0_OOD_BOTH,
    ThreePieceAssembly_D0_OOD_BOTH, HammerCleanup_FixedHead_Yaw45_Spawn25_OOD_BOTH,
    MugCleanup_D1_OOD_BOTH,
    Coffee_D2_OOD_BOTH, Kitchen_D1_OOD_BOTH, CoffeePreparation_D1_OOD_BOTH,
    NutAssembly_D0_OOD_BOTH, PickPlace_D0_OOD_BOTH,
]
for _both in _BOTH_CLASSES:
    for _cls in _make_axis_variants(_both):
        globals()[_cls.__name__] = _cls


# task -> training env + the three evaluation rungs
TASK_LADDER = {
    task: (stem if stem.startswith("HammerCleanup") else f"{stem}_ID90",
           f"{stem}_OOD_POS", f"{stem}_OOD_YAW", f"{stem}_OOD_BOTH")
    for task, stem in [
        ("stack_d1", "Stack_D1"),
        ("stack_three_d1", "StackThree_D1"),
        ("square_d2", "Square_D2"),
        ("threading_d0", "Threading_D0"),
        ("three_piece_assembly_d0", "ThreePieceAssembly_D0"),
        # NOTE: hammer's ID env is the training distribution itself, not a `_ID90` alias --
        # HammerCleanup_FixedHead_Yaw45_Spawn25 already carries its own ID window.
        ("hammer_cleanup_d1", "HammerCleanup_FixedHead_Yaw45_Spawn25"),
        ("mug_cleanup_d1", "MugCleanup_D1"),
        ("coffee_d2", "Coffee_D2"),
        ("kitchen_d1", "Kitchen_D1"),
        ("pick_place_d0", "PickPlace_D0"),
        ("coffee_preparation_d1", "CoffeePreparation_D1"),
        ("nut_assembly_d0", "NutAssembly_D0"),
    ]
}

# eval scenes are split evenly across the three rungs (20 + 20 + 20 = 60 per task)
EVAL_SCENES_PER_RUNG = 20


# --- graded severity ladder -------------------------------------------------------------------
#
# The three rungs above answer "does it break OOD". They do not answer "how far out does it break",
# because each is a single point on its axis. These variants replace each point with five, so a
# task's degradation can be read as a CURVE against distance from the training distribution rather
# than one number.
#
#   _OOD_POS_L1 .. _L5   x/y half-width x 1.2, 1.4, 1.6, 1.8, 2.0 about the ID center;
#                        yaw held at the ID window throughout.
#   _OOD_YAW_L1 .. _L5   x/y held at the ID ranges; yaw drawn from the k-th of five DISJOINT bands
#                        walking outward from the edge of the ID window to the far side of the
#                        object's orientation orbit.
#
# The two ladders differ in kind, deliberately. The position levels are NESTED (a 2.0x box contains
# the 1.2x one), which is what a scale factor means and what the existing 1.67x rung already does.
# The rotation levels are DISJOINT: a widened yaw window keeps re-drawing the ID orientations it
# contains, so severity would grow only as fast as the new fraction, and L5 would still be mostly
# ID. Bands remove that -- every episode at level k is at least `k-1` steps away from anything the
# policy trained on, and L5 sits at the far end of the orbit (180 deg for an asymmetric object,
# 45 for a C4 cube), the furthest a yaw can be from the training window.
#
# The bands are ONE-SIDED (they walk in the +yaw direction only). A two-sided band -- |dyaw| in
# [a, b] on both sides -- is not expressible as the single (lo, hi) interval robosuite's
# UniformRandomSampler takes, and the only symmetric single interval outside a window is the whole
# outside, which is nested rather than disjoint. Disjointness was the requirement, so the sign is
# what gives. Worth knowing when reading these: this project has already measured an asymmetry
# between the two yaw halves (the clean-frame probe found the negative half both rarer in the
# training data and worse at test), so an L5 number is "180 deg away on the + side", not an average
# over both directions. A symmetric version needs a custom sampler, not a different window.
POS_LADDER_SCALES = (1.2, 1.4, 1.6, 1.8, 2.0)
YAW_LADDER_LEVELS = 5


def yaw_band_range(lo, hi, band, n_bands=YAW_LADDER_LEVELS, fold=1):
    """The `band`-th of `n_bands` disjoint yaw windows walking outward from the ID window (lo, hi).

    The region available to walk into runs from the ID window's edge to the far side of the
    object's orientation orbit -- pi for an asymmetric object, pi/fold for one with N-fold symmetry
    about z, since yaw and yaw + 2pi/fold are then the same physical scene and anything past the
    half-orbit is closer to the ID window again, not further. Splitting that region evenly makes
    band `n_bands` end exactly at the far point.
    """
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    far = np.pi / fold
    step = (far - half) / n_bands
    return (center + half + (band - 1) * step, center + half + band * step)


class YawBandMixin:
    """Draws yaw from one disjoint band instead of a widened window; x/y stay at ID.

    Sits in front of a task's `_OOD_BOTH` class with both widen flags off, so the inherited
    `_get_initial_placement_bounds()` hands back exactly the ID spec for every object -- targets
    included, since OODPlacementMixin exempts those -- and this replaces the z_rot of the primaries
    only. An object whose z_rot is a single fixed point upstream (coffee_d2's pod) has no axis to
    walk away from and keeps it, exactly as the plain _OOD_YAW rung does.
    """

    yaw_band = 1
    yaw_n_bands = YAW_LADDER_LEVELS
    widen_xy = False    # the position axis is held at ID for the whole rotation ladder
    widen_yaw = False   # the band REPLACES the widened window; it does not widen on top of it

    def _get_initial_placement_bounds(self):
        base = super()._get_initial_placement_bounds()
        out = {}
        for name, spec in base.items():
            new_spec = dict(spec)
            if name not in getattr(self, "target_objects", ()) and "z_rot" in spec:
                lo, hi = spec["z_rot"]
                if hi - lo > EPS:
                    new_spec["z_rot"] = yaw_band_range(
                        lo, hi, self.yaw_band, self.yaw_n_bands, getattr(self, "yaw_fold", 1))
            out[name] = new_spec
        return out


class PickPlaceYawBandMixin:
    """The same band, applied where pick_place actually keeps its ranges.

    pick_place builds its sampler from bin geometry rather than from the bounds dict, so the
    dict-rewriting mixin above would be a no-op here. `widen_yaw = False` turns off the parent's
    180 degree widening and the band is written onto the sampler in its place.
    """

    yaw_band = 1
    yaw_n_bands = YAW_LADDER_LEVELS
    widen_xy = False
    widen_yaw = False

    def _get_placement_initializer(self):
        super()._get_placement_initializer()
        sampler = self.placement_initializer.samplers["CollisionObjectSampler"]
        if isinstance(sampler.rotation, (tuple, list)):
            lo, hi = sampler.rotation
            sampler.rotation = yaw_band_range(lo, hi, self.yaw_band, self.yaw_n_bands)


def _make_graded_variants(both_cls):
    """Derive the ten graded envs from a task's _OOD_BOTH class, as _make_axis_variants does.

    Everything task-specific -- target_objects, the pinned peg, the per-cube sampler, the yaw fold,
    pick_place's bin handling -- already lives in that class, so the levels inherit it rather than
    restating it twelve times over.
    """
    stem = both_cls.__name__[: -len("_OOD_BOTH")]
    is_pick_place = issubclass(both_cls, PickPlace_D0)
    made = []
    for level, scale in enumerate(POS_LADDER_SCALES, start=1):
        made.append(type(
            f"{stem}_OOD_POS_L{level}", (both_cls,),
            dict(__doc__=f"Position ladder level {level}/{len(POS_LADDER_SCALES)} for {stem}: "
                         f"x/y half-width x {scale} about the ID center, yaw held at ID.",
                 ood_scale=scale, widen_xy=True, widen_yaw=False)))
    band_mixin = PickPlaceYawBandMixin if is_pick_place else YawBandMixin
    for level in range(1, YAW_LADDER_LEVELS + 1):
        made.append(type(
            f"{stem}_OOD_YAW_L{level}", (band_mixin, both_cls),
            dict(__doc__=f"Rotation ladder level {level}/{YAW_LADDER_LEVELS} for {stem}: yaw from "
                         f"band {level} outward from the ID window, x/y held at ID.",
                 yaw_band=level)))
    return made


for _both in _BOTH_CLASSES:
    for _cls in _make_graded_variants(_both):
        globals()[_cls.__name__] = _cls


# task -> the ten graded evaluation envs, position ladder first
TASK_GRADED_LADDER = {
    task: (tuple(f"{stem}_OOD_POS_L{i}" for i in range(1, len(POS_LADDER_SCALES) + 1))
           + tuple(f"{stem}_OOD_YAW_L{i}" for i in range(1, YAW_LADDER_LEVELS + 1)))
    for task, (id_env, _pos, _yaw, both) in TASK_LADDER.items()
    for stem in (both[: -len("_OOD_BOTH")],)
}

# Degenerate levels, for the same reason their single-point counterparts are degenerate. Reporting
# either as OOD is what put nine wrong cells in the paper's first table, so they are named here
# rather than left to be rediscovered:
#   coffee_d2      every _OOD_YAW_L* == ID -- the pod is a body of revolution with a fixed yaw
#   pick_place_d0  every _OOD_POS_L* == ID -- its x/y range is the bin interior and is not widened
GRADED_DEGENERATE = {
    "coffee_d2": tuple(f"Coffee_D2_OOD_YAW_L{i}" for i in range(1, YAW_LADDER_LEVELS + 1)),
    "pick_place_d0": tuple(f"PickPlace_D0_OOD_POS_L{i}"
                           for i in range(1, len(POS_LADDER_SCALES) + 1)),
}
