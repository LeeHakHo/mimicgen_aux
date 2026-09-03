# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Contains environments for BUDS hammer place task from robosuite task zoo repo.
(https://github.com/ARISE-Initiative/robosuite-task-zoo)
"""

import os
import random
import numpy as np
from six import with_metaclass
from copy import deepcopy

import robosuite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import HammerObject, MujocoXMLObject
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, string_to_array, find_elements, add_material
from robosuite.utils.buffers import RingBuffer

import robosuite_task_zoo
from robosuite_task_zoo.environments.manipulation.hammer_place import HammerPlaceEnv

import mimicgen
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG
from mimicgen.models.robosuite.objects import DrawerObject


class HammerCleanup_D0(HammerPlaceEnv, SingleArmEnv_MG):
    """
    Augment BUDS hammer place task for mimicgen.
    """
    def __init__(self, robot_init_qpos=None, **kwargs):
        self.robot_init_qpos = robot_init_qpos
        HammerPlaceEnv.__init__(self, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def _load_model(self):
        """
        Copied exactly from HammerPlaceEnv, but swaps out the cabinet object.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414, 0.49641484022140503, 0.6968992352485657]
        )
        
        
        bread = CustomMaterial(
            texture="Bread",
            tex_name="bread",
            mat_name="MatBread",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        tex_attrib = {
            "type": "cube"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }
        
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        ingredient_size = [0.03, 0.018, 0.025]
        
        self.sorting_object = HammerObject(name="hammer",
                                           handle_length=(0.045, 0.05),
                                           handle_radius=(0.012, 0.012),
                                           head_density_ratio=1.0
        )

        self.cabinet_object = DrawerObject(
            name="CabinetObject")
        cabinet_object = self.cabinet_object.get_obj(); cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03))); mujoco_arena.table_body.append(cabinet_object)
        
        for obj_body in [
                self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                 naming_prefix=obj_body.naming_prefix,
                                                                 custom_material=deepcopy(material))
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        ingredient_size = [0.015, 0.025, 0.02]
        
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
        sampler = UniformRandomSampler(
            name="ObjectSampler-pot",
            mujoco_objects=self.sorting_object,
            x_range=[0.10,  0.18],
            y_range=[-0.20, -0.13],
            rotation=(-0.1, 0.1),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))
        
        mujoco_objects = [
            self.sorting_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)


class HammerCleanup_D1(HammerCleanup_D0):
    """
    Move object and drawer with wide initialization. Note we had to make some objects movable that were fixtures before.
    """
    def _check_success(self):
        """
        Update from superclass to have a more stringent check that's not buggy
        (e.g. there's no check in x-position before) and that supports
        different drawer (cabinet) positions.
        """
        object_pos = self.sim.data.body_xpos[self.sorting_object_id]
        # object_in_drawer = 1.0 > object_pos[2] > 0.94 and object_pos[1] > 0.22

        cabinet_closed = self.sim.data.qpos[self.cabinet_qpos_addrs] > -0.01

        # new contact-based drawer check - object in contact with bottom drawer geom
        drawer_bottom_geom = "CabinetObject_drawer_bottom"
        object_in_drawer = self.check_contact(drawer_bottom_geom, self.sorting_object)

        return object_in_drawer and cabinet_closed

    def _get_sorting_object(self):
        """
        Method that constructs object to place into drawer. Subclasses can override this method to
        construct different objects.
        """
        return HammerObject(
            name="hammer",
            handle_length=(0.045, 0.05),
            handle_radius=(0.012, 0.012),
            head_density_ratio=1.0,
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            hammer=dict(
                x=(-0.2, 0.2),
                y=(-0.25, -0.13),
                z_rot=(0., 2. * np.pi),
                reference=self.table_offset,
                init_quat=self.sorting_object.init_quat,
                # NOTE: this rotation axis needs to be y, not z because of hammer's init_quat
                rotation_axis="y",
            ),
            drawer=dict(
                x=(0.0, 0.2),
                y=(0.2, 0.3),
                # z_rot=(0., 0.),
                z_rot=(-np.pi / 6., np.pi / 6.),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-hammer",
                mujoco_objects=self.sorting_object,
                x_range=bounds["hammer"]["x"],
                y_range=bounds["hammer"]["y"],
                rotation=bounds["hammer"]["z_rot"],
                rotation_axis=bounds["hammer"]["rotation_axis"],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["hammer"]["reference"],
                z_offset=0.02,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-drawer",
                mujoco_objects=self.cabinet_object,
                x_range=bounds["drawer"]["x"],
                y_range=bounds["drawer"]["y"],
                rotation=bounds["drawer"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["drawer"]["reference"],
                z_offset=0.03,
            )
        )

    def _load_model(self):
        """
        Update to include drawer (cabinet) in placement initializer.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Adjust initial robot joint configuration accordingly
        if self.robot_init_qpos is not None:
            self.robots[0].init_qpos = self.robot_init_qpos

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414, 0.49641484022140503, 0.6968992352485657]
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        tex_attrib = {
            "type": "cube"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }
        
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        self.sorting_object = self._get_sorting_object()

        self.cabinet_object = DrawerObject(name="CabinetObject")

        # # old: manually set position in xml and add to mujoco arena
        # cabinet_object = self.cabinet_object.get_obj()
        # cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        # mujoco_arena.table_body.append(cabinet_object)
        
        for obj_body in [
                self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                 naming_prefix=obj_body.naming_prefix,
                                                                 custom_material=deepcopy(material))
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)
        
        self._get_placement_initializer()
        
        mujoco_objects = [
            self.sorting_object,
            self.cabinet_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)

    def _reset_internal(self):
        """
        Update to make sure placement initializer can be used to set drawer (cabinet) pose
        even though it doesn't have a joint.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.cabinet_object:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = 0.905 # hardcode z-value to correspond to parent class
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []


class HammerCleanup_D1_FixedDrawer(HammerCleanup_D1):
    """
    Deconfounded OOD variant of D1. Keeps the D1 hammer randomization (full 360-deg
    z-yaw + wide xy spawn) but PINS the drawer to the fixed D0 training pose, so the
    only factor pushed out of distribution is the hammer -- the drawer no longer moves
    or rotates. Measured D0 drawer world pose = table_offset + (0.2, 0.30), z_rot=0
    (table_offset = [-0.2, 0, 0.9]); the D1 sampler references table_offset, so a
    degenerate range at that offset reproduces the D0 drawer exactly. Everything else
    (movable-drawer mechanism, z-hardcode in _reset_internal, improved _check_success)
    is inherited from D1.
    """
    def _get_initial_placement_bounds(self):
        bounds = super()._get_initial_placement_bounds()
        bounds["drawer"]["x"] = (0.2, 0.2)
        bounds["drawer"]["y"] = (0.30, 0.30)
        bounds["drawer"]["z_rot"] = (0.0, 0.0)
        return bounds


class FixedHeadHammerObject(HammerObject):
    """HammerObject whose head always faces the same way.

    Upstream's `init_quat` is a PROPERTY that flips a coin on every access:

        return np.array([0.5,-0.5,0.5,-0.5]) if np.random.rand() >= 0.5 else np.array([-0.5]*4)

    so the head direction is drawn independently of the yaw window, and drawn AGAIN every time the
    attribute is read -- `_get_initial_placement_bounds()` stores one draw while `sample()` uses
    another. Measured in the 200 training demos of HammerCleanup_Yaw45_Spawn12: the hammer is
    perfectly flat at t=0 (body y-axis . world z = 1.000, min = max), and its in-plane heading is
    BIMODAL, 121 episodes in one lobe and 79 in the lobe 180 degrees away, which folds to a single
    clean 89.3 degree window. In other words the declared +-45 degree window is really two 90
    degree lobes on opposite sides of the circle, and which lobe an episode lands in is noise the
    policy cannot predict and the aux rotation target cannot be a function of the image across.

    Pinning it makes the orientation orbit a real 360 degrees again, so a +-45 degree window means
    +-45 degrees. NOTE that this HALVES the orientation variety the policy sees (two 90 degree
    lobes -> one), so a yaw window widened to compensate is what keeps the task as hard as it was.
    """

    @property
    def init_quat(self):
        return np.array([0.5, -0.5, 0.5, -0.5])


class FixedHeadMixin:
    """Gives a HammerCleanup env the deterministic-head hammer."""

    def _get_sorting_object(self):
        return FixedHeadHammerObject(
            name="hammer",
            handle_length=(0.045, 0.05),
            handle_radius=(0.012, 0.012),
            head_density_ratio=1.0,
        )


class HammerCleanup_YawBand(FixedHeadMixin, HammerCleanup_D1_FixedDrawer):
    """
    Base for the yaw-only difficulty ladder. Inherits D1's hammer rotation MECHANISM
    (rotation_axis='y' about the hammer init_quat, which produces true table-plane yaw --
    see D1 _get_initial_placement_bounds) and the fixed D0 drawer, but PINS the xy spawn
    back to the narrow D0 training box so the ONLY factor that varies is the hammer yaw.
    Concrete subclasses set YAW_HALF (half-range in radians); the band is symmetric
    (-YAW_HALF, +YAW_HALF) and nests inside D1's full (0, 2*pi). D0 narrow spawn box:
    x=[0.10,0.18], y=[-0.20,-0.13], reference=table_offset. Used both for the ~45-deg
    TRAINING distribution (regenerate demos via mimicgen datagen) and the wider OOD
    eval rungs, all sharing spawn+drawer so the ladder isolates yaw.
    """
    YAW_HALF = np.pi / 4.0  # default 45 deg; override per rung

    def _get_initial_placement_bounds(self):
        bounds = super()._get_initial_placement_bounds()
        bounds["hammer"]["x"] = (0.10, 0.18)
        bounds["hammer"]["y"] = (-0.20, -0.13)
        bounds["hammer"]["z_rot"] = (-self.YAW_HALF, self.YAW_HALF)
        return bounds


class HammerCleanup_Yaw45(HammerCleanup_YawBand):
    """~45-deg yaw band, narrow D0 spawn, fixed drawer -- the new TRAINING distribution."""
    YAW_HALF = np.pi / 4.0


class HammerCleanup_Yaw45_Spawn12(HammerCleanup_YawBand):
    """
    TRAINING distribution variant: yaw band +-45 deg + fixed D0 drawer (like Yaw45), but
    the hammer xy spawn is widened to 1.2x the D0 training half-width about the same center
    (D0 x=[0.10,0.18] c0.14 h0.04, y=[-0.20,-0.13] c-0.165 h0.035 -> x=[0.092,0.188],
    y=[-0.207,-0.123]). Slightly more spawn variation than Yaw45 so in-dist is less trivially
    saturated. Regenerate demos via mimicgen datagen (config hammer_yaw45_spawn12.json).
    """
    YAW_HALF = np.pi / 4.0

    def _get_initial_placement_bounds(self):
        bounds = super()._get_initial_placement_bounds()  # narrow D0 spawn + +-45 deg yaw band
        bounds["hammer"]["x"] = (0.092, 0.188)      # 1.2x D0 half-width about center 0.14
        bounds["hammer"]["y"] = (-0.207, -0.123)    # 1.2x D0 half-width about center -0.165
        return bounds


class HammerCleanup_FixedHead_Yaw45_Spawn25(HammerCleanup_YawBand):
    """TRAINING distribution for the 12-task ID90 study, replacing Yaw45_Spawn12.

    Three measurements drove it, all taken on Yaw45_Spawn12's own 200 demos and evaluations:

    1. The head flip made the yaw window a lie. `init_quat` is a coin flip, so the declared +-45
       degree band was really two 90 degree lobes 180 degrees apart (121 / 79 episodes), and which
       lobe an episode landed in was unpredictable noise. FixedHeadMixin pins it, which turns the
       orientation orbit back into a real 360 degrees.

       The band STAYS at +-45. Widening it to +-90 to preserve the old two-lobe coverage was the
       first instinct and it is wrong twice over. The ladder gives every task an ID window that is
       a QUARTER of its orientation orbit and an OOD window that is a half, so a 180 degree ID
       window makes hammer the one task trained on OOD-width rotation -- which is part of why it
       was the easiest of the twelve. And mechanically it breaks the rung: OODPlacementMixin builds
       its windows from ID_ROT_HALF_WIDTH, so a 180 degree ID window comes back NARROWED to 90 on
       the position rung and unchanged at 180 on the rotation rung, i.e. _OOD_YAW would equal ID.
       That was measured, not predicted -- mg_verify_sampler_ranges.py printed
       `OOD_POS yaw 89.0/90` and `OOD_YAW yaw 179.1/180` against an ID window of 178.4/180.

    2. This was the easiest of the twelve tasks by a wide margin: baseline in-distribution 0.856
       against square_d2's 0.533 and stack_d1's 0.733, and 0.72 mean over the training rollouts
       where the next task was 0.62.

    3. The reason is that it had almost no position axis. Measured spawn extent at t=0 was
       0.095 x 0.084 m = 0.0080 m2, against square_d2's 0.443 x 0.498 = 0.221 -- twenty-seven times
       smaller -- while position is the axis this project has repeatedly found policies actually
       fail on. The spawn is widened to 2.5x the D0 half-width (0.20 x 0.175 m = 0.035 m2, about
       2.3x threading_d0's), which is a real position axis while staying well inside the 0.8 m
       table and far from the pinned drawer at (0.2, 0.30).

    Deliberately NOT changed: the drawer stays pinned at its D0 pose. Un-pinning it is what makes
    D1 hard, but it also moves the place target, and separating "carry to a moving target" from
    "grasp a rotated object" is the entire point of the single-factor ladder.
    """

    YAW_HALF = np.pi / 4.0                      # +-45 deg: a quarter orbit, as every task gets

    def _get_initial_placement_bounds(self):
        bounds = super()._get_initial_placement_bounds()  # narrow D0 spawn + YAW_HALF band
        bounds["hammer"]["x"] = (0.04, 0.24)        # 2.5x D0 half-width about center 0.14
        bounds["hammer"]["y"] = (-0.2525, -0.0775)  # 2.5x D0 half-width about center -0.165
        return bounds


class HammerCleanup_OOD_Spawn15_Yaw45(HammerCleanup_YawBand):
    """
    Moderate OOD *eval* variant (not for training). Inherits the YawBand mechanism
    (fixed D0 drawer + true table-plane yaw via rotation_axis='y') with the default
    +-45 deg band, but WIDENS the hammer xy spawn to 1.5x the D0 training half-width
    about the same center. D0 box x=[0.10,0.18] (c=0.14, half=0.04), y=[-0.20,-0.13]
    (c=-0.165, half=0.035) -> 1.5x half -> x=[0.08,0.20], y=[-0.2175,-0.1125].
    Used to eval the D0-trained models on a mild position+yaw OOD, avoiding the D1
    floor (360 deg + 5x spawn + moving drawer).
    """
    YAW_HALF = np.pi / 4.0

    def _get_initial_placement_bounds(self):
        bounds = super()._get_initial_placement_bounds()  # narrow D0 spawn + +-45 deg yaw band
        bounds["hammer"]["x"] = (0.08, 0.20)        # 1.5x D0 half-width about center 0.14
        bounds["hammer"]["y"] = (-0.2175, -0.1125)  # 1.5x D0 half-width about center -0.165
        return bounds


class HammerCleanup_OOD_Spawn20_Yaw90(HammerCleanup_YawBand):
    """
    HARD OOD *eval* variant (not for training), for models trained on Yaw45_Spawn12.
    Widens BOTH axes past that training distribution:
      yaw  +-45 deg -> +-90 deg   (half the band is novel rotation)
      xy   1.2x     -> 2.0x D0 half-width about the same center
           x=[0.06,0.22] (c=0.14, half 0.04*2.0), y=[-0.235,-0.095] (c=-0.165, half 0.035*2.0)
    Both still nest inside D1's full (0, 2*pi) + 5x spawn, so this sits between the
    Spawn15_Yaw45 rung (which for a Yaw45_Spawn12-trained model is ~in-distribution:
    identical yaw, and 64% of its spawn area lies inside the training box) and the
    D1 floor. NOTE: yaw and spawn move together here by explicit request, so a
    baseline-vs-aux gap measured on this env CANNOT be attributed to either factor
    alone -- use the yaw-only YawBand rungs if factor attribution is needed.
    """
    YAW_HALF = np.pi / 2.0  # 90 deg

    def _get_initial_placement_bounds(self):
        bounds = super()._get_initial_placement_bounds()  # narrow D0 spawn + YAW_HALF band
        bounds["hammer"]["x"] = (0.06, 0.22)      # 2.0x D0 half-width about center 0.14
        bounds["hammer"]["y"] = (-0.235, -0.095)  # 2.0x D0 half-width about center -0.165
        return bounds
