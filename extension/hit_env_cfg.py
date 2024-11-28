import os
import math

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, CameraCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG

import hit_omniverse.extension.mdp as mdp
import hit_omniverse.extension.mdp.rough as rough
from hit_omniverse.extension.hit_humanoid import HIT_HUMANOID_CFG, HIT_DOF_NAME
from hit_omniverse import HIT_SIM_ROOT_DIR, HIT_SIM_ASSET_DIR
from hit_omniverse.utils.helper import setup_config

from hit_omniverse.extension.g1 import G1_CFG

import yaml
import os
from dataclasses import MISSING

config = setup_config(os.environ.get("CONFIG"))

@configclass
class HITSceneCfg(InteractiveSceneCfg):
    """
    Configuration for a HIT humanoid robot scene.
    """

    # ground plane
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/defaultGroundPlane",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=config["terrain"]["static_friction"],
    #                                                     dynamic_friction=config["terrain"]["dynamic_friction"],
    #                                                     restitution=config["terrain"]["restitution"]),
    #     debug_vis=False,
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/defaultGroundPlane",
        terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        terrain_generator=rough.ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    #
    # # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    # )


    # HIT humanoid robot
    robot: ArticulationCfg = HIT_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    # robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")


    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/.*",
        history_length=3,
        track_air_time=True
    )

    # Ray
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/robot/base_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/defaultGroundPlane"],
    #     # mesh_prim_paths=["/World/ground"],
    # )


@configclass
class ActionCFg:
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    @configclass
    class ObsCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    class PrivilegedCfg(ObsCfg):
        # right_contact = ObsTerm(func=mdp.get_contact_sensor_data, params= {"bodies": config["END_EFFECTOR_NAME"]["left_foot"]})
        # left_contact = ObsTerm(func=mdp.get_contact_sensor_data, params={"bodies": config["END_EFFECTOR_NAME"]["right_foot"]})
        # contact_mask = ObsTerm(func=mdp.get_gait_phase)
        # dataset_pos = ObsTerm(func=mdp.dataset_dof_pos)
        # world_xyz = ObsTerm(func=mdp.dataset_world_xyz)
        # world_rpy = ObsTerm(func=mdp.dataset_world_rpy)
        # dataset_vel = ObsTerm(func=mdp.dataset_dof_vel)
        pass

    policy: ObsCfg = ObsCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # reset_robot_position = EventTerm(
    #     func=mdp.reset_robot_position,
    #     mode="reset",
    # )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    pass


@configclass
class RewardsCfg:
    # # Penclity
    # alive = RewTerm(func=mdp.is_alive, weight=1)
    alive = RewTerm(func=mdp.is_alive, weight=5)
    # # terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # # # Regularization
    # # torques = RewTerm(func=mdp.torques, weight=1)
    # reward_feet_contact_force = RewTerm(func=mdp.reward_feet_contact_force, weight=1)
    # smooth = RewTerm(func=mdp.reward_action_smooth, weight=500)
    # # # Imitation
    # # track_pos = RewTerm(func=mdp.joint_pos_distance, weight=50)
    # track_lower_pos = RewTerm(func=mdp.joint_lower_pos_distance, weight=2)
    # track_upper_pos = RewTerm(func=mdp.joint_upper_pos_distance, weight=1)
    # track_lin = RewTerm(func=mdp.track_lin, weight=5)
    # track_ang = RewTerm(func=mdp.track_ang, weight=5)
    # track_lin_x = RewTerm(func=mdp.track_lin_x, weight=5)
    # track_yaw_row = RewTerm(func=mdp.track_yaw_row, weight=5)

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            # "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
        # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    # (1) Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Illegal contact
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
        # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    pass


@configclass
class CommandsCfg:
    # dataset
    # dataset = mdp.dataset_cfg.DatasetCommandCfg()
    # dataset = mdp.dataset_cfg.HITDatasetCommandCfg()
    # # slope_lone = mdp.dataset_cfg.SlopeCommandCfg()
    # # squat_walk = mdp.dataset_cfg.SquatCommandCfg()
    # # stair_full = mdp.dataset_cfg.StairCommandCfg()
    # # hit_save_people = mdp.dataset_cfg.PeopleCommandCfg()
    # # forsquat_down = mdp.dataset_cfg.DownCommandCfg()
    # # forsquat_up = mdp.dataset_cfg.UpCommandCfg()
    # # squat_with_people = mdp.dataset_cfg.SquatWithPeopleCommandCfg()
    # imitation = mdp.dataset_cfg.ImitationCommandCfg()
    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )
    # # 11-26_19-11-42
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.6, 1.0), lin_vel_y=(0, 0), ang_vel_z=(0, 0)
        ),
    )
    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )
    pass
    # # velocity
    # velocity = mdp.velocity_command()
    # Sine
    # sine = mdp.dataset_cfg.SineCommandCfg()


@configclass
class HITRLEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(device=config["SIM"]["device"])
    scene: HITSceneCfg = HITSceneCfg(num_envs=config["SIM"]["num_envs"],
                                     env_spacing=config["SIM"]["env_spacing"]
                                     )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionCFg = ActionCFg()
    events: EventCfg = EventCfg()
    # # curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    episode_length_s = 20

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005

        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0