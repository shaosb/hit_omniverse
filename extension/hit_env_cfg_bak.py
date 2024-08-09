import os
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils import configclass

import hit_omniverse.extension.mdp as mdp
from hit_omniverse.extension.hit_humanoid import HIT_HUMANOID_CFG, HIT_DOF_NAME
from hit_omniverse import HIT_SIM_ROOT_DIR, HIT_SIM_ASSET_DIR
from hit_omniverse.utils.helper import setup_config

import yaml
import os
from dataclasses import MISSING

# config = setup_config("robot_alldof_config.yaml")
# config = setup_config("robot_hu_config.yaml")
config = setup_config("robot_87_config.yaml")

@configclass
class HITSceneCfg(InteractiveSceneCfg):
    """
    Configuration for a HIT humanoid robot scene.
    """

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=config["terrain"]["static_friction"],
                                                        dynamic_friction=config["terrain"]["dynamic_friction"],
                                                        restitution=config["terrain"]["restitution"]),
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # simple_room = AssetBaseCfg(
    #     prim_path="/World/simple_room",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(HIT_SIM_ASSET_DIR, "Office", "office.usd")
    #     )
    # )

    # people = AssetBaseCfg(
    #     prim_path="/World/peopls",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(HIT_SIM_ASSET_DIR, "original_male_adult_police_04", "male_adult_police_04.usd")
    #     )
    # )

    # HIT humanoid robot
    robot: ArticulationCfg = HIT_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # Contact_sensor
    contact_sensor = ContactSensorCfg(
        # prim_path="{ENV_REGEX_NS}/robot/.*_leg_link6",
        # prim_path="{ENV_REGEX_NS}/robot/link_.*_foot",
        prim_path="{ENV_REGEX_NS}/robot/.*_foot",
        update_period=0.0,
        history_length=15,
        debug_vis=False,
        force_threshold=1,
        )


@configclass
class ActionCFg:
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HIT_DOF_NAME,
        use_default_offset=False,
    )

    # joint_force = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=HIT_DOF_NAME,
    #     scale=100,
    # )


@configclass
class ObservationsCfg:
    @configclass
    class ObsCfg(ObsGroup):
        # clock = ObsTerm(func=mdp.generated_commands, params={"command_name": "sine"})
        command = ObsTerm(func=mdp.constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos, scale=config["normalization"]["obs_scales"]["dof_pos"])
        joint_vel = ObsTerm(func=mdp.joint_vel, scale=config["normalization"]["obs_scales"]["dof_vel"])
        base_quat = ObsTerm(func=mdp.get_euler_xyz_tensor, scale=config["normalization"]["obs_scales"]["quat"])  # 三维朝向
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=config["normalization"]["obs_scales"]["ang_vel"]) #角速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) #线速度
        # base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        action = ObsTerm(func=mdp.last_action)

    class PrivilegedCfg(ObsCfg):
        # right_contact = ObsTerm(func=mdp.get_contact_sensor_data, params= {"bodies": config["END_EFFECTOR_NAME"]["left_foot"]})
        # left_contact = ObsTerm(func=mdp.get_contact_sensor_data, params={"bodies": config["END_EFFECTOR_NAME"]["right_foot"]})
        # contact_mask = ObsTerm(func=mdp.get_gait_phase)
        dataset_pos = ObsTerm(func=mdp.dataset_dof_pos)
        dataset_vel = ObsTerm(func=mdp.dataset_dof_vel)


    policy: ObsCfg = ObsCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class RewardsCfg:
    # # Penclity
    # alive = RewTerm(func=mdp.is_alive, weight=2)
    # terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    # # Regularization
    # torques = RewTerm(func=mdp.torques, weight=-1e-5)
    # smooth = RewTerm(func=mdp.reward_action_smooth, weight=-0.002)
    # # feet_contact_force = RewTerm(func=mdp.reward_feet_contact_number, weight=-0.01)
    # # foot_slip = RewTerm(func=mdp.foot_slip, weight=-0.05)
    # # joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    # # Task Reward
    # joint_pos = RewTerm(func=mdp.reward_joint_pos, weight=1.6)
    # # feet_clearance = RewTerm(func=mdp.reward_feet_clearance, weight=1)
    # # feet_air_time = RewTerm(func=mdp.reward_feet_air_time, weight=1)
    # # feet_distance = RewTerm(func=mdp.feet_distance, weight=0.2)
    # # feet_contact_number = RewTerm(func=mdp.reward_feet_contact_number, weight=1.2)
    # track = RewTerm(func=mdp.track_velocity, weight=0.5)
    # track_lin = RewTerm(func=mdp.track_lin, weight=1.1)
    # track_ang = RewTerm(func=mdp.track_ang, weight=1.2)

    # imitate
    # Pencity
    alive = RewTerm(func=mdp.is_alive, weight=5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)
    # Regularization
    torques = RewTerm(func=mdp.torques, weight=-1e-5)
    smooth = RewTerm(func=mdp.reward_action_smooth, weight=-0.002)
    # joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    # R_t
    r_p = RewTerm(func=mdp.joint_pos_distance, weight=0.65)
    r_v = RewTerm(func=mdp.joint_vel_distance, weight=0.1)
    #TODO by ssb 8.7
    # r_e, r_c

    # Task
    # track = RewTerm(func=mdp.track_velocity, weight=0.5)
    track_lin = RewTerm(func=mdp.track_lin, weight=1.1)
    track_ang = RewTerm(func=mdp.track_ang, weight=1.2)



@configclass
class TerminationsCfg:
    # (1) Bogy height
    body_height_below = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5}
    )
    body_height_over = DoneTerm(
        func=mdp.root_height_over_maximum,
        params={"maximum_height": 1.1}
    )

    # (2) Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CommandsCfg:
    # dataset
    dataset = mdp.dataset_cfg.DatasetCommandCfg()
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

    episode_length_s = 10

    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 0.001

        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0