import os

import torch
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, CameraCfg
from omni.isaac.lab.utils import configclass

import hit_omniverse.extension.mdp as mdp
from hit_omniverse.extension.hit_humanoid import HIT_HUMANOID_CFG, HIT_DOF_NAME
from hit_omniverse import HIT_SIM_ROOT_DIR, HIT_SIM_ASSET_DIR, HIT_SIM_ROBOT_DIR
from hit_omniverse.utils.helper import setup_config

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
    terrain = TerrainImporterCfg(
        prim_path="/World/defaultGroundPlane",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=config["terrain"]["static_friction"],
                                                        dynamic_friction=config["terrain"]["dynamic_friction"],
                                                        restitution=config["terrain"]["restitution"]),
        debug_vis=False,
    )
    #
    # # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # cangku = AssetBaseCfg(
    #     prim_path="/World/cangku",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="G:\\Tianzp\\ShenHaoBei_scene\\ShenHaoBei_scene\\yuan_cangku3_reset.usd"
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=tuple((55., 10., -5)),
    #         rot=tuple((0.70711,0.,0.,0.70711)),
    #         )
    # )# (55., 10., -0.015)

    # HIT humanoid robot
    # robot: ArticulationCfg = HIT_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    # l_hip_roll = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/l_hip_roll",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "left_arm_link4.usd")
    #     ),
    # )
    pelvis = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/pelvis",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "pelvis.usd")
        ),
    )

    r_hip_roll = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/r_hip_roll",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "r_hip_roll.usd")
        ),
    )
    r_hip_yaw = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/r_hip_yaw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "r_hip_yaw.usd")
        ),
    )
    r_upper_leg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/r_upper_leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "r_upper_leg.usd")
        ),
    )
    r_lower_leg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/r_lower_leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "r_lower_leg.usd")
        ),
    )
    r_ankle = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/r_ankle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "r_ankle.usd")
        ),
    )
    r_foot = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/r_foot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "r_foot.usd")
        ),
    )
    l_hip_roll = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/l_hip_roll",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "l_hip_roll.usd")
        ),
    )
    l_hip_yaw = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/l_hip_yaw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "l_hip_yaw.usd")
        ),
    )
    l_upper_leg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/l_upper_leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "l_upper_leg.usd")
        ),
    )
    l_lower_leg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/l_lower_leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "l_lower_leg.usd")
        ),
    )
    l_ankle = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/l_ankle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "l_ankle.usd")
        ),
    )
    l_foot = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/l_foot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "l_foot.usd")
        ),
    )
    waist_link1 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/waist_link1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "waist_link1.usd")
        ),
    )
    body_link = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/body_link",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "body_link.usd"),
        ),
    )
    right_arm_link1 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/right_arm_link1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "right_arm_link1.usd")
        ),
    )
    right_arm_link2 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/right_arm_link2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "right_arm_link2.usd")
        ),
    )
    right_arm_link3 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/right_arm_link3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "right_arm_link3.usd")
        ),
    )
    right_arm_link4 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/right_arm_link4",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "right_arm_link4.usd")
        ),
    )
    left_arm_link1 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/left_arm_link1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "left_arm_link1.usd")
        ),
    )
    left_arm_link2 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/left_arm_link2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "left_arm_link2.usd")
        ),
    )
    left_arm_link3 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/left_arm_link3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "left_arm_link3.usd")
        ),
    )
    left_arm_link4 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/left_arm_link4",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(HIT_SIM_ROBOT_DIR, "robot_simplify", "rigid", "left_arm_link4.usd")
        ),
    )

@configclass
class ActionCFg:
    pass

@configclass
class ObservationsCfg:
    @configclass
    class ObsCfg(ObsGroup):
        command = ObsTerm(func=mdp.constant_commands)
        pass

    class PrivilegedCfg(ObsCfg):
        pass


    policy: ObsCfg = ObsCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    # reset_scene = EventTerm(
    #     func=mdp.reset_scene_to_default,
    #     mode="reset",
    # )
    pass

@configclass
class CurriculumCfg:
    pass


@configclass
class RewardsCfg:
    pass



@configclass
class TerminationsCfg:
    pass


@configclass
class CommandsCfg:
    # dataset
    # dataset = mdp.dataset_cfg.DatasetCommandCfg()
    # slope_lone = mdp.dataset_cfg.SlopeCommandCfg()
    # squat_walk = mdp.dataset_cfg.SquatCommandCfg()
    # stair_full = mdp.dataset_cfg.StairCommandCfg()
    # hit_save_people = mdp.dataset_cfg.PeopleCommandCfg()
    # forsquat_down = mdp.dataset_cfg.DownCommandCfg()
    # forsquat_up = mdp.dataset_cfg.UpCommandCfg()
    # squat_with_people = mdp.dataset_cfg.SquatWithPeopleCommandCfg()
    imitation = mdp.dataset_cfg.ImitationCommandCfg()
    pass
    # # velocity
    # velocity = mdp.velocity_command()
    # Sine
    # sine = mdp.dataset_cfg.SineCommandCfg()


@configclass
class HITRighdEnvCfg(ManagerBasedRLEnvCfg):
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
        self.sim.dt = 0.005
        # self.sim.dt = 0.01

        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0