"""
Configuration for a HIT humanoid robot.

created by ssb in 2024.6.14
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

from hit_omniverse import HIT_SIM_ROBOT_DIR
from hit_omniverse.utils.helper import setup_config, get_args

import argparse

parser = get_args()
config = setup_config(parser.config_file)

HIT_DOF_NAME = list(config["INIT_JOINT_POS"].keys())

USD_PATH = config["USD_PATH"]

HIT_HUMANOID_CFG = ArticulationCfg(
    # prim_path="{ENV_REGEX_NX}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{HIT_SIM_ROBOT_DIR}\\{USD_PATH}",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=config["DISABLE_GRAVITY"],
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=config["FIXED_BASE"]
        ),
        # copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=tuple(config["ROBOT_POS"]),
        rot=tuple(config["ROBOT_ROT"]),
        joint_pos=config["INIT_JOINT_POS"],
        joint_vel={".*": 0.0},
    ),
    actuators={key: ImplicitActuatorCfg(
            joint_names_expr=key,
            effort_limit=float(value["effort_limit"]),
            velocity_limit=float(value["velocity_limit"]),
            stiffness=float(value["stiffness"]),
            damping=float(value["damping"]),
        ) for key, value in config["DOF_ASSET"].items()},
)