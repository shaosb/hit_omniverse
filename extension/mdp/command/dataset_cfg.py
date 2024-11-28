"""
DatasetCfg command
created by ssb 24.7.22
"""
import math

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from hit_omniverse.extension.mdp.command.dataset_command import DatasetCommand
from hit_omniverse.extension.mdp.command.reference_command import SineCommand

@configclass
class DatasetCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "30-run_HIT.hit"
	command_name: list = ["dof_pos", "robot_world_xyz", "robot_world_rpy"]
	command_dimension: int = [22, 3, 3]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class HITDatasetCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "walk.csv"
	command_name: list = ["dof_pos"]
	command_dimension: int = [22]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class SlopeCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "slope_lone.hit"
	command_name: list = ["dof_pos", "robot_world_xyz", "robot_world_rpy"]
	command_dimension: int = [22, 3, 3]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class SquatCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "squat_walk.hit"
	command_name: list = ["dof_pos", "robot_world_xyz", "robot_world_rpy"]
	command_dimension: int = [22, 3, 3]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class StairCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "stair_full.hit"
	command_name: list = ["dof_pos", "robot_world_xyz", "robot_world_rpy"]
	command_dimension: int = [22, 3, 3]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class PeopleCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "hit_save_people.hit"
	command_name: list = ["dof_pos", "robot_world_xyz", "robot_world_rpy"]
	command_dimension: int = [22, 3, 3]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class SquatWithPeopleCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "squat_walk_with_people.hit"
	command_name: list = ["dof_pos", "robot_world_xyz", "robot_world_rpy"]
	command_dimension: int = [22, 3, 3]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)
@configclass
class DownCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "forsquat_down.hit"
	command_name: list = ["dof_pos", "robot_world_xyz", "robot_world_rpy"]
	command_dimension: int = [22, 3, 3]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class UpCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "forsquat_up.hit"
	command_name: list = ["dof_pos", "robot_world_xyz", "robot_world_rpy"]
	command_dimension: int = [22, 3, 3]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class ImitationCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	dataset_file = "motion_retarget\\02_04.npy"
	command_name: list = ['pelvis', 'r_hip_roll', 'r_hip_yaw',
					   'r_upper_leg', 'r_lower_leg', 'r_ankle',
					   'r_foot', 'l_hip_roll', 'l_hip_yaw',
					   'l_upper_leg', 'l_lower_leg', 'l_ankle',
					   'l_foot', 'waist_link1', 'body_link',
					   'right_arm_link1', 'right_arm_link2', 'right_arm_link3',
					   'right_arm_link4', 'left_arm_link1', 'left_arm_link2', 'left_arm_link3',
					   'left_arm_link4']
	command_dimension: int = [7] * len(command_name)
	# command_name: list = ["time", "dof_pos", "dof_vel"]
	# command_dimension: int = [1, 22, 22]

	resampling_time_range: tuple[float, float] = (0.0, 0.001)

@configclass
class SineCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = SineCommand

	# dataset_file: str = "walk.csv"
	command_dimension: int = 22
	cycle_time: float = 0.64
	dt: float = 0.005
	target_joint_pos_scale: float = 0.17

	resampling_time_range: tuple[float, float] = (0.0, 0.005)