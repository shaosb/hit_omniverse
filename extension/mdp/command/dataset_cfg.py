"""
DatasetCfg command
created by ssb 24.7.22
"""
from dataclasses import MISSING
import math

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from hit_omniverse.extension.mdp.command.dataset_command import DatasetCommand
from hit_omniverse.extension.mdp.command.reference_command import SineCommand

@configclass
class DatasetCommandCfg(CommandTermCfg):
	"""Configuration for the dataset command generator."""
	class_type: type = DatasetCommand

	# dataset_file: str = "walk.csv"
	dataset_file = "09-run_HIT.npz"
	command_name: list = ["dof_pos", "dof_vel"]
	command_dimension: int = [22, 22]

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