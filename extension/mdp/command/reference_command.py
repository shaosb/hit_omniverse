"""
Sub-module containing command generators for reference motion for locomotion tasks.

Created by ssb in 24.7.21
"""

import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.envs import ManagerBasedEnv

from typing import TYPE_CHECKING
from hit_omniverse.extension.mdp.command.dataset import get_dataLoader
from hit_omniverse.utils.helper import ALL_DOF_NAME

if TYPE_CHECKING:
    from hit_omniverse.extension.mdp.command.dataset_cfg import DatasetCommandCfg, SineCommandCfg


class SineCommand(CommandTerm):
    # cfg: DatasetCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.sine_command = torch.zeros(self.num_envs, cfg.command_dimension, device=self.device)
        self.cycle_time = cfg.cycle_time
        self.dt = cfg.dt
        self.target_joint_pos_scale = cfg.target_joint_pos_scale

    def __str__(self) -> str:
        msg = "SineCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired datates command in the base frame. Shape is (num_envs, command_dimension)."""
        return self.sine_command

    def _update_metrics(self):
        pass


    def _resample_command(self, env_ids):
        phase = self._get_phase(env_ids)
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        scale_1 = self.target_joint_pos_scale
        scale_2 = scale_1 * 2

        sin_pos_l[sin_pos_l > 0] = 0
        self.sine_command[env_ids, ALL_DOF_NAME.index("left_leg_hip_pitch")] = sin_pos_l * scale_1
        self.sine_command[env_ids, ALL_DOF_NAME.index("left_leg_knee_pitch")] = sin_pos_l * scale_2
        self.sine_command[env_ids, ALL_DOF_NAME.index("left_leg_ankle_pitch")] = sin_pos_l * scale_1

        sin_pos_r[sin_pos_r < 0] = 0
        self.sine_command[env_ids, ALL_DOF_NAME.index("right_leg_hip_pitch")] = sin_pos_r * scale_1
        self.sine_command[env_ids, ALL_DOF_NAME.index("right_leg_hip_pitch")] = sin_pos_r * scale_2
        self.sine_command[env_ids, ALL_DOF_NAME.index("right_leg_hip_pitch")] = sin_pos_r * scale_1

        # self.sine_command[torch.abs(sin_pos) < 0.1] = 0


    def _update_command(self):
        pass


    def _get_phase(self, env_ids=None):
        if env_ids is None:
            phase = self.command_counter * self.dt / self.cycle_time
        else:
            phase = self.command_counter[env_ids] * self.dt / self.cycle_time
        return phase