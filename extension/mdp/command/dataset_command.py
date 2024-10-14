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

if TYPE_CHECKING:
    from hit_omniverse.extension.mdp.command.dataset_cfg import DatasetCommandCfg



class DatasetCommand(CommandTerm):
    # cfg: DatasetCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.dataloader = get_dataLoader(cfg.dataset_file)
        self.command_name: list = cfg.command_name
        self.command_dimension: list = cfg.command_dimension
        self.len_command: int = len(self.command_name)

        self.data_iter_list = []
        for _ in range(self.num_envs):
            self.data_iter_list.append(iter(self.dataloader))

        self.dataset_command = {}
        for i in range(len(self.command_name)):
            self.dataset_command.update({self.command_name[i]:
                                        torch.zeros(self.num_envs, self.command_dimension[i], device=self.device)})



    def __str__(self) -> str:
        msg = "DatasetCommand:\n"
        msg += f"\tCommand tyoe: {tuple(self.command_name)}\n"
        msg += f"\tCommand dimension: {tuple(self.command_dimension)}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired datates command in the base frame. Shape is (num_envs, command_dimension)."""
        return self.dataset_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        for env_id in env_ids:
            if self.command_counter[env_id] == 1:
                data_iter = iter(self.dataloader)
                self.data_iter_list[env_id] = data_iter

            data_iter = self.data_iter_list[env_id]
            try:
                for i in range(10):
                    batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                self.data_iter_list[env_id] = data_iter
                batch = next(data_iter)

            for key in self.command_name:
                self.dataset_command[key][env_id, :] = batch[0].get(key).to(self.device)


    def _update_command(self):
        pass