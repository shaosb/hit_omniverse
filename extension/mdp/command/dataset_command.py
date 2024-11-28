"""
Sub-module containing command generators for reference motion for locomotion tasks.

Created by ssb in 24.7.21
"""
import time

import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.envs import ManagerBasedEnv

from typing import TYPE_CHECKING
from hit_omniverse.extension.mdp.command.dataset import get_dataLoader, get_dataset
import random
from collections.abc import Sequence

if TYPE_CHECKING:
    from hit_omniverse.extension.mdp.command.dataset_cfg import DatasetCommandCfg


class DatasetCommand(CommandTerm):
    # cfg: DatasetCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.data = get_dataset(cfg.dataset_file)
        self.command_name: list = cfg.command_name
        self.command_dimension: list = cfg.command_dimension
        self.len_command: int = len(self.command_name)
        self.len_counter: int = len(self.data.get("time"))
        self.command_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.dataset_command = {}
        for i in range(len(self.command_name)):
            self.dataset_command.update({self.command_name[i]:
                                        torch.zeros(self.num_envs, self.command_dimension[i], device=self.device)})

        random.seed(time.time())
        for env_id in range(self.num_envs):
            self.command_counter[env_id] = random.randint(0, int(2 * self.len_counter / 3))
            batch = self._prepare_data(self.command_counter[env_id])
            for key in self.command_name:
                self.dataset_command[key][env_id, :] = batch.get(key).to(self.device)

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
            try:
                batch = self._prepare_data(self.command_counter[env_id])
                # self.command_counter[env_id] += 9
            except IndexError as e:
                self.command_counter[env_id] = 0
                batch = self._prepare_data(self.command_counter[env_id])

            for key in self.command_name:
                self.dataset_command[key][env_id, :] = batch.get(key).to(self.device)

    def _prepare_data(self, index):
        item = {}
        for key in self.data.keys():
            item.update({key: torch.tensor(self.data.get(key)[index])})
        return item

    def _update_command(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)
        # set the command counter to zero
        # self.command_counter[env_ids] = 0
        for env_id in env_ids:
            self.command_counter[env_id] = random.randint(0, int(2 * self.len_counter / 3))
        # resample the command
        self._resample(env_ids)
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0
        return extras