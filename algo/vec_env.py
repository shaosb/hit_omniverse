import torch
from typing import Tuple, Union, Dict
from abc import abstractmethod
import copy


def add_env_variable(env):
    env.num_envs = env.num_envs
    env.num_actions = env.action_space.shape[1]
    env.observation_buf = copy.deepcopy(env.obs_buf)
    env.obs_buf = env.observation_buf["policy"]
    env.num_obs = env.observation_space["policy"].shape[1]

    try:
        env.num_privileged_obs = env.observation_space["privileged"].shape[1]
        env.privileged_obs_buf = env.observation_buf["privileged"]
    except:
        env.num_privileged_obs = None
        env.privileged_obs_buf = env.obs_buf

    env.max_episode_length = env.max_episode_length
    env.rew_buf = env.reward_buf
    env.reset_buf = env.reset_buf
    env.episode_length_buf = env.episode_length_buf
    env.extras = env.extras
    env.device = env.device


def add_env_method(env):
    def get_observations():
        return env.obs_buf

    def get_privileged_observations():
        return env.privileged_obs_buf

    env.get_observations = get_observations
    env.get_privileged_observations = get_privileged_observations

# minimal interface of the environment
# class VecEnv(env):
#     num_envs: env.num_envs
#     # num_obs: int
#     # num_privileged_obs: int
#     # num_actions: int
#     # max_episode_length: int
#     # privileged_obs_buf: torch.Tensor
#     # obs_buf: torch.Tensor
#     # rew_buf: torch.Tensor
#     # reset_buf: torch.Tensor
#     # episode_length_buf: torch.Tensor # current episode duration
#     # extras: dict
#     # device: torch.device
#     @abstractmethod
#     def step(self, actions: torch.Tensor) -> Tuple[Union[torch.Tensor, Dict], Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
#         pass
#     @abstractmethod
#     def reset(self, env_ids: Union[Dict, torch.Tensor]):
#         pass
#     @abstractmethod
#     def get_observations(self) -> torch.Tensor:
#         return env.obs_buf["policy"]
#
#     @abstractmethod
#     def get_privileged_observations(self) -> Union[torch.Tensor, None]:
#         try:
#             return env.obs_buf["privileged"]
#         except:
#             return None