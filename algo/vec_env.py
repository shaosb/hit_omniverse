import torch
from typing import Tuple, Union, Dict
from collections import deque
from abc import abstractmethod
import copy
import os
import types

from hit_omniverse.utils.helper import setup_config

training_config = setup_config(os.environ.get("TRAINING_CONFIG"))["runner"]

def add_env_variable(env):
    env.num_envs = env.num_envs
    env.num_actions = int(env.action_space.shape[1] / 2)
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

    # make obs buffer
    env.env.env.obs_queue = deque(maxlen=training_config["max_actor_history"])
    env.env.env.critic_queue = deque(maxlen=training_config["max_critic_history"])

    for _ in range(training_config["max_actor_history"]):
        env.env.env.obs_queue.append(torch.zeros(env.num_envs, env.num_obs,
                                     dtype=torch.float, device=env.device))
    for _ in range(training_config["max_critic_history"]):
        env.env.env.critic_queue.append(torch.zeros(env.num_envs, env.num_privileged_obs,
                                        dtype=torch.float, device=env.device))

    env.env.env.obs_buf_all = torch.stack([env.env.obs_queue[i]
                               for i in range(training_config["max_actor_history"])], dim=1)
    env.env.env.critic_buf_all = torch.stack([env.env.critic_queue[i]
                                  for i in range(training_config["max_critic_history"])], dim=1)

    env.env.env.obs_input = env.env.env.obs_buf_all.reshape(env.env.env.num_envs, -1)
    env.env.env.critic_input = env.env.env.critic_buf_all.reshape(env.env.env.num_envs, -1)


def add_env_method(env):
    def get_observations():
        return env.obs_buf

    def get_privileged_observations():
        return env.privileged_obs_buf

    def wrapper_env_step():
        original_step = getattr(env, "step")

        def wrapper_step(self, action):
            obs, rew, terminated, truncated, info = original_step(action)
            critic_obs = env.get_privileged_observations()

            self.env.env.obs_queue.append(obs["policy"])
            self.env.env.critic_queue.append(critic_obs)

            self.env.env.obs_buf_all = torch.stack([self.env.env.obs_queue[i]
                                           for i in range(training_config["max_actor_history"])], dim=1)
            self.env.env.critic_buf_all = torch.stack([self.env.env.critic_queue[i]
                                              for i in range(training_config["max_critic_history"])], dim=1)

            self.env.env.obs_input = self.env.env.obs_buf_all.reshape(self.env.num_envs, -1)
            self.env.env.critic_input = self.env.env.critic_buf_all.reshape(self.env.num_envs, -1)

            return obs, rew, terminated, truncated, info

        bound_method = types.MethodType(wrapper_step, env)
        setattr(env, "step", bound_method)

    env.get_observations = get_observations
    env.get_privileged_observations = get_privileged_observations
    wrapper_env_step()