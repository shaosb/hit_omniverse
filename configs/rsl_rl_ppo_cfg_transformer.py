# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from hit_omniverse.rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent
)
from dataclasses import MISSING


@configclass
class RslRlPpoActorCriticTransformerCfg:
    """Configuration for the PPO actor-critic transformer networks."""
    class_name: str = "ActorCriticTransformer"
    """The policy class name. Default is ActorCriticTransformer."""

    obs_context_len: int = MISSING
    """The length of observation context."""

    num_actor_obs: int = MISSING

    num_critic_obs: int = MISSING

    num_actions: int = MISSING


@configclass
class HumanoidPPORunnerTransformerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 100000
    save_interval = 50
    experiment_name = "humanoid"
    empirical_normalization = False
    obs_context_len = 4
    privileged_context_len = 4
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     # actor_hidden_dims=[400, 200, 100],
    #     # critic_hidden_dims=[400, 200, 100],
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    # )
    policy = RslRlPpoActorCriticTransformerCfg(
        num_actor_obs=80,
        num_critic_obs=80,
        num_actions=22,
        obs_context_len=4,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
