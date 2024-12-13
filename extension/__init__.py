import gymnasium as gym
import os

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse.extension.hit_env_cfg_scene import HITRLSceneEnvCfg
from hit_omniverse.extension.hit_env_cfg_recover import HITRecoverRLEnvCfg
from hit_omniverse.extension.SA01_env_cfg import SA01RLEnvCfg

from hit_omniverse.configs.rsl_rl_ppo_cfg_transformer import HumanoidPPORunnerTransformerCfg
from hit_omniverse.configs.rsl_rl_ppo_cfg import HumanoidPPORunnerCfg


from omni.isaac.lab.envs import ManagerBasedRLEnv

gym.register(
    id="HIT-Humanoid-Imitate-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": HITRLEnvCfg,
        "rsl_rl_cfg_entry_point": HumanoidPPORunnerCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HumanoidPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="SA01-Humanoid-Imitate-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": SA01RLEnvCfg,
        "rsl_rl_cfg_entry_point": HumanoidPPORunnerCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HumanoidPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="HIT-Humanoid-Recover-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": HITRecoverRLEnvCfg,
        "rsl_rl_cfg_entry_point": HumanoidPPORunnerCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HumanoidPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="HIT-Humanoid-rigid-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    # disable_env_checker=True,
    kwargs={
        "cfg": HITRLSceneEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HumanoidPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)