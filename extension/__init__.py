import gymnasium as gym
from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse.extension.hit_env_cfg_camera import HITCameraRLEnvCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv

HITRLEnvCfg = HITRLEnvCfg()

gym.register(
    id="HIT-Humanoid-Imitate-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    # disable_env_checker=True,
    kwargs={
        "cfg": HITRLEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HumanoidPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="HIT-Humanoid-play-camera",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    # disable_env_checker=True,
    kwargs={
        "cfg": HITCameraRLEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HumanoidPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)