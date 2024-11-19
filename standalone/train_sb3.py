import argparse
from hit_omniverse.utils.helper import setup_config
import os

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=2, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Training config file to be import")
parser.add_argument("--log_dir", type=str, default="default", help="Config file to be import")

config = setup_config(parser.parse_args().training_config)
os.environ["TRAINING_CONFIG"] = parser.parse_args().training_config
parser.add_argument("--config_file", type=str, default=config["robot"], help="Robot config file to be import")
os.environ["CONFIG"] = parser.parse_args().config_file

from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse import HIT_SIM_ROOT_DIR

from hit_omniverse.algo.ppo.on_policy_runner import OnPolicyRunner
from hit_omniverse.algo.vec_env import add_env_method, add_env_variable

from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import torch
import gymnasium as gym
from datetime import datetime
from pprint import pprint


def main():
	env_cfg = HITRLEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device

	env = gym.make("HIT-Humanoid-Imitate-v0", cfg=env_cfg)

	env = Sb3VecEnvWrapper(env)
	model = PPO("MlpPolicy", env, verbose=1, batch_size=256, n_steps=512, n_epochs=10)
	model.learn(total_timesteps=5e7, progress_bar=True)
	model.save("model")
	evaluate_policy(model, env, n_eval_episodes=20)

	model.load("model")
	pass

if __name__ == '__main__':
	main()
	simulation_app.close()