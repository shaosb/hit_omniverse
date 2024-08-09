import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=1, help="Spacing between different envs")
parser.add_argument("--policy_path", type=str, default="model/model1.pt", help="Model to be import")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--config_file", type=str, default="robot_alldof_config", help="Config file to be import")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse import HIT_SIM_DATASET_DIR
from omni.isaac.lab_tasks.manager_based.classic import humanoid

from hit_omniverse.utils.helper import TransCMU2HIT
from hit_omniverse.standalone.get_action_dataset import get_action

import torch
import gymnasium as gym
import os
from hit_omniverse.algo.vec_env import add_env_variable, add_env_method

dataset_paths = [os.path.join(HIT_SIM_DATASET_DIR, "CMU_007_03.hdf5")]

def main():
	env_cfg = HITRLEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device

	env_cfg = humanoid.humanoid_env_cfg.HumanoidEnvCfg()
	env = gym.make("Isaac-Humanoid-v0", cfg=env_cfg)
	# env = gym.make("HIT-Humanoid-Imitate-v0", cfg=env_cfg)
	# add_env_variable(env)
	# add_env_method(env)
	# print(env.num_actions)

	# print(env.action_space.shape[1], type(env.action_space.shape[1]))
	# print(env.observation_space["policy"].shape[1], type(env.observation_space["policy"].shape[1]))
	# print(env.observation_space["privileged"])


	# DOF_INDEX = torch.tensor(TransCMU2HIT())
	# data_loader = get_action(train_dataset_paths=dataset_paths)
	# temp = iter(data_loader)
	#
	# # file_bytes = read_file(args_cli.policy_path)
	# # policy = torch.jit.load(file_bytes).to(env.device).eval()
	#
	obs, _ = env.reset()
	# print(env.observation_manager.observation)
	count = 0
	while simulation_app.is_running():
		count += 1
		# env.render()
		# if count % 500 == 0:
		# 	obs, _ = env.reset()
		action = torch.zeros_like(env.action_manager.action)
		# if count >= 100:
		# 	action = torch.ones_like(env.action_manager.action)
		# for batch in data_loader:
		# 	action = batch[0]["walker/joints_pos"][:,DOF_INDEX]
		obs, rew, terminated, truncated, info = env.step(action)
		# 	print(obs["obs"], rew, terminated, truncated, info)
		# obs, _ = env.reset()
		# print(rew)
		# print(terminated)
		# print(obs["policy"])


if __name__ == '__main__':
	main()
	simulation_app.close()