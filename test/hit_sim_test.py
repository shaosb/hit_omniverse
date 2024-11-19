import argparse
import os
import cProfile

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=1, help="Spacing between different envs")
parser.add_argument("--policy_path", type=str, default="model/model1.pt", help="Model to be import")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--config_file", type=str, default="robot_87_config.yaml", help="Config file to be import")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Config file to be import")

os.environ["CONFIG"] = parser.parse_args().config_file
os.environ["TRAINING_CONFIG"] = parser.parse_args().training_config

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
import hit_omniverse.extension.mdp as mdp
# from hit_omniverse import HIT_SIM_DATASET_DIR
from omni.isaac.lab_tasks.manager_based.classic import humanoid
import torch
import gymnasium as gym
import os
import time

def main():
	env_cfg = HITRLEnvCfg()
	# env_cfg = humanoid.humanoid_env_cfg.HumanoidEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device

	# env = gym.make("Isaac-Humanoid-v0", cfg=env_cfg)
	env = gym.make("HIT-Humanoid-Imitate-v0", cfg=env_cfg)
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
		# action = torch.zeros_like(env.action_manager.action)
		#
		action = mdp.generated_commands(env, "dataset")["dof_pos"]
		# if count >= 100:
		# 	action = torch.ones_like(env.action_manager.action)
		# for batch in data_loader:
		# 	action = batch[0]["walker/joints_pos"][:,DOF_INDEX]
		# asset: Articulation = env.scene["robot"]
		# asset.write_root_velocity_to_sim(
		# 	torch.tensor([[[0.18, 0.22, 0, 0, 0, 0]]]))
		t = time.time()
		obs, rew, terminated, truncated, info = env.step(action)
		print(f"{time.time() - t}")
		# 	print(obs["obs"], rew, terminated, truncated, info)
		# obs, _ = env.reset()
		# print(rew)
		# print(terminated)
		# print(obs["policy"])

		if count >= 500:
			break


if __name__ == '__main__':
	# main()
	cProfile.run("main()", sort=1)
	simulation_app.close()