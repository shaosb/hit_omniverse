import argparse
import os

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=10, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--config_file", type=str, default="robot_87_config.yaml", help="Config file to be import")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Config file to be import")

os.environ["TRAINING_CONFIG"] = parser.parse_args().training_config
os.environ["CONFIG"] = parser.parse_args().config_file

from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg_rigid import HITRighdEnvCfg
from hit_omniverse import HIT_SIM_DATASET_DIR
from hit_omniverse.utils.helper import TransCMU2HIT, quaternion_multiply

import torch
import gymnasium as gym
import hit_omniverse.extension.mdp as mdp
import numpy as np
from omni.isaac.lab.assets import Articulation
import time

LINK_NAMES = ['pelvis', 'r_hip_roll', 'r_hip_yaw',
			   'r_upper_leg', 'r_lower_leg', 'r_ankle',
			   'r_foot', 'l_hip_roll', 'l_hip_yaw',
			   'l_upper_leg', 'l_lower_leg', 'l_ankle',
			   'l_foot', 'waist_link1', 'body_link',
			   'right_arm_link1', 'right_arm_link2', 'right_arm_link3',
			   'right_arm_link4', 'left_arm_link1', 'left_arm_link2', 'left_arm_link3',
			   'left_arm_link4']

QUAT_INIT = {}

def main():
	env_cfg = HITRighdEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device

	env = gym.make("HIT-Humanoid-rigid-v0", cfg=env_cfg)
	# add_env_variable(env)
	# add_env_method(env)

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

	# body = env.scene["body"]
	body = {}
	for link in LINK_NAMES:
		body.update({link: env.scene[link]})
		QUAT_INIT.update({link:env.scene[link].data.body_quat_w.cpu().numpy()[0][0]})

	while simulation_app.is_running():
		count += 1
		# env.render()
		# if count % 500 == 0:
		# 	obs, _ = env.reset()
		# asset.write_root_velocity_to_sim(torch.tensor([[[2.2, 0, 0, 0, 0, 0]]]))
		# asset.write_root_velocity_to_sim(torch.tensor([[[1.9, 0, 0, 0, 0, 0]]]))
		# body.set_world_poses(positions=torch.tensor([[5,5,5]]))
		# body_transforms = [mdp.generated_commands(env, "imitation")[link].cpu().numpy() for link in link_name]
		# body_transforms = [np.concatenate((transform[0][4:], transform[0][:4])) for transform in body_transforms]
		# body_transforms = torch.tensor(body_transforms).to(env_cfg.sim.device)
		# body.set_transforms(body_transforms, torch.tensor([0]).to(env_cfg.sim.device))
		# action = torch.ones_like(env.action_manager.action)*0.01
		# if count >= 100:
		# 	action = torch.ones_like(env.action_manager.action)
		# for batch in data_loader:
		# 	action = batch[0]["walker/joints_pos"][:,DOF_INDEX]
		# obs, rew, terminated, truncated, info = env.step(action)
		# temp = mdp.joint_pos(env)[0].cpu().numpy()

		for link in LINK_NAMES:
			transform = mdp.generated_commands(env, "imitation")[link].cpu().numpy()
			transform += np.asarray([0, 0, 0, 0, 0, 0, 1])
			body_transforms = np.concatenate((transform[0][4:], [transform[0][3]], transform[0][:3]))
			body_transforms = np.concatenate((body_transforms[:3], quaternion_multiply(body_transforms[3:], QUAT_INIT[link])))
			body_transforms = torch.tensor(body_transforms).to(env_cfg.sim.device)
			body.get(link).write_root_pose_to_sim(body_transforms)

		# t = time.time()
		env.command_manager.compute(dt=env.step_dt)
		env.sim.step(render=False)
		env.scene.update(dt=env.physics_dt)
		# print(f"spend {time.time() - t}")

		env.render()
		# time.sleep(0.05)
		pass


if __name__ == '__main__':
	main()
	simulation_app.close()