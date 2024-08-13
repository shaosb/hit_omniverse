import argparse
import os

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=10, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--config_file", type=str, default="robot_87_config.yaml", help="Config file to be import")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Config file to be import")

os.environ["CONFIG"] = parser.parse_args().config_file
os.environ["TRAINING_CONFIG"] = parser.parse_args().training_config

from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg_camera import HITCameraRLEnvCfg
from hit_omniverse import HIT_SIM_DATASET_DIR
from hit_omniverse.utils.helper import TransCMU2HIT
from hit_omniverse.standalone.get_action_dataset import get_action

import torch
import gymnasium as gym
from hit_omniverse.algo.vec_env import add_env_variable, add_env_method
import hit_omniverse.extension.mdp as mdp
from omni.isaac.lab.assets import Articulation
from hit_omniverse.utils.hit_keyboard import Se2Keyboard

import numpy as np
import matplotlib.pyplot as plt
import cv2


dataset_paths = [os.path.join(HIT_SIM_DATASET_DIR, "CMU_007_03.hdf5")]

def main():
	env_cfg = HITCameraRLEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device

	env = gym.make("HIT-Humanoid-play-camera", cfg=env_cfg)
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
	keyboard = Se2Keyboard(v_x_sensitivity=1.8, v_y_sensitivity=2.2, omega_z_sensitivity=3)
	keyboard.reset()
	print(keyboard)
	# print(env.observation_manager.observation)
	count = 0

	# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	# out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (480, 640))
	frame_count = 1

	asset: Articulation = env.scene["robot"]

	while simulation_app.is_running():
		count += 1
		# env.render()
		# if count % 500 == 0:
		# 	obs, _ = env.reset()
		# asset.write_root_velocity_to_sim(torch.tensor([[[2.2, 0, 0, 0, 0, 0]]]))
		print(keyboard.advance())
		if int(keyboard.advance()[0]) == int(keyboard.advance()[1]) == int(keyboard.advance()[2]):
			if int(keyboard.advance()[0]) == 1:
				break
			
		asset.write_root_velocity_to_sim(
			torch.tensor([[[keyboard.advance()[0], keyboard.advance()[1], 0, 0, 0, keyboard.advance()[-1]]]]))

		action = mdp.generated_commands(env, "dataset")["dof_pos"]

		# if count >= 100:
		# 	action = torch.ones_like(env.action_manager.action)
		# for batch in data_loader:
		# 	action = batch[0]["walker/joints_pos"][:,DOF_INDEX]
		obs, rew, terminated, truncated, info = env.step(action)
		# temp = mdp.joint_pos(env)[0].cpu().numpy()
		# pass

		# tensor = env.scene["RGB_camera"].data.output["rgb"].cpu()
		# frame = tensor.numpy()[0]
		# frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
		#
		# out.write(frame_bgr)

		output_dir = "output_images"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		tensor = env.scene["RGB_camera"].data.output["rgb"].cpu()  # 更新 Tensor

		# 创建一个图形并保存
		fig, ax = plt.subplots()
		ax.imshow(tensor.numpy()[0])
		ax.axis('off')  # 关闭坐标轴

		# 保存图像
		frame_count += 1
		save_path = os.path.join(output_dir, f"{frame_count}.png")
		plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

	# out.release()
	print("Write video success")
	print("End simulation ")
	simulation_app.close()


if __name__ == '__main__':
	main()
	simulation_app.close()