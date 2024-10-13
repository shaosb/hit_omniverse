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

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse import HIT_SIM_DATASET_DIR
from hit_omniverse.utils.helper import TransCMU2HIT, calculate_eye_and_target
from hit_omniverse.utils.hit_keyboard import Se2Keyboard
from hit_omniverse.standalone.get_action_dataset import get_action
from hit_omniverse.algo.vec_env import add_env_variable, add_env_method
import hit_omniverse.extension.mdp as mdp
from hit_omniverse.utils.helper import setup_config, rotation_matrin, yaw_rotation_and_translation_matrix, interpolate_arrays

import torch
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.lab.assets import Articulation
# from omni.isaac.lab.utils.math import euler_xyz_from_quat
import omni.isaac.lab.utils.math as math_utils

config = setup_config(os.environ["CONFIG"])

def main():
	env_cfg = HITRLEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device

	env = gym.make("HIT-Humanoid-Imitate-v0", cfg=env_cfg)
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

	# keyboard = Se2Keyboard(v_x_sensitivity=0.001, v_y_sensitivity=0.001, omega_z_sensitivity=0.01) # 1.8,2.2,3
	keyboard = Se2Keyboard(v_x_sensitivity=0.001, v_y_sensitivity=0.001, omega_z_sensitivity=0.003)
	keyboard.reset()
	print(keyboard)
	# print(env.observation_manager.observation)

	asset: Articulation = env.scene["robot"]

	# dataset config
	gait_mapping = {
        config["GAIT"]["30-run_HIT"]: "dataset",
        config["GAIT"]["slope_lone"]: "slope_lone",
        config["GAIT"]["squat_walk"]: "squat_walk",
        config["GAIT"]["stair_full"]: "stair_full",
        config["GAIT"]["hit_save_people"]: "hit_save_people",
        config["GAIT"]["forsquat_down"]: "forsquat_down",
        config["GAIT"]["forsquat_up"]: "forsquat_up",
    }
	init_dataset = "30-run_HIT"
	dataset = gait_mapping[config["GAIT"][init_dataset]]
	# Initilization position
	pos_init = asset.data.root_pos_w.to(torch.float64)
	pos_init = pos_init.cpu().numpy()
	x_init = pos_init[-1][0]
	y_init = pos_init[-1][1]
	pos_init[-1][-1] = 0
	pos_init = torch.tensor(pos_init).to(env_cfg.sim.device)
	# Accumulated pos and rpy
	rpy = pos = None
	total_yaw = 0
	total_x = 0
	total_y = 0
	# Gait transformation
	tarnsform_switch = True
	transform = False
	count = 0
	len_transform = 0
	interval = 0.01

	while simulation_app.is_running():
		# env.render()
		# if count % 500 == 0:
		# 	obs, _ = env.reset()
		# asset.write_root_velocity_to_sim(torch.tensor([[[2.2, 0, 0, 0, 0, 0]]]))
		# print(keyboard.advance())
		if int(keyboard.advance()[-1]) != 0:
			dataset = gait_mapping[int(keyboard.advance()[-1])]
			env.command_manager.get_term(dataset).reset([0])
			pos = pos.cpu().numpy()
			pos[-1][-1] = 0
			pos_init = torch.tensor(pos).to(env_cfg.sim.device)
			print(f"Changing gait to {dataset}")
			transform = True
			if tarnsform_switch:
				temp1 = asset.data.root_state_w[:,:3].cpu().numpy()
				r, p, y = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
				temp2 = np.asarray([[r.item(), p.item(), y.item()]])
				temp2[temp2 > np.pi] -= 2 * np.pi
				status_init = np.concatenate((temp1, temp2, asset.data.joint_pos.cpu().numpy()), axis=1)[0]
				status_dataset = np.concatenate((mdp.generated_commands(env, dataset)["robot_world_xyz"].cpu().numpy(),
												 mdp.generated_commands(env, dataset)["robot_world_rpy"].cpu().numpy(), mdp.generated_commands(env, dataset)["dof_pos"].cpu().numpy()), axis=1)[0]
				t = status_dataset - status_init
				interpolated_array = interpolate_arrays(status_init, status_dataset, interval)
				len_transform = interpolated_array.shape[0]

		if transform and tarnsform_switch:
			if count < len_transform:
				pos = torch.tensor(interpolated_array[count, :3]).to(env_cfg.sim.device)
				rpy = interpolated_array[count, 3:6]
				action = torch.tensor([interpolated_array[count, 6:]]).to(env_cfg.sim.device)
				count += 1
				print(count, len_transform)
			else:
				count = 0
				transform = False
				env.command_manager.get_term(dataset).reset([0])
				print(f"tarnsform to {dataset} completed")
		else:
			action = mdp.generated_commands(env, dataset)["dof_pos"]

		pos = mdp.generated_commands(env, dataset)["robot_world_xyz"]
		rpy = mdp.generated_commands(env, dataset)["robot_world_rpy"].cpu().numpy()
		bias = torch.tensor([[0, 0, 0]]).cuda()
		total_x += keyboard.advance()[0]
		total_y += keyboard.advance()[1]
		keyboard_pos = torch.tensor([[total_x, total_y, 0]]).cuda()
		pos = pos + bias + keyboard_pos + pos_init
		# if rpy is not None:
			# T = rotation_matrin(rpy.tolist()[0][0], rpy.tolist()[0][1], rpy.tolist()[0][2])
		T = yaw_rotation_and_translation_matrix(total_yaw, x_init, y_init)
		temp = pos.cpu().numpy()[0]
		temp = np.append(temp, 1)
		temp = np.dot(T, temp)
		temp = temp[:3]
		pos = torch.tensor([temp]).to(env_cfg.sim.device)

		total_yaw += keyboard.advance()[2]
		keyboard_rpy = np.asarray([[0, 0, total_yaw]])
		rpy = rpy + keyboard_rpy
		rotation = R.from_euler('xyz', rpy, degrees=False)
		rot = np.roll(rotation.as_quat(), 1)
		rot = torch.tensor(rot).to(env_cfg.sim.device)

		pose = torch.cat((pos, rot), dim=1)
		# pos = asset.data.root_pos_w.cpu().numpy()[0]
		# pos = [pos[0] - 2.1, pos[1] + 0.5, pos[2] + 0.5]
		# rot = asset.data.root_quat_w.cpu().numpy()[0]
		# eye, target = calculate_eye_and_target(pos, rot)
		# env.unwrapped.sim.set_camera_view(eye, target)

		# print("action:",action)
		# if count >= 100:
		# 	action = torch.ones_like(env.action_manager.action)
		# for batch in data_loader:
		# 	action = batch[0]["walker/joints_pos"][:,DOF_INDEX]
		asset.write_root_pose_to_sim(pose)
		obs, rew, terminated, truncated, info = env.step(action)
		# temp = mdp.joint_pos(env)[0].cpu().numpy()
		# pass


if __name__ == '__main__':
	main()
	simulation_app.close()