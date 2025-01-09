import argparse
from hit_omniverse.utils.helper import setup_config
import os

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=5, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=2.5, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--task_name", type=str, default="SA01-Humanoid-Imitate-v0", help="Name of the task")
parser.add_argument("--experiment_name", type=str, default="SA01-Humanoid-Imitate-v0", help="Experiment name")
parser.add_argument("--seed", type=int, default=3407, help="Seed used for the environment")
parser.add_argument("--log_path", type=str, default="2025-01-08_18-50-51\\model_48750.pt", help="Model to be import")
parser.add_argument("--config_file", type=str, default="SA01_config.yaml", help="Robot config file to be import")
parser.add_argument("--output", default=False, action="store_true", help="Whether to output onnx policy")
parser.add_argument("--output_dir", type=str, default=None, help="The output dirname of onnx policy")

os.environ["CONFIG"] = parser.parse_args().config_file

from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse.extension.SA01_env_cfg import SA01RLEnvCfg
from hit_omniverse.extension.hit_env_cfg_recover import HITRecoverRLEnvCfg
from hit_omniverse import HIT_SIM_LOGS_DIR
from hit_omniverse.utils.helper import export_onnx

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry
from hit_omniverse.rsl_rl.runners import OnPolicyRunner, OnPolicyTransformerRunner
from hit_omniverse.rsl_rl.env import HistoryEnv, TransformerEnv

import hit_omniverse.extension.mdp as mdp

import torch
import gymnasium as gym
from datetime import datetime
from pprint import pprint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

torch.set_printoptions(precision=3, sci_mode=False)

def main():
	# env_cfg = HITRLEnvCfg()
	# env_cfg = HITRecoverRLEnvCfg()
	env_cfg = SA01RLEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device
	agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task_name, "rsl_rl_cfg_entry_point")
	policy_path = os.path.join(HIT_SIM_LOGS_DIR, "rsl_rl", args_cli.experiment_name, args_cli.log_path)

	env = gym.make(args_cli.task_name, cfg=env_cfg)
	# env = RslRlVecEnvWrapper(env)
	env = HistoryEnv(env, agent_cfg.to_dict())
	# env = TransformerEnv(env, agent_cfg.to_dict())

	# env.seed(args_cli.seed)

	runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
	# runner = OnPolicyTransformerRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
	runner.load(policy_path)
	policy = runner.get_inference_policy(device=agent_cfg.device)

	obs, extras = env.get_observations()
	if args_cli.output:
		export_onnx(policy_path, obs.shape[1], runner.alg.actor_critic, args_cli.output_dir)

	asset = env.unwrapped.scene["robot"]
	while simulation_app.is_running():
		with torch.inference_mode():
			# actions = mdp.generated_commands(env.unwrapped, "dataset")["dof_pos"] * 2
			# print(actions)
			actions = policy(obs)
			# actions = torch.zeros([1,12])
			# print(obs)
			obs, rewards, dones, infos = env.step(actions)
			# print(actions[:,14:16],actions[:,18:20])
			# print(mdp.generated_commands(env.unwrapped, "dataset")["dof_pos"])
			# print(asset.data.joint_pos)
			# # 	  mdp.generated_commands(env.unwrapped, "dataset")["dof_pos"][:,18:20])
			# print("*" * 20)
		# print(env.unwrapped.command_manager.get_command("base_velocity"))
		# print(torch.norm(env.unwrapped.command_manager.get_command("base_velocity")[:, :2], dim=1))
		# print("*" * 20)
		# print(mdp.mdp.generated_commands(env.unwrapped, "base_velocity"))
		# print(f"is alive:{mdp.is_alive(env.unwrapped)}")
		# print(f"terminal:{mdp.is_terminated(env.unwrapped)}")
		# print(f"smooth:{mdp.reward_action_smooth(env.unwrapped)}")
		# temp = mdp.constant_commands(env.unwrapped)[:,3:6] - asset.data.root_ang_vel_w[:, :]
		# print(torch.exp(-torch.sum(torch.square(temp))))
		# temp1 = mdp.constant_commands(env.unwrapped)[:,0:3] - asset.data.root_lin_vel_w[:, :]
		# print(torch.exp(-torch.sum(torch.square(temp1))))
		# print(mdp.constant_commands(env.unwrapped))
		# print(mdp.track_lin_x(env.unwrapped))
		# print(mdp.is_alive(env.unwrapped))
		# print(mdp.base_yaw_roll(env.unwrapped))
		# print(mdp.torques(env.unwrapped))
		# print(mdp.reward_feet_contact_force(env.unwrapped))
		# print(mdp.track_ang(env.unwrapped))
		# print(mdp.torques(env.unwrapped))
		# print(mdp.joint_lower_pos_distance(env.unwrapped))
		# print(mdp.joint_toqrue(env.unwrapped))
		# print(mdp.last_action(env.unwrapped))
		# print(mdp.track_lin_x(env.unwrapped))
		pass

	env.close()

if __name__ == '__main__':
	main()
	simulation_app.close()