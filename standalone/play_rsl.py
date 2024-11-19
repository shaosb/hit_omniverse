import argparse
from hit_omniverse.utils.helper import setup_config
import os

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=2, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Training config file to be import")
parser.add_argument("--task_name", type=str, default="HIT-Humanoid-Imitate-v0", help="Name of the task")
parser.add_argument("--experiment_name", type=str, default="HIT-Humanoid-Imitate-v0", help="Experiment name")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_path", type=str, default="2024-11-19_11-03-08\\model_1000.pt", help="Model to be import")


config = setup_config(parser.parse_args().training_config)
os.environ["TRAINING_CONFIG"] = parser.parse_args().training_config
parser.add_argument("--config_file", type=str, default=config["robot"], help="Robot config file to be import")
os.environ["CONFIG"] = parser.parse_args().config_file

from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse import HIT_SIM_LOGS_DIR

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry
from hit_omniverse.rsl_rl.runners import OnPolicyRunner
from hit_omniverse.rsl_rl.env import HistoryEnv

import hit_omniverse.extension.mdp as mdp

import torch
import gymnasium as gym
from datetime import datetime
from pprint import pprint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main():
	env_cfg = HITRLEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device
	agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task_name, "rsl_rl_cfg_entry_point")
	policy_path = os.path.join(HIT_SIM_LOGS_DIR, "rsl_rl", args_cli.experiment_name, args_cli.log_path)

	env = gym.make(args_cli.task_name, cfg=env_cfg)
	# env = RslRlVecEnvWrapper(env)
	env = HistoryEnv(env, agent_cfg.to_dict())
	# env.seed(args_cli.seed)

	runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
	runner.load(policy_path)
	policy = runner.get_inference_policy(device=agent_cfg.device)

	obs, _ = env.get_observations()
	asset = env.unwrapped.scene["robot"]
	# actions = mdp.generated_commands(env.unwrapped, "dataset")["dof_pos"]
	while simulation_app.is_running():
		with torch.inference_mode():
			actions = policy(obs)
			obs, rewards, dones, infos = env.step(actions)
		# print(f"is alive:{mdp.is_alive(env.unwrapped)}")
		# print(f"terminal:{mdp.is_terminated(env.unwrapped)}")
		# print(f"smooth:{mdp.reward_action_smooth(env.unwrapped)}")
		# temp = mdp.constant_commands(env.unwrapped)[:,3:6] - asset.data.root_ang_vel_w[:, :]
		# print(torch.exp(-torch.sum(torch.square(temp))))
		# temp1 = mdp.constant_commands(env.unwrapped)[:,0:3] - asset.data.root_lin_vel_w[:, :]
		# print(torch.exp(-torch.sum(torch.square(temp1))))

		# print(mdp.is_alive(env.unwrapped))
		# print(mdp.base_yaw_roll(env.unwrapped))
		# print(mdp.torques(env.unwrapped))
		# print(mdp.reward_feet_contact_force(env.unwrapped))
		# print(mdp.track_ang(env.unwrapped))
		# print(mdp.torques(env.unwrapped))
		# print(mdp.joint_lower_pos_distance(env.unwrapped))
		# print(mdp.joint_toqrue(env.unwrapped))
		# print(mdp.last_action(env.unwrapped))

	env.close()

if __name__ == '__main__':
	main()
	simulation_app.close()