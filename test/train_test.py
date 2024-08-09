import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=1, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--config_file", type=str, default="ppo_alldof_test.yaml", help="Config file to be import")
parser.add_argument("--log_dir", type=str, default="default", help="Config file to be import")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse import HIT_SIM_ROOT_DIR
from hit_omniverse.utils.helper import setup_config

from hit_omniverse.algo.ppo.on_policy_runner import OnPolicyRunner
from hit_omniverse.algo.vec_env import add_env_method, add_env_variable

from omni.isaac.lab_tasks.manager_based.classic import humanoid

import torch
import gymnasium as gym
import os
from datetime import datetime

config = setup_config(args_cli.config_file)


def main():
	env_cfg = HITRLEnvCfg()
	env_cfg.scene.num_envs = args_cli.num_envs
	env_cfg.scene.env_spacing = args_cli.env_spacing
	env_cfg.sim.device = args_cli.device

	# env = gym.make("HIT-Humanoid-Imitate-v0", cfg=env_cfg)
	env_cfg = humanoid.humanoid_env_cfg.HumanoidEnvCfg()
	env = gym.make("Isaac-Humanoid-v0", cfg=env_cfg)
	action = torch.zeros_like(env.action_manager.action)
	_, _ = env.reset()
	_, _, _, _, _ = env.step(action)
	add_env_variable(env)
	add_env_method(env)
	_, _ = env.reset()

	train_cfg = {key: config[key] for key in ["runner", "policy", "algorithm"] if key in config}

	experiment_name = train_cfg["runner"]["experiment_name"]
	experiment_version = train_cfg["runner"]["run_name"]
	if args_cli.log_dir == "default":
		log_dir = os.path.join(HIT_SIM_ROOT_DIR, "logs", experiment_name, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + experiment_version)
	elif args_cli.log_dir is None:
		log_dir = None
	else:
		log_dir = os.path.join(args_cli.log_dir, experiment_name, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + experiment_version)

	ppo_runner = OnPolicyRunner(env=env,
								train_cfg=train_cfg,
								log_dir=log_dir,
								device=args_cli.device
								)

	ppo_runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"],
					 init_at_random_ep_len=True,
					 )

if __name__ == '__main__':
	main()
	simulation_app.close()