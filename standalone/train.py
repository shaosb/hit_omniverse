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

	# TODO be pythonic
	# observation
	env.unwrapped.last_feet_z = 0.05
	env.unwrapped.feet_height = torch.zeros((env.num_envs, 2), device=env.device)
	env.unwrapped.last_contacts = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device, requires_grad=False)
	env.unwrapped.feet_air_time = torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device, requires_grad=False)

	action = torch.zeros_like(env.action_manager.action)
	_, _ = env.reset()
	_, _, _, _, _ = env.step(action)
	add_env_variable(env)
	add_env_method(env)
	_, _ = env.reset()

	train_cfg = {key: config[key] for key in ["runner", "policy", "algorithm"] if key in config}
	pprint(train_cfg)

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