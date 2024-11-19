import argparse
from hit_omniverse.utils.helper import setup_config
import os

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=2, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Training config file to be import")
parser.add_argument("--task_name", type=str, default="HIT-Humanoid-Imitate-v0", help="Name of the task")
parser.add_argument("--experiment_name", type=str, default="HIT-Humanoid-Imitate-v0", help="Experiment name")
parser.add_argument("--seed", type=int, default=3407, help="Seed used for the environment")


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
from hit_omniverse import HIT_SIM_LOGS_DIR

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry
from hit_omniverse.rsl_rl.runners import OnPolicyRunner
from hit_omniverse.rsl_rl.env import HistoryEnv

import torch
import gymnasium as gym
from datetime import datetime

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
	log_dir = os.path.join(HIT_SIM_LOGS_DIR, "rsl_rl", args_cli.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

	env = gym.make(args_cli.task_name, cfg=env_cfg)
	# env = RslRlVecEnvWrapper(env)
	env = HistoryEnv(env, agent_cfg.to_dict())

	runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

	# env.seed(args_cli.seed)

	runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
	env.close()

if __name__ == '__main__':
	main()
	simulation_app.close()