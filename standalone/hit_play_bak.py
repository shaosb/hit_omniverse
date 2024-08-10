import argparse
import os
from hit_omniverse.utils.helper import setup_config

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=2, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=5, help="Spacing between different envs")
parser.add_argument("--policy_path", type=str, default="D:\humanoid\GITHUB\hit_omniverse\hit_omniverse\logs\HIT_all_dof_mlp\Aug09_18-28-34_8.9-2-reference-position\model_1100.pt", help="Model to be import")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Config file to be import")
parser.add_argument("--export_JIT", type=bool, default="False", help="Export weights in JIT")
parser.add_argument("--max_epochs", type=int, default=5000, help="Max epochs to play")

config = setup_config(parser.parse_args().training_config)
parser.add_argument("--config_file", type=str, default=config["robot"], help="Robot config file to be import")
os.environ["CONFIG"] = parser.parse_args().config_file

from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
from hit_omniverse.algo.ppo.on_policy_runner import OnPolicyRunner
from hit_omniverse.algo.vec_env import add_env_variable, add_env_method
from hit_omniverse.utils.helper import setup_config, export_policy_as_jit
from hit_omniverse import HIT_SIM_LOGS_DIR

import torch
import gymnasium as gym
from tqdm import tqdm
import os
from collections import deque


def main():
    env_cfg = HITRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.sim.device = args_cli.device

    env = gym.make("HIT-Humanoid-Imitate-v0", cfg=env_cfg)
    action = torch.zeros_like(env.action_manager.action)
    _, _ = env.reset()
    _, _, _, _, _ = env.step(action)
    add_env_variable(env)
    add_env_method(env)
    obs, _ = env.reset()
    observation = obs["policy"]

    obs_queue = deque(maxlen=config["runner"]["max_actor_history"])
    for _ in range(config["runner"]["max_actor_history"]):
        obs_queue.append(torch.zeros(env.num_envs, env.num_obs,
                                             dtype=torch.float, device=env.device))

    train_cfg = {key: config[key] for key in ["runner", "policy", "algorithm"] if key in config}

    ppo_runner = OnPolicyRunner(env=env,
                                train_cfg=train_cfg,
                                device=args_cli.device
                                )

    ppo_runner.load(args_cli.policy_path)
    policy = ppo_runner.get_inference_policy(device=args_cli.device)

    if args_cli.export_JIT:
        path = os.path.join(HIT_SIM_LOGS_DIR, train_cfg["runner"]["experiment_name"], 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for _ in tqdm(range(args_cli.max_epochs)):
        obs_queue.append(observation)

        obs_buf_all = torch.stack([obs_queue[i]
                                   for i in range(config["runner"]["max_actor_history"])], dim=1)
        if config["policy"]["actor_class"] in ["SimpleMLP", "NormalizedSimpleMLP"]:
            obs_input = obs_buf_all.reshape(env.num_envs, -1)
        elif config["policy"]["actor_class"] in ["SimpleLSTM"]:
            obs_input = obs_buf_all

        actions = policy(obs_input)
        obs, rew, terminated, truncated, info = env.step(actions)
        observation = obs["policy"]

        print(actions)


if __name__ == '__main__':
    main()
    simulation_app.close()