import argparse
import os
from hit_omniverse.utils.helper import setup_config

parser = argparse.ArgumentParser(description="HIT humanoid robot exhibit in isaac sim")
parser.add_argument("--num_envs", type=int, default=3, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=5, help="Spacing between different envs")
parser.add_argument("--policy_path", type=str, default="D:\humanoid\GITHUB\hit_omniverse\hit_omniverse\logs\HIT_all_dof_mlp\Aug10_12-56-30_8.10-1-reference-position\model_1300.pt", help="Model to be import")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Config file to be import")
parser.add_argument("--export_JIT", type=bool, default="False", help="Export weights in JIT")
parser.add_argument("--max_epochs", type=int, default=5000, help="Max epochs to play")

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
from hit_omniverse.algo.ppo.on_policy_runner import OnPolicyRunner
from hit_omniverse.algo.vec_env import add_env_variable, add_env_method
from hit_omniverse.utils.helper import setup_config, export_policy_as_jit
from hit_omniverse import HIT_SIM_LOGS_DIR
import hit_omniverse.extension.mdp as mdp

import torch
import gymnasium as gym
from tqdm import tqdm
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

    asset_reference = env.scene["robot_reference"]
    frame_count = 0
    for _ in tqdm(range(args_cli.max_epochs)):
        actions = policy(env.env.env.obs_input)

        asset_reference.write_root_velocity_to_sim(torch.tensor([[[2.2, 0, 0, 0, 0, 0]]]))
        actions_reference = mdp.generated_commands(env, "dataset")["dof_pos"]
        actions = torch.cat((actions, actions_reference), dim=1)

        obs, rew, terminated, truncated, info = env.step(actions)
        observation = obs["policy"]

        print(env.scene["RGB_camera"].data.output["rgb"].shape)

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

        # 关闭当前图形，释放内存
        plt.close(fig)
        print("succeed save")
        # print(env.env.env.obs_input)
        # print(actions)


if __name__ == '__main__':
    main()
    simulation_app.close()