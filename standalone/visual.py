import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import multiprocessing as mp
import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="A vritual system of HIT humanoid robot")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=2, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--policy", type=str, default=None, help="Policy file to be import")
parser.add_argument("--config_file", type=str, default="robot_87_config.yaml", help="Config file to be import")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
import hit_omniverse.extension.mdp as mdp
from hit_omniverse.utils.helper import DynamicPlotApp


def startup_sim(queue:mp.Queue):
    env_cfg = HITRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.sim.device = args_cli.device

    env = gym.make("HIT-Humanoid-Imitate-v0", cfg=env_cfg)
    #TODO be pythonic
    # observation
    env.unwrapped.last_feet_z = 0.05
    env.unwrapped.feet_height = torch.zeros((env.num_envs, 2), device=env.device)
    env.unwrapped.last_contacts = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device, requires_grad=False)
    env.unwrapped.feet_air_time = torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device,
                                              requires_grad=False)

    obs, _ = env.reset()

    while simulation_app.is_running():
        # action = mdp.generated_commands(env, "dataset")
        action = mdp.generated_commands(env, "dataset")["dof_pos"]
        obs, rew, terminated, truncated, info = env.step(action)

        joint_pos = mdp.joint_pos(env)[0].cpu().numpy()
        reference = action[0].cpu().numpy()
        data = []
        for i in range(len(joint_pos)):
            data.append([joint_pos[i], reference[i]])
        queue.put(data)


def get_atcion(env, policy, observation):
    if policy is not None:
        return policy(observation)
    return mdp.generated_commands(env, "dataset")


if __name__ == "__main__":
    queue = mp.Queue(maxsize=50)
    num_tensors = 22

    data_process = mp.Process(target=startup_sim, args=(queue, ))
    data_process.start()

    root = tk.Tk()
    app = DynamicPlotApp(root, queue, num_tensors)
    root.mainloop()

    data_process.terminate()
