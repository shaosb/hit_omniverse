import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import multiprocessing as mp
import argparse
import os

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="A vritual system of HIT humanoid robot")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot to spawn")
parser.add_argument("--env_spacing", type=int, default=2, help="Spacing between different envs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
parser.add_argument("--policy", type=str, default=None, help="Policy file to be import")
parser.add_argument("--config_file", type=str, default="robot_87_config.yaml", help="Config file to be import")
parser.add_argument("--training_config", type=str, default="ppo_87_mlp.yaml", help="Config file to be import")
parser.add_argument("--task_name", type=str, default="HIT-Humanoid-Imitate-v0", help="Name of the task")
parser.add_argument("--experiment_name", type=str, default="HIT-Humanoid-Imitate-v0", help="Experiment name")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_path", type=str, default="2024-11-25_15-45-34\\model_700.pt", help="Model to be import")


os.environ["CONFIG"] = parser.parse_args().config_file
os.environ["TRAINING_CONFIG"] = parser.parse_args().training_config
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import os

from hit_omniverse.extension.hit_env_cfg import HITRLEnvCfg
import hit_omniverse.extension.mdp as mdp
from hit_omniverse.utils.helper import DynamicPlotApp, setup_config
from hit_omniverse import HIT_SIM_LOGS_DIR

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry
from hit_omniverse.rsl_rl.runners import OnPolicyRunner
from hit_omniverse.rsl_rl.env import HistoryEnv

config = setup_config(os.environ["CONFIG"])

gait_mapping = {
    config["GAIT"]["30-run_HIT"]: "dataset",
    config["GAIT"]["slope_lone"]: "slope_lone",
    config["GAIT"]["squat_walk"]: "squat_walk",
    config["GAIT"]["stair_full"]: "stair_full",
    config["GAIT"]["hit_save_people"]: "hit_save_people",
    config["GAIT"]["forsquat_down"]: "forsquat_down",
    config["GAIT"]["forsquat_up"]: "forsquat_up",
    config["GAIT"]["squat_with_people"]: "square_with_people",
}

init_dataset = "squat_walk"

def startup_sim(queue:mp.Queue):
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

    while simulation_app.is_running():
        # actions = mdp.generated_commands(env, "dataset")
        # actions = mdp.generated_commands(env.unwrapped, "dataset")["dof_pos"]
        actions = get_action(env, policy, obs)
        obs, rewards, dones, infos = env.step(actions)

        # joint_pos = mdp.joint_pos(env.unwrapped)[0].cpu().numpy()
        # joint_pos = numpy.asarray([0 for i in range(22)])
        joint_pos = mdp.generated_commands(env.unwrapped, "dataset")["dof_pos"][0].cpu().numpy()
        reference = actions[0].cpu().detach().numpy()
        data = []
        for i in range(len(joint_pos)):
            data.append([joint_pos[i], reference[i]])
        queue.put(data)


def get_action(env, policy, observation):
    if policy is not None:
        return policy(observation)
    return mdp.generated_commands(env, init_dataset)["dof_pos"]


if __name__ == "__main__":
    queue = mp.Queue(maxsize=50)
    num_tensors = 22

    data_process = mp.Process(target=startup_sim, args=(queue, ))
    data_process.start()

    root = tk.Tk()
    app = DynamicPlotApp(root, queue, num_tensors)
    root.mainloop()

    data_process.terminate()
