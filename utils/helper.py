"""
Helper function for HIT robot

created by ssb in 2024.6.25
"""

import os
import yaml
import copy
import torch
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import argparse

from hit_omniverse import HIT_SIM_ROOT_DIR, HIT_SIM_CONFIGS_DIR

ALL_DOF_NAME = ["left_leg_hip_roll", "right_leg_hip_roll", "waist_yaw", "left_leg_hip_yaw",
                "right_leg_hip_yaw", "waist_pitch", "left_leg_hip_pitch", "right_leg_hip_pitch",
                "left_arm_pitch", "right_arm_pitch", "left_leg_knee_pitch", "right_leg_knee_pitch",
                "left_arm_roll", "right_arm_roll", "left_leg_ankle_pitch", "right_leg_ankle_pitch",
                "left_arm_yaw", "right_arm_yaw", "left_leg_ankle_roll", "right_leg_ankle_roll",
                "left_arm_forearm_pitch", "right_arm_forearm_pitch"]

def get_args():
    parser = argparse.ArgumentParser(description="HIT humanoid robot in isaac sim")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of robot to spawn")
    parser.add_argument("--env_spacing", type=int, default=1, help="Spacing between different envs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for running")
    parser.add_argument("--config_file", type=str, help="Config file to be import")

    return parser

def TransCMU2HIT():
    config = setup_config("robot_config.yaml")

    CMU_DOF = config["CUM_DOF"]
    MAPPING = config["DOF_MAPPING"]

    DOF_NAME = list(config["DOF_ASSET"].keys())
    HIT_DOF_INDEX = [0 for _ in range(len(DOF_NAME))]

    for index, value in enumerate(CMU_DOF):
        try:
            if MAPPING[value] in DOF_NAME:
                HIT_DOF_INDEX[DOF_NAME.index(MAPPING[value])] = index
        except:
            pass

    return HIT_DOF_INDEX


def setup_config(file_path):
    file_path = os.path.join(HIT_SIM_CONFIGS_DIR, file_path)
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    base_files = data.get('_BASE_')
    if base_files:
        if isinstance(base_files, str):
            base_files = [base_files]
        elif not isinstance(base_files, list):
            raise ValueError('_BASE_ must be a string or a list of strings')

        merged_data = {}
        for base_file in base_files:
            base_file = os.path.join(HIT_SIM_CONFIGS_DIR, base_file)
            with open(base_file, 'r') as base:
                base_data = yaml.safe_load(base)
            merged_data = {**merged_data, **base_data}

        del data['_BASE_']
        merged_data = {**merged_data, **data}
        return merged_data

    else:
        return data


def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "export_policy.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


class DynamicPlotApp:
    def __init__(self, root, queue, num_tensors):
        self.root = root
        self.queue = queue
        self.num_tensors = num_tensors

        # 图形的尺寸
        self.figure_width = 280
        self.figure_height = 200
        # 每行的图形数量
        self.columns = 6
        # 图形之间的间距
        self.padx = 5
        self.pady = 5

        # 创建一个主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建一个Canvas和滚动条
        self.canvas = tk.Canvas(self.main_frame)
        self.scroll_y = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(self.main_frame, orient="horizontal", command=self.canvas.xview)

        # 创建一个框架用于放置内容
        self.canvas_frame = tk.Frame(self.canvas)

        # 在Canvas上创建一个窗口
        self.canvas.create_window((0, 0), window=self.canvas_frame, anchor="nw")

        # 配置Canvas和滚动条
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        # 布局
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.figures = []
        self.axes = []
        self.canvases = []
        self.data_history = [[] for _ in range(num_tensors)]

        for i in range(num_tensors):
            fig, ax = plt.subplots()
            self.figures.append(fig)
            self.axes.append(ax)

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.config(width=self.figure_width, height=self.figure_height)

            # 设置图形的网格位置
            row = i // self.columns
            column = i % self.columns
            canvas_widget.grid(row=row, column=column, padx=self.padx, pady=self.pady)

            self.canvases.append(canvas)

        # 更新Canvas区域以适应内容
        self.canvas_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # 初始窗口大小设置为尽可能容纳所有图形
        total_width = self.canvas.bbox("all")[2] + self.scroll_y.winfo_width()
        total_height = self.canvas.bbox("all")[3] + self.scroll_x.winfo_height()
        self.root.geometry(f"{int(total_width)}x{int(total_height)}")

        self.update_plots()

    def update_plots(self):
        while not self.queue.empty():
            new_data = self.queue.get()
            for i, tensor_data in enumerate(new_data):
                self.data_history[i].append(tensor_data)
                if len(self.data_history[i]) > 2000:
                    self.data_history[i] = self.data_history[i][-2000:]

        for i in range(self.num_tensors):
            self.axes[i].clear()
            self.axes[i].plot(self.data_history[i])

            self.axes[i].set_title(ALL_DOF_NAME[i], fontsize=8)
            self.axes[i].set_xlabel('X-axis', fontsize=8)
            self.axes[i].set_ylabel('Y-axis', fontsize=8)
            self.axes[i].tick_params(axis='both', labelsize=6)

        for canvas in self.canvases:
            canvas.draw()

        # 更新Canvas区域以适应内容
        self.canvas_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.root.after(200, self.update_plots)


