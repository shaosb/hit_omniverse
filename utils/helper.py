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

from hit_omniverse import HIT_SIM_ROOT_DIR, HIT_SIM_CONFIGS_DIR

ALL_DOF_NAME = ["left_leg_hip_roll", "right_leg_hip_roll", "waist_yaw", "left_leg_hip_yaw",
                "right_leg_hip_yaw", "waist_pitch", "left_leg_hip_pitch", "right_leg_hip_pitch",
                "left_arm_pitch", "right_arm_pitch", "left_leg_knee_pitch", "right_leg_knee_pitch",
                "left_arm_roll", "right_arm_roll", "left_leg_ankle_pitch", "right_leg_ankle_pitch",
                "left_arm_yaw", "right_arm_yaw", "left_leg_ankle_roll", "right_leg_ankle_roll",
                "left_arm_forearm_pitch", "right_arm_forearm_pitch"]


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


def quaternion_to_rotation_matrix(quat):
	w, x, y, z = quat
	# 归一化四元数
	norm = np.sqrt(w * w + x * x + y * y + z * z)
	w, x, y, z = w / norm, x / norm, y / norm, z / norm

	# 计算旋转矩阵
	R = np.array([
		[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
		[2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
		[2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
	])
	return R


def calculate_eye_and_target(pos, quat, follow_distance=2.0, height_offset=1.0):
	# 将四元数转换为旋转矩阵
	rot_matrix = quaternion_to_rotation_matrix(quat)

	# 定义摄像头位置 (eye)
	# eye的位置在机器人的后方并稍微抬高
	eye_offset = np.array([-follow_distance, 0, height_offset])
	eye = pos + np.dot(rot_matrix, eye_offset)

	# 定义摄像头朝向的目标 (target)
	# target在机器人前方
	target_offset = np.array([follow_distance, 0, 0])
	target = pos + np.dot(rot_matrix, target_offset)

	return eye, target

def make_scene(env_cfg, scene_cfg:list):
    from omni.isaac.lab.assets import AssetBaseCfg
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.utils.math import quat_from_euler_xyz

    from hit_omniverse import HIT_SIM_ASSET_DIR

    import torch

    for obj in scene_cfg:
        assert type(obj) == dict
        assert obj.get("usd_path") is not None and type(obj.get("usd_path")) == str
        assert obj.get("pos") is not None and type(obj.get("pos")) == list and len(obj.get("pos")) == 3
        assert obj.get("rot") is not None and type(obj.get("rot")) == float

        usd_path = obj.get("usd_path")
        usd_name = usd_path.split("\\")[-1].split(".")[0]
        pos = obj.get("pos")
        rot = obj.get("rot")

        usd_cfg = AssetBaseCfg(
            prim_path=f"/World/{usd_name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(HIT_SIM_ASSET_DIR, "scene", usd_path),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(
                pos=tuple(pos),
                rot=quat_from_euler_xyz(roll=torch.tensor(0),
                                        pitch=torch.tensor(0),
                                        yaw=torch.tensor(rot),
                                        ),
            )
        )

        setattr(env_cfg.scene, usd_name, usd_cfg)

    return env_cfg

def rotation_matrin(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    
    return rotation_matrix


# def yaw_rotation_and_translation_matrix(yaw, x, y):
#     return np.array([
#         [np.cos(yaw), -np.sin(yaw), 0, x],
#         [np.sin(yaw), np.cos(yaw), 0, y],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])


def interpolate_arrays(start, end, interval):
    distance = np.linalg.norm(end - start)
    num_steps = int(np.ceil(distance / interval)) + 1
    return np.linspace(start, end, num_steps, axis=0)

# def yaw_rotation_and_translation_matrix(yaw, x, y, offset_x, offset_y=0):
#     cos_yaw = np.cos(yaw)
#     sin_yaw = np.sin(yaw)
#     return np.array([
#         [cos_yaw, -sin_yaw, 0, x - offset_x * (1 - cos_yaw) + offset_y * sin_yaw],
#         [sin_yaw, cos_yaw, 0, y - offset_x * sin_yaw - offset_y * (1 - cos_yaw)],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])

# def yaw_rotation_and_translation_matrix(yaw, x, y, z, offset_x, offset_y=0, offset_z=0):
#     cos_yaw = np.cos(yaw)
#     sin_yaw = np.sin(yaw)
#     return np.array([
#         [cos_yaw, -sin_yaw, 0, x - offset_x * (1 - cos_yaw) + offset_y * sin_yaw],
#         [sin_yaw, cos_yaw, 0, y - offset_x * sin_yaw - offset_y * (1 - cos_yaw)],
#         [0, 0, 1, z + offset_z],
#         [0, 0, 0, 1]
#     ])

def yaw_rotation_matrix(yaw):
    """生成仅包含偏航旋转的矩阵"""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array([
        [cos_yaw, -sin_yaw, 0, 0],
        [sin_yaw, cos_yaw, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translation_matrix(x, y, z):
    """生成仅包含平移的矩阵"""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

import numpy as np

def in_place_rotation(yaw, x, y, z):
    """
    在物体当前位置(x, y, z)进行原地旋转
    """
    # 创建平移矩阵
    translate_to_origin = np.array([
        [1, 0, 0, -x],
        [0, 1, 0, -y],
        [0, 0, 1, -z],
        [0, 0, 0, 1]
    ])
    
    translate_back = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    
    # 创建旋转矩阵
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation = np.array([
        [cos_yaw, -sin_yaw, 0, 0],
        [sin_yaw, cos_yaw, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # 组合变换：先移到原点，然后旋转，最后移回原位置
    return np.dot(translate_back, np.dot(rotation, translate_to_origin))

def translate(matrix, dx, dy, dz):
    """
    对给定的变换矩阵应用平移
    """
    translation = np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])
    return np.dot(translation, matrix)


def update_robot_position(pos_x, pos_y, pos_z, key_x, key_y, key_z, yaw):
    auto_move = np.array([pos_x, pos_y, pos_z])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    current_rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    rotated_auto_move = np.dot(current_rotation_matrix, auto_move)

    key_move = np.array([key_x, key_y, key_z])

    total_move = rotated_auto_move + key_move

    move_direction = np.array([total_move[0], total_move[1], 0])
    move_distance = np.linalg.norm(move_direction)
    
    if move_distance > 0:
        new_yaw = np.arctan2(move_direction[1], move_direction[0])
    else:
        new_yaw = yaw

    new_x = total_move[0]
    new_y = total_move[1]
    new_z = total_move[2]
    
    return new_x, new_y, new_z, new_yaw

def quaternion_multiply(q1, q2):
    """计算两个四元数的乘积"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def export_onnx(policy_dir, observation_dim, actor_critic, output_dir=None):
    model = copy.deepcopy(actor_critic.actor).to('cpu')
    policy_jit_model = torch.jit.script(model)

    # policy_jit_model = torch.jit.load(policy_dir)
    policy_jit_model.eval()

    if output_dir is None:
        directory = os.path.join(os.path.dirname(policy_dir), "output")
        if not os.path.exists(directory):
            os.makedirs(directory)
        policy_onnx_model = os.path.join(directory, "zqsa01_policy.onnx")
    else:
        policy_onnx_model = os.path.join(output_dir, "zqsa01_policy.onnx")

    test_input_tensor = torch.randn(1, observation_dim)

    torch.onnx.export(policy_jit_model,
                      test_input_tensor,
                      policy_onnx_model,  # params below can be ignored
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      )

    print(f"Succeed to output at {policy_onnx_model}")