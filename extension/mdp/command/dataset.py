"""
Make a Dataset by robot pose
created by ssb 24.7.22
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from hit_omniverse import HIT_SIM_DATASET_DIR

import os
import csv


def read_csv(file_path):
	time = []
	pos = []
	ori = []
	dof_pos = []
	dof_vel = []
	base_vel = []
	base_ang = []

	data = {"time": time,
			"pos": pos,
			"ori": ori,
			"dof_pos": dof_pos,
			"dof_vel": dof_vel,
			"base_vel": base_vel,
			"base_ang": base_ang,
			}

	with open(file_path, "r") as file:
		reader = csv.reader(file)

		for _ in range(4):
			next(reader)

		for index, row in enumerate(reader):
			if index % 5 == 0 and index >= 6540:
				row_float = [float(item) for item in row]
				row = row_float
				time.append(row[0])
				pos.append(row[1:4])
				ori.append(row[4:8])
				dof_pos.append(row[8:30])
				base_vel.append(row[30:33])
				base_ang.append(row[33:36])
				dof_vel.append(row[36:62])

		# for row in reader:
		# 	row_float = [float(item) for item in row]
		# 	row = row_float
		# 	time.append(row[0])
		# 	pos.append(row[1:4])
		# 	ori.append(row[4:8])
		# 	dof_pos.append(row[8:30])
		# 	base_vel.append(row[30:33])
		# 	base_ang.append(row[33:36])
		# 	dof_vel.append(row[36:62])

	return data


def read_npz(file_path):
	npz = np.load(file_path)

	length = len(npz["q_hip_rotation_l"])
	waist_pitch = np.zeros(length)
	l_ankle_pitch = np.zeros(length)
	r_ankle_pitch = np.zeros(length)
	dof_pos_numpy = [npz["q_hip_adduction_l"], npz["q_hip_adduction_r"], npz["q_back_bkz"], npz["q_hip_rotation_l"],
					 npz["q_hip_rotation_r"],
					 waist_pitch, npz["q_hip_flexion_l"], npz["q_hip_flexion_r"], npz["q_l_arm_shy"],
					 npz["q_r_arm_shy"], npz["q_knee_angle_l"],
					 npz["q_knee_angle_r"], npz["q_l_arm_shx"], npz["q_r_arm_shx"], npz["q_ankle_angle_l"],
					 npz["q_ankle_angle_r"],
					 npz["q_l_arm_shz"], npz["q_r_arm_shz"], l_ankle_pitch, r_ankle_pitch,  npz["q_left_elbow"],
					 npz["q_right_elbow"]]
	dof_vel_numpy = [npz["dq_hip_adduction_l"], npz["dq_hip_adduction_r"], npz["dq_back_bkz"], npz["dq_hip_rotation_l"],
					 npz["dq_hip_rotation_r"],
					 waist_pitch, npz["dq_hip_flexion_l"], npz["dq_hip_flexion_r"], npz["dq_l_arm_shy"],
					 npz["dq_r_arm_shy"], npz["dq_knee_angle_l"],
					 npz["dq_knee_angle_r"], npz["dq_l_arm_shx"], npz["dq_r_arm_shx"], npz["dq_ankle_angle_l"],
					 npz["dq_ankle_angle_r"],
					 npz["dq_l_arm_shz"], npz["dq_r_arm_shz"], l_ankle_pitch, r_ankle_pitch,  npz["dq_left_elbow"],
					 npz["dq_right_elbow"]]

	dof_pos = np.column_stack(dof_pos_numpy)
	dof_vel = np.column_stack(dof_vel_numpy)
	time = [i for i in range(length)]

	data = {"time": time,
			"dof_pos": dof_pos,
			"dof_vel": dof_vel,
			}
	return data

def read_hit_npz(file_path):
	npz = np.load(file_path)

	length = len(npz["q_hip_rotation_l"])
	mapping = {
		'l_hip_roll': 'q_hip_adduction_l',
		'r_hip_roll': 'q_hip_adduction_r',
		'waist_yaw': 'q_back_bkz',
		'l_hip_yaw': 'q_hip_rotation_l',
		'r_hip_yaw': 'q_hip_rotation_r',
		'waist_pitch': 'q_back_bky',
		'l_hip_pitch': 'q_hip_flexion_l',
		'r_hip_pitch': 'q_hip_flexion_r',
		'left_arm_pitch': 'q_l_arm_shy',
		'right_arm_pitch': 'q_r_arm_shy',
		'l_knee': 'q_knee_angle_l',
		'r_knee': 'q_knee_angle_r',
		'left_arm_roll': 'q_l_arm_shx',
		'right_arm_roll': 'q_r_arm_shx',
		'l_ankle_pitch': 'q_ankle_angle_l',
		'r_ankle_pitch': 'q_ankle_angle_r',
		'left_arm_yaw': 'q_l_arm_shz',
		'right_arm_yaw': 'q_r_arm_shz',
		'l_ankle_roll': 'q_l_ankle_roll',
		'r_ankle_roll': 'q_r_ankle_roll',
		'left_arm_forearm_pitch': 'q_left_elbow',
		'right_arm_forearm_pitch': 'q_right_elbow',
		'x': 'q_pelvis_tx',
		'y': 'q_pelvis_tz',
		'z': 'q_pelvis_ty',
		'roll': 'q_pelvis_list',
		'pitch': 'q_pelvis_tilt',
		'yaw': 'q_pelvis_rotation',
	}

	dof_pos_numpy = [
		npz[mapping["l_hip_roll"]], npz[mapping["r_hip_roll"]], npz[mapping["waist_yaw"]], npz[mapping["l_hip_yaw"]],
		npz[mapping["r_hip_yaw"]], npz[mapping["waist_pitch"]], npz[mapping["l_hip_pitch"]], npz[mapping["r_hip_pitch"]],
		npz[mapping["left_arm_pitch"]], npz[mapping["right_arm_pitch"]], npz[mapping["l_knee"]],
		npz[mapping["r_knee"]], npz[mapping["left_arm_roll"]], npz[mapping["right_arm_roll"]],
		npz[mapping["l_ankle_pitch"]], npz[mapping["r_ankle_pitch"]], npz[mapping["left_arm_yaw"]],
		npz[mapping["right_arm_yaw"]], npz[mapping["l_ankle_roll"]], npz[mapping["r_ankle_roll"]],
		npz[mapping["left_arm_forearm_pitch"]], npz[mapping["right_arm_forearm_pitch"]]
	]

	robot_world_xyz = [npz[mapping["x"]], npz[mapping["y"]], npz[mapping["z"]]]
	robot_world_rpy = [npz[mapping["roll"]], npz[mapping["pitch"]], npz[mapping["yaw"]]]
	# robot_world_rpy = [np.zeros(length), np.zeros(length), np.zeros(length)]

	dof_pos = np.column_stack(dof_pos_numpy)
	robot_world_xyz = np.column_stack(robot_world_xyz)
	robot_world_rpy = np.column_stack(robot_world_rpy)
	time = [i for i in range(length)]

	data = {"time": time,
			"dof_pos": dof_pos,
			"robot_world_xyz": robot_world_xyz,
			"robot_world_rpy": robot_world_rpy,
			}
	return data


def read_txt(file_path):
	pass


def read_npy(file_path):
	from poselib.skeleton.skeleton3d import SkeletonMotion

	motion = SkeletonMotion.from_file(file_path)
	# motion = np.load(file_path, allow_pickle=True)
	node_names = motion.skeleton_tree.node_names

	length = len(motion.global_transformation.numpy()[:, 0, :])
	
	data = {}
	data = {name: motion.global_transformation.numpy()[:, i, :] for i, name in enumerate(node_names)}
	time = [i for i in range(length)]
	data.update({"time": time})
	# data.update({"dof_pos": motion.dof_pos})
	# data.update({"dof_vel": motion.dof_vels})

	return data


class BaseDataset(Dataset):
	def __init__(self, file: str):
		self.file = os.path.join(HIT_SIM_DATASET_DIR, file)
		if self.file.endswith("csv"):
			self.data = read_csv(self.file)
		elif self.file.endswith("txt"):
			self.data = read_txt(self.file)
		elif self.file.endswith("npz"):
			self.data = read_npz(self.file)
		elif self.file.endswith("hit"):
			self.data = read_hit_npz(self.file)
		elif self.file.endswith("npy"):
			self.data = read_npy(self.file)

	def __getitem__(self, index):
		item = {}
		for key in self.data.keys():
			item.update({key: torch.tensor(self.data.get(key)[index])})
		return item

	def __len__(self):
		return len(self.data.get("time"))


def custom_collate_fn(batch):
	# data = {}
	# print(batch)
	# for key in batch.keys():
	# 	data.update({key: batch.get(key)})
	return batch


def get_dataLoader(file: str,
				   batch_size=1,
				   shuffle=False,
				   num_workers=0,
				   pin_memory=True,
				   drop_last=True,
				   collate_fn=custom_collate_fn,
				   ):

	return DataLoader(BaseDataset(file),
					  batch_size=batch_size,
					  shuffle=shuffle,
					  num_workers=num_workers,
					  pin_memory=pin_memory,
					  drop_last=drop_last,
					  collate_fn=collate_fn,
					  )


def get_dataset(file: str):
	file = os.path.join(HIT_SIM_DATASET_DIR, file)
	if file.endswith("csv"):
		data = read_csv(file)
	elif file.endswith("txt"):
		data = read_txt(file)
	elif file.endswith("npz"):
		data = read_npz(file)
	elif file.endswith("hit"):
		data = read_hit_npz(file)
	elif file.endswith("npy"):
		data = read_npy(file)

	return data