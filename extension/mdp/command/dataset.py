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
	dof_pos_numpy = [npz["q_hip_adduction_r"], npz["q_hip_adduction_r"], npz["q_back_bkz"], npz["q_hip_rotation_l"],
					 npz["q_hip_rotation_r"],
					 waist_pitch, npz["q_hip_flexion_l"], npz["q_hip_flexion_r"], npz["q_l_arm_shy"],
					 npz["q_r_arm_shy"], npz["q_knee_angle_l"],
					 npz["q_knee_angle_r"], npz["q_l_arm_shx"], npz["q_r_arm_shx"], npz["q_ankle_angle_l"],
					 npz["q_ankle_angle_r"],
					 npz["q_l_arm_shz"], npz["q_r_arm_shz"], l_ankle_pitch, r_ankle_pitch,  npz["q_left_elbow"],
					 npz["q_right_elbow"]]
	dof_vel_numpy = [npz["dq_hip_adduction_r"], npz["dq_hip_adduction_r"], npz["dq_back_bkz"], npz["dq_hip_rotation_l"],
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


def read_txt(file_path):
	pass


class BaseDataset(Dataset):
	def __init__(self, file: str):
		self.file = os.path.join(HIT_SIM_DATASET_DIR, file)
		if self.file.endswith("csv"):
			self.data = read_csv(self.file)
		elif self.file.endswith("txt"):
			self.data = read_txt(self.file)
		elif self.file.endswith("npz"):
			self.data = read_npz(self.file)

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
