import numpy as np

class Bone:
    def __init__(self, name, parent, direction, length, axis, dof, limits):
        self.name = name
        self.parent = parent
        self.direction = np.array(direction)
        self.length = length
        self.axis = np.array(axis)
        self.dof = dof
        self.limits = limits
        self.rotation = np.zeros(3)  # Initial rotation is zero

def parse_asf(filename):
    bones = {}
    current_bone = None
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == ':bonedata':
                continue
            elif tokens[0] == 'begin':
                current_bone = {}
            elif tokens[0] == 'end':
                bones[current_bone['name']] = Bone(
                    name=current_bone['name'],
                    parent=current_bone.get('parent', None),
                    direction=[float(x) for x in current_bone['direction']],
                    length=float(current_bone['length']),
                    axis=[float(x) for x in current_bone['axis']],
                    dof=current_bone.get('dof', []),
                    limits=current_bone.get('limits', [])
                )
                current_bone = None
            elif current_bone is not None:
                if tokens[0] == 'name':
                    current_bone['name'] = tokens[1]
                elif tokens[0] == 'direction':
                    current_bone['direction'] = tokens[1:]
                elif tokens[0] == 'length':
                    current_bone['length'] = tokens[1]
                elif tokens[0] == 'axis':
                    current_bone['axis'] = tokens[1:4]
                elif tokens[0] == 'dof':
                    current_bone['dof'] = tokens[1:]
                elif tokens[0] == 'limits':
                    current_bone['limits'] = tokens[1:]
    return bones

def parse_amc(filename, bones):
    motion_data = []
    current_frame = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0].isdigit():
                if current_frame:
                    motion_data.append(current_frame)
                current_frame = {'frame': int(tokens[0])}
            elif tokens[0] in bones:
                current_frame[tokens[0]] = [float(x) for x in tokens[1:]]
    if current_frame:
        motion_data.append(current_frame)
    return motion_data


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_skeleton(bones, frame_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_bone(bone, origin):
        direction = bone.direction
        endpoint = origin + direction * bone.length
        ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]], [origin[2], endpoint[2]], 'b-')
        return endpoint

    def recursive_plot(bone_name, origin):
        bone = bones[bone_name]
        endpoint = plot_bone(bone, origin)
        for child_name, child_bone in bones.items():
            if child_bone.parent == bone_name:
                recursive_plot(child_name, endpoint)

    root_position = np.array([0.0, 0.0, 0.0])  # Assuming root at origin for simplicity
    recursive_plot('root', root_position)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

import os
from hit_omniverse import HIT_SIM_DATASET_DIR

# Example usage
asf_filename = os.path.join(HIT_SIM_DATASET_DIR, "allasfamc", "all_asfamc", "subjects", "01", "01.asf")
amc_filename = os.path.join(HIT_SIM_DATASET_DIR, "allasfamc", "all_asfamc", "subjects", "01", "01_01.amc")

bones = parse_asf(asf_filename)
motion_data = parse_amc(amc_filename, bones)

# Plot the first frame
plot_skeleton(bones, motion_data[0])
