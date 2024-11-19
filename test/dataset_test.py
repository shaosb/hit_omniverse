# from hit_omniverse.extension.mdp.command import dataset
import torch
from hit_omniverse import HIT_SIM_DATASET_DIR
import os
from torch.utils.data import Dataset, DataLoader

def read_npy(file_path):
    from poselib.skeleton.skeleton3d import SkeletonMotion

    motion = SkeletonMotion.from_file(file_path)
    node_names = motion.skeleton_tree.node_names

    length = len(motion.global_transformation.numpy()[:, 0, :])

    data = {name: motion.global_transformation.numpy()[:, i, :] for i, name in enumerate(node_names)}
    time = [i for i in range(length)]
    data.update({"time": time})

    return data


class BaseDataset(Dataset):
    def __init__(self, file: str):
        self.file = os.path.join(HIT_SIM_DATASET_DIR, file)
        self.data = read_npy(self.file)

    def __getitem__(self, index):
        item = {}
        for key in self.data.keys():
            item.update({key: torch.tensor(self.data.get(key)[index])})
        return item

    def __len__(self):
        return len(self.data.get("time"))


if __name__ == '__main__':
    t = BaseDataset("motion_retarget\\01_01.npy")
    print(t[3])
    pass