import torch
from torch.utils.data.dataloader import DataLoader
import dataset
from dataclasses import MISSING
import os
from hit_omniverse import HIT_SIM_DATASET_DIR

from pprint import pprint


train_dataset_paths = [os.path.join(HIT_SIM_DATASET_DIR, "CMU_001_01.hdf5")]
dataset_metrics_path = os.path.join(HIT_SIM_DATASET_DIR, "dataset_metrics.npz")
clip_ids = None

MULTI_CLIP_OBSERVABLES_SANS_ID = (
    'walker/actuator_activation',
    'walker/appendages_pos',
    'walker/body_height',
    'walker/end_effectors_pos',
    'walker/gyro_control',
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/joints_vel_control',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/time_in_clip',
    'walker/velocimeter_control',
    'walker/world_zaxis',
    'walker/reference_rel_joints',
    'walker/reference_rel_bodies_pos_global',
    'walker/reference_rel_bodies_quats',
    'walker/reference_rel_bodies_pos_local',
    'walker/reference_ego_bodies_quats',
    'walker/reference_rel_root_quat',
    'walker/reference_rel_root_pos_local',
    'walker/reference_appendages_pos',
)


def get_action(train_dataset_paths=train_dataset_paths,
               MULTI_CLIP_OBSERVABLES_SANS_ID=MULTI_CLIP_OBSERVABLES_SANS_ID,
               dataset_metrics_path=dataset_metrics_path,
               clip_ids=None):
    train_dataset = dataset.ExpertDataset(
        train_dataset_paths,
        MULTI_CLIP_OBSERVABLES_SANS_ID,
        dataset_metrics_path,
        clip_ids,
        min_seq_steps=1,
        max_seq_steps=1,
        n_start_rollouts=-1,
        n_rsi_rollouts=-1,
        normalize_obs=False,
        clip_len_upsampling=False,
        clip_weighted=False,
        advantage_weights=True,
        temperature=None,
        concat_observables=False,
        keep_hdf5s_open=False,
    )
    train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                              batch_size=2, num_workers=0)
    return train_loader


if __name__ == '__main__':


    # train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
    #                           batch_size=64, num_workers=8)

    train_loader = get_action(train_dataset_paths, MULTI_CLIP_OBSERVABLES_SANS_ID, dataset_metrics_path, clip_ids)

    # temp = iter(train_loader)
    # for _ in range(10):
    #     print(next(temp)[0]["walker/joints_pos"])

    for batch in train_loader:
        pprint(batch)
        pprint(batch[0]["walker/joints_pos"])

        pass