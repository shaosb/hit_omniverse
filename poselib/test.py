from utils.motion_lib import MotionLib
from pprint import pprint

if __name__ == '__main__':
    motion_file = "data/retarget_npy/07_03.npy"
    # dof_body_ids = [1, 2, 3,  # Hip, Knee, Ankle
    #                       4, 5, 6,
    #                       7,  # Torso
    #                       8, 9,  # Shoulder, Elbow, Hand
    #                       10, 11]
    #
    # dof_offsets = [0, 3, 4, 5, 8, 9, 10,
    #                11,
    #                14, 15, 18, 19]

    # dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    dof_body_ids = [7, 1, 13, 8, 2, 14, 9, 3, 19, 15, 10, 4, 20, 16, 11, 5, 21, 17, 12, 6, 22, 18]
    dof_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    # dof_offsets = [n * 3 for n in dof_offsets]

    # key_bodies = ["pelvis", "torso_link"]
    # self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
    key_bodies = [0, 14] # ["pelvis", "body_link"]
    motion_lib = MotionLib(motion_file=motion_file,
                                 dof_body_ids=dof_body_ids,
                                 dof_offsets=dof_offsets,
                                 key_body_ids=key_bodies,
                                 no_keybody=True,
                                 device="cpu")
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
        = motion_lib.get_motion_state([0], 1000)
    motion_lib.serialize_motion("07_03.npy")
    # print(root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos)

    pass
