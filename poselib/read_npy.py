from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import numpy as np

if __name__ == "__main__":
    # npy_path= "data/retarget_npy/02_01.npy"
    # motion = SkeletonMotion.from_file(npy_path)

    npy_path = "output.npy"
    # motion = np.load(npy_path, allow_pickle=True)
    motion = SkeletonMotion.from_file(npy_path)
    print(motion.skeleton_tree.node_names)
    temp0 = motion.global_rotation.numpy()[0,:,:]
    temp1 = motion.global_translation.numpy()[0,:,:]
    temp2 = motion.global_transformation.numpy()[0,:,:]
    temp4 = motion.skeleton_tree.node_names
    temp5 = len(motion.global_transformation.numpy()[:,0,:])
    temp6 = motion.global_transformation.numpy()[:,0,:]
    length = len(motion.global_transformation.numpy()[:, 0, :])
    time = [i for i in range(length)]
    plot_skeleton_motion_interactive(motion)