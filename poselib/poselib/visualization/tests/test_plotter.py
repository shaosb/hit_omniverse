# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from poselib.visualization.core import BasePlotterTask, BasePlotterTasks
from poselib.visualization.plt_plotter import Matplotlib3DPlotter
from poselib.visualization.simple_plotter_tasks import Draw3DDots, Draw3DLines

task = Draw3DLines(task_name="test", 
    lines=np.array([[[0, 0, 0], [0, 0, 1]], [[0, 1, 1], [0, 1, 0]]]), color="blue")
task2 = Draw3DDots(task_name="test2", 
    dots=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]), color="red")
task3 = BasePlotterTasks([task, task2])
plotter = Matplotlib3DPlotter(cast(BasePlotterTask, task3))
plt.show()
