# from sim2sim_leggedgym import LEGGED_GYM_ROOT_DIR, envs
# from time import time
# from warnings import WarningMessage
# import numpy as np
# import os

from isaacgym.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import numpy as np
import torch
# from typing import Tuple, Dict
# from collections import deque

# from sim2sim_leggedgym import LEGGED_GYM_ROOT_DIR
# from sim2sim_leggedgym.envs.base.base_task import BaseTask
# from sim2sim_leggedgym.utils.terrain import Terrain
# from sim2sim_leggedgym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
# from sim2sim_leggedgym.utils.helpers import class_to_dict
# from .legged_robot_config import LeggedRobotCfg


gravity_vec = np.array([[0, 0, -1]])
print("#################################333", (gravity_vec).dtype)
base_quat = np.array([[0, 0 , 0, 1]])

projected_gravity = quat_rotate_inverse(torch.from_numpy(base_quat), torch.from_numpy(gravity_vec))