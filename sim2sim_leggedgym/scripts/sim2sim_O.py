# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from sim2sim_leggedgym import LEGGED_GYM_ROOT_DIR
from sim2sim_leggedgym.envs import IUSTsim2simCfg
from sim2sim_leggedgym.algo.ppo.actor_critic import ActorCritic
from sim2sim_leggedgym.algo.ppo import PPO
from sim2sim_leggedgym.algo import OnPolicyRunner
from sim2sim_leggedgym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from sim2sim_leggedgym.algo.vec_env import VecEnv
import torch
import os


class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    linvelocity = data.sensor('linear-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec, linvelocity)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    # return (target_q - q) * kp + (target_dq - dq) * kd
    default_dof_pos = np.array([[ 0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000,
                                   -1.5000, -0.1000,  1.0000, -1.5000]])
    return kp * (target_q - q + default_dof_pos ) - kd * (dq)

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    # print("##################################################",help(data))
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    
    #### from legged gym config
    default_dof_pos = np.array([[ 0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000,
                                   -1.5000, -0.1000,  1.0000, -1.5000]])


    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
        # print("##########################################", hist_obs, cfg.env.num_single_obs)
    
    

    count_lowlevel = 0
    q_isaac = np.zeros(12)
    dq_isaac = np.zeros(12)
    target_q_isaac = np.zeros(12)
    max_episode_length = np.ceil(20 / 4 * 0.005)
    for _ in range(1000000000*int(max_episode_length)):

    #for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        
        # Obtain an observation
        q, dq, quat, v, omega, gvec, linvelocity = get_obs(data)    
        #print("##################################################",len(q))
        q [10:13] = np.array(q_isaac[:3]) #FL
        q [7:10] = q_isaac[3:6] #FR
        q [16:19] = q_isaac [6:9] #RL
        q [13:16] = q_isaac [9:12] #RR
        q = q[7:19]

        
        dq [9:12] = dq_isaac [:3]
        dq [6:9] = dq_isaac [3:6]
        dq [15:18] = dq_isaac [6:9]
        dq [12:15] = dq_isaac [9:12]
        
        dq = dq[6:18]
        # print("#########$$$$$$$$$$$$$$$", dq[18])
        
        ####
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
           
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
           
            obs[0, 0:3] = v * cfg.normalization.obs_scales.lin_vel 
            obs[0, 3:6] = omega * cfg.normalization.obs_scales.ang_vel 

            obs[0, 6:9] = gvec

            obs[0, 9:10] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 10:11] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 11:12] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel

            obs[0, 12:24] = (np.array(q_isaac) - default_dof_pos) * cfg.normalization.obs_scales.dof_pos 
            obs[0, 24:36] = np.array(dq_isaac) * cfg.normalization.obs_scales.dof_vel 

            obs[0, 36:48] = action

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()
          
            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            target_q_isaac = action * cfg.control.action_scale

        # target_q_isaac [3:6] = target_q [:3]
        # target_q_isaac [:3] = target_q [3:6]
        # target_q_isaac [9:12] = target_q [6:9]
        # target_q_isaac [6:9] = target_q [9:12]
        target_q [:3] = target_q_isaac [3:6]
        target_q [3:6] = target_q_isaac [:3]
        target_q [6:9] = target_q_isaac [9:12]
        target_q [9:12] = target_q_isaac [6:9]

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                    target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    # parser.add_argument('--load_model', type=str, required=True,
    #                     help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(IUSTsim2simCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/iust/mjcf/quad.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/iust/mjcf/quad.xml'
            sim_duration = 60.0
            dt = 0.00001
            decimation = 10
        
        class robot_config:
            kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)
    type_load = 'load_nn'
    path = "/home/mehtimans/sim2sim_leggedgym/logs/rough_iust/Nov04_18-38-25_/model_1500.pt"
    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export_policy_as_jit(actor_critic, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(path, "model_1500.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)
        print(f"Model saved as jit: {path}")
    

    if type_load == 'load_nn':
        loaded_dict = torch.load(path)
        actor_critic = ActorCritic(720, 144, 12,[512,256,128],[512,256,128])
        actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        actor_critic.eval()
        actor_critic.to('cpu')
        policy = actor_critic.act_inference

    elif type_load == 'load_jit':
        actor_critic = ActorCritic(720, 144, 12,[512,256,128],[512,256,128])
        export_policy_as_jit(actor_critic, path)
        policy_net_file = "{LEGGED_GYM_ROOT_DIR}/logs/rough_iust/Nov04_18-38-25_/model_1500.pt"
        policy_network_path = policy_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        policy = torch.jit.load(policy_network_path)

    print('policy loaded!...')
    run_mujoco(policy, Sim2simCfg())
