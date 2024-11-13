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
    vx = 1
    vy = 0.0
    dyaw = 0.0

def quat_rotate_inverse(quat, vec):
    shape = quat.shape
    q_w = quat[:, -1]
    q_vec = quat[:, :3]
    a = vec * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, vec, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), vec.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c



def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos[7:].astype(np.double)
    dq = data.qvel[6:].astype(np.double)


    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double) # default of mujoco repo
    quat_mujoco = data.qpos[3:7].astype(np.double) # Gives [q_w, q_x, q_y, q_z]
    quat_standard = quat_mujoco[[1, 2, 3, 0]].reshape(1, -1).astype(np.double) # Gives [q_x, q_y, q_z, q_w]
    print("####################################### from qpos", quat_standard)
    
    gravity_vec = np.array([[0, 0, -1]]).astype(np.double)
    # print("####################################### from sensors", gravity_vec.dtype)

    projected_gravity = quat_rotate_inverse(torch.from_numpy(quat_standard), torch.from_numpy(gravity_vec))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ from sensors", projected_gravity)

    base_lin_vel = data.qvel[:3].astype(np.double)
    base_ang_vel = data.qvel[3:6].astype(np.double)
    return (q, dq, quat_standard, base_lin_vel, base_ang_vel, projected_gravity)



def pd_control(target_q, q, kp, dq, kd):
    '''Calculates torques from position commands
    '''
    # return (target_q - q) * kp + (target_dq - dq) * kd

    # mujoco ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint',
    #         'FL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 
    #         'RL_thigh_joint', 'RL_calf_joint']

    # isaac gym ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint',
    #            'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 
    #            'RR_thigh_joint', 'RR_calf_joint']

    default_dof_pos = np.array([[ -0.1000,  0.8000, -1.5000, 0.1000,  0.8000, -1.5000,  -0.1000,  1.0000,
                                   -1.5000, 0.1000,  1.0000, -1.5000]])
    return kp * (target_q - q + default_dof_pos ) - kd * (dq) # checking for target q


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    default_dof_pos_isaac = np.array([[ 0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000,
         -1.5000, -0.1000,  1.0000, -1.5000]])
    
    ### des
    qdes = np.array([[ -0.1000,  0.8000, -1.5000, 0.1000,  0.8000, -1.5000,  -0.1000,  1.0000,
                                   -1.5000, 0.1000,  1.0000, -1.5000]])
    pdes = np.array([0.0, 0.0, 0.32])
    rotdes = np.array([0.0, 0.0, 0.0, 1.0]) # x y z w

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    # print("##################################################",help(data))
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    data.qpos[7:19] = qdes
    data.qpos[0:3] = pdes  
    data.qpos[3:7] = rotdes[[3, 0, 1, 2]] # w x y z 

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    

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
        q, dq, quat, base_lin_vel, base_ang_vel, projected_gravity = get_obs(data)    
        # print("################################################## base linear velocity:",base_lin_vel)
        # print("################################################## base angular velocity:",base_ang_vel)
        # print("################################################## projected gravity:",projected_gravity)
        
        q_isaac[: 3] = q [3: 6] # FL
        q_isaac[3: 6] = q [: 3] # FR
        q_isaac [6: 9] = q [9: 12] # RL
        q_isaac [9: 12] = q [6: 9] # RR
        
        
        dq_isaac [: 3] = dq [3: 6]
        dq_isaac [3: 6] = dq [: 3]
        dq_isaac [6: 9] = dq [9: 12]
        dq_isaac [9: 12] = dq [6: 9]
        
        # print("#########$$$$$$$$$$$$$$$", dq[18])
        
        ####
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
           
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
           
            obs[0, 0:3] = base_lin_vel * cfg.normalization.obs_scales.lin_vel 
            obs[0, 3:6] = base_ang_vel * cfg.normalization.obs_scales.ang_vel 
            obs[0, 6:9] = projected_gravity
            obs[0, 9:10] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 10:11] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 11:12] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 12:24] = (np.array(q_isaac) - default_dof_pos_isaac) * cfg.normalization.obs_scales.dof_pos 
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


        target_q [:3] = target_q_isaac [3:6]
        target_q [3:6] = target_q_isaac [:3]
        target_q [6:9] = target_q_isaac [9:12]
        target_q [9:12] = target_q_isaac [6:9]

        # target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                    dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        print("##############################################################", tau)
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
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/xml/go1.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/xml/go1.xml'
            sim_duration = 60.0
            dt = 0.005
            decimation = 4
        
        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)
    
    type_load = 'load_jit'
    path = "/home/mehtimans/sim2sim_leggedgym/logs/go1/Nov11_08-58-40_/model_best.pt"

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export_policy_as_jit(actor_critic, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(path, "model_3000_jit.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)
        print(f"Model saved as jit: {path}")


    if type_load == 'load_nn':
        print('path:',path)
        loaded_dict = torch.load(path)
        actor_critic = ActorCritic(720, 144, 12,[512,256,128],[512,256,128])
        # policy = PPO(actor_critic = actor_critic)
        actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        actor_critic.eval()
        actor_critic.to('cpu')
        policy = actor_critic.act_inference

    elif type_load == 'load_jit':
        # actor_critic = ActorCritic(720, 144, 12,[512,256,128],[512,256,128])
        # export_policy_as_jit(actor_critic, "/home/mehtimans/sim2sim_leggedgym/logs/go1/Nov11_08-58-40_/")
        # policy_net_file = "{LEGGED_GYM_ROOT_DIR}/sim2sim_leggedgym/logs/rough_iust/Nov6-48-48-highreward.pt"
        # policy_network_path = policy_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        # print(policy_network_path)
        # print('-----------------------------')
        policy = torch.jit.load(path)

    print('policy loaded!...')
    run_mujoco(policy, Sim2simCfg())