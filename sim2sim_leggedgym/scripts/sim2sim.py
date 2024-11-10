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
#!/usr/bin/env python
import os
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from envs.iust.iust_config import IUSTsim2simCfg
import torch
from algo.ppo.actor_critic import ActorCritic
from algo.ppo.ppo import PPO
import mujoco_py
import copy
from mujoco_py import load_model_from_xml, MjSim, MjViewer
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import time

i = 0
class cmd:
    vx = 0.0 # 0.4
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

@property
def projected_gravity(self):
    w, x, y, z = self.data.qpos[3:7]
    euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z))
    projected_gravity_not_normalized = (
        np.dot(self._gravity_vector, euler_orientation) * euler_orientation
    )
    if np.linalg.norm(projected_gravity_not_normalized) == 0:
        return projected_gravity_not_normalized
    else:
        return projected_gravity_not_normalized / np.linalg.norm(
            projected_gravity_not_normalized
        )

'''
def get_obs(self): #rl_mujoco
        dofs_position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]
        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.projected_gravity

        # curr_obs = np.concatenate(
        #     (
        #         base_linear_velocity * self._obs_scale["linear_velocity"],
        #         base_angular_velocity * self._obs_scale["angular_velocity"],
        #         projected_gravity,
        #         desired_vel * self._obs_scale["linear_velocity"], #command
        #         dofs_position * self._obs_scale["dofs_position"],
        #         dofs_velocity * self._obs_scale["dofs_velocity"],
        #         last_action,
        #     )
        # ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        return curr_obs
        '''



def pd_control(target_q, q, kp, dq, kd):
    '''Calculates torques from position commands
       target_q = actions_scaled
       torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

       default : return (target_q - q) * kp + (target_dq - dq) * kd
    '''
    global default_dof_pos

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
    default_dof_pos = np.array([[ 0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000,
         -1.5000, -0.1000,  1.0000, -1.5000]])
    

    model = mujoco.MjModel.from_xml_path("/home/lenovo/sim_leggedgym/resources/mjcf/quad.xml")
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    # p_gains = torch.tensor([20., 22., 25., 20., 22., 25., 20., 25., 25., 20., 25., 25.])
    # d_gains = 0.5
    # # set default joint
    # default_dof_pos = torch.zeros(12, dtype=torch.float, device='cpu', requires_grad=False)
    # dof_names = ['RR_hip_joint','RR_thigh_joint','RR_calf_joint', 'RL_hip_joint','RL_thigh_joint', 'RL_calf_joint','FL_hip_joint','FL_thigh_joint','FL_calf_joint','FR_hip_joint','FR_thigh_joint', 'FR_calf_joint']
    # for i in range(12):
    #     name = dof_names[i]
    #     angle = cfg.init_state.default_joint_angles[name]
    #     default_dof_pos[i] = angle
    #     for dof_name in cfg.control.stiffness.keys():
    #         if dof_name in name:
    #             print(dof_name)
    #             p_gains[i] = cfg.control.stiffness[dof_name]
    #             d_gains[i] = cfg.control.damping[dof_name]


    count_lowlevel = 0
    q_isaac = np.zeros(12)
    dq_isaac = np.zeros(12)
    target_q_isaac = np.zeros(12)
    max_episode_length = np.ceil(20 / 4 * 0.005)
    for _ in range(1000000000*int(max_episode_length)):

    # for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
    
        # Obtain an observation
        q, dq, quat, v, omega, gvec, linvelocity = get_obs(data)
        
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



        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            # eu_ang = quaternion_to_euler_array(quat)
            # eu_ang[eu_ang > math.pi] -= 2 * math.pi
            obs[0, 0:3] = v * cfg.normalization.obs_scales.lin_vel # base lin vel (*2.0)
            obs[0, 3:6] = omega * cfg.normalization.obs_scales.ang_vel # base ang vel (*0.25)
            obs[0, 6:9] = gvec # gravity
            obs[0, 9:10] = cmd.vx * 2.
            obs[0, 10:11] = cmd.vy * 2.
            obs[0, 11:12] = cmd.dyaw * 0.25
            obs[0, 12:24] = (np.array(q_isaac) - default_dof_pos) * cfg.normalization.obs_scales.dof_pos # (*1.)
            obs[0, 24:36] = np.array(dq_isaac)* cfg.normalization.obs_scales.dof_vel # (*0.05)
            obs[0, 36:48] = action
            
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            print('##########################################################################')
            print('obseravtions =',obs)
            hist_obs.append(obs)
            hist_obs.popleft()
            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32) # 48
            for i in range(cfg.env.frame_stack): # 15
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            policy_input_tensor = torch.tensor(policy_input, dtype=torch.float32).to('cpu')
            if policy_input_tensor.shape[1] != 720:
                policy_input_tensor = policy_input_tensor[:, :720]  # Example to take only the first 720 features
            action[:] = policy(policy_input_tensor).detach().numpy() # position

                
            # policy_input_tensor = torch.tensor(policy_input)
            # policy_input_tensor = policy_input_tensor.unsqueeze(0)
            # print(np.shape(policy_input_tensor))
            # action[:] = policy(policy_input_tensor)[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
        target_q_isaac = action * cfg.control.action_scale 

        target_q_isaac [3:6] = target_q [:3]
        target_q_isaac [:3] = target_q [3:6]
        target_q_isaac [9:12] = target_q [6:9]
        target_q_isaac [6:9] = target_q [9:12]


        # target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit) # Clamp torques

        print('torques =',tau)
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
                mujoco_model_path = "/home/lenovo/sim_leggedgym/resources/robots/XBot/mjcf/XBot-L-terrain.xml"
            else:
                mujoco_model_path = "/home/lenovo/sim_leggedgym/resources/mjcf/quad.xml"
            sim_dursation = 80 #60
            dt = 0.001 #0.001
            decimation = 4 #10
            max_episode_length_s = 20

        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            # tau_limit = 10. * np.ones(12, dtype=np.double) #200
            tau_limit = np.array([20., 55., 55., 20., 55., 55., 20., 55., 55., 20., 55., 55.])
            # tau_limit = np.array([8,8,8,8,8,8,8,8,8,8,8,8])
            # [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

    type_load = 'load_jit'
    path = "/home/lenovo/sim_leggedgym/sim2sim_leggedgym/logs/rough_iust/"
    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export_policy_as_jit(actor_critic, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(path, "Nov6-48-48-highrewardjit.pt")
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
        actor_critic = ActorCritic(720, 144, 12,[512,256,128],[512,256,128])
        export_policy_as_jit(actor_critic, "/home/lenovo/sim_leggedgym/sim2sim_leggedgym/logs/rough_iust/")
        # policy_net_file = "{LEGGED_GYM_ROOT_DIR}/sim2sim_leggedgym/logs/rough_iust/Nov6-48-48-highreward.pt"
        # policy_network_path = policy_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        # print(policy_network_path)
        # print('-----------------------------')
        policy = torch.jit.load("/home/lenovo/sim_leggedgym/sim2sim_leggedgym/logs/rough_iust/Nov6-48-48-highrewardjit.pt")

    print('policy loaded!...')
    run_mujoco(policy, Sim2simCfg())
    



