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

from sim2sim_leggedgym.envs.base.legged_robot_config import LeggedRobotCfg
import isaacgym

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from isaacgym.terrain_utils import *

import torch
import matplotlib.pyplot as plt


from sim2sim_leggedgym.envs.base.legged_robot import LeggedRobot

import os
from sim2sim_leggedgym.utils.terrain import Terrain
import numpy as np
import time
from collections import deque


class GO1FreeEnv(LeggedRobot):
    '''
    IUSTFreeEnv is a class that represents a custom environment for a legged robot.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # global sensors
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.error_linear_x = deque(maxlen=100)
        self.error_linear_y = deque(maxlen=100)
        self.error_angular_yaw = deque(maxlen=100)
        self.actions_commands_all = deque(maxlen=100)
        self.log_dir = None 
        # sensors = self._create_envs()
        self.compute_observations()

    def _push_robots(self):
        """ Random pushes the quadruped robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        # TODO using rand push vel as privileged obs
        self.rand_push_vel = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_vel
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
    def _get_phase(self):
        """ Compute the phase of the gait cycle for the quadruped based on time. """
        cycle_time = self.cfg.rewards.cycle_time  
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase
    
    def _get_gait_phase(self):
        """ Define gait phases for a quadruped robot: stance (1) or swing (0) for each leg. """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)

        # stance mask: [Front-Left, Front-Right, Rear-Left, Rear-Right]
        stance_mask = torch.zeros((self.num_envs, 4), device=self.device)

        # Define a trot gait where diagonal pairs of legs move together
        # Front-left (FL) and Rear-right (RR) stance when sin_pos >= 0
        stance_mask[:, 0] = sin_pos >= 0  # Front-left in stance
        stance_mask[:, 3] = sin_pos >= 0  # Rear-right in stance

        # Front-right (FR) and Rear-left (RL) stance when sin_pos < 0
        stance_mask[:, 1] = sin_pos < 0  # Front-right in stance
        stance_mask[:, 2] = sin_pos < 0  # Rear-left in stance

        # Optionally, add a double support phase (when all legs are on the ground)
        stance_mask[torch.abs(sin_pos) < 0.1] = 1  # When the sine position is close to zero, all legs are in stance
        
        return stance_mask
    
    def compute_ref_state(self):
        """ Compute the reference joint positions based on the current gait phase for a quadruped robot. """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)

        sin_pos_fl = sin_pos.clone()
        sin_pos_fr = sin_pos.clone()
        sin_pos_rl = sin_pos.clone()
        sin_pos_rr = sin_pos.clone()

        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = 0.17  # target_joint_pos_scale = 0.17 
        scale_2 = 2 * scale_1

        # FL stance phase
        sin_pos_fl[sin_pos_fl > 0] = 0
        self.ref_dof_pos[:, 0] = sin_pos_fl * scale_1  # FL hip
        self.ref_dof_pos[:, 1] = sin_pos_fl * scale_2  # FL knee
        self.ref_dof_pos[:, 2] = sin_pos_fl * scale_1  # FL ankle

        # RR stance phase
        sin_pos_rr[sin_pos_rr > 0] = 0
        self.ref_dof_pos[:, 9] = sin_pos_rr * scale_1  # RR hip
        self.ref_dof_pos[:, 10] = sin_pos_rr * scale_2  # RR knee
        self.ref_dof_pos[:, 11] = sin_pos_rr * scale_1  # RR ankle

        # FR stance phase
        sin_pos_fr[sin_pos_fr < 0] = 0
        self.ref_dof_pos[:, 3] = sin_pos_fr * scale_1  # FR hip
        self.ref_dof_pos[:, 4] = sin_pos_fr * scale_2  # FR knee
        self.ref_dof_pos[:, 5] = sin_pos_fr * scale_1  # FR ankle

        # RL stance phase
        sin_pos_rl[sin_pos_rl < 0] = 0
        self.ref_dof_pos[:, 6] = sin_pos_rl * scale_1  # RL hip
        self.ref_dof_pos[:, 7] = sin_pos_rl * scale_2  # RL knee
        self.ref_dof_pos[:, 8] = sin_pos_rl * scale_1  # RL ankle

        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
                self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type == 'uneven':
            self._creat_uneven_ground()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _creat_uneven_ground(self):
        
        num_terrains = 1
        terrain_width = 50.
        terrain_length = 50.
        horizontal_scale = 0.1  # [m] resolution in x
        vertical_scale = 0.005  # [m] resolution in z
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_terrains*num_rows, num_cols), dtype=np.int16)
        
        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        heightfield[0:1*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.0, step=0.05, downsampled_scale=0.3).height_field_raw

        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)

        tm_params = gymapi.TriangleMeshParams()

        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -terrain_width/3
        tm_params.transform.p.y = -terrain_length/3
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

    def _init_privilaged(self):

        ## TODO: using for privileged observation
        self.rand_push_vel = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.env_restitution = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        ##

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(48)
        # print("######################################3", np.shape(noise_vec))
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[50:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        
        return noise_vec
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                num_buckets_fri = self.cfg.domain_rand.num_buckets_friction
                bucket_ids = torch.randint(0, num_buckets_fri, (self.num_envs, 1))

                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets_fri,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            
            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                num_buckets_res = self.cfg.domain_rand.num_buckets_restitution
                bucket_ids = torch.randint(0, num_buckets_res, (self.num_envs, 1))

                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets_res,1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
            
            
            self.env_restitution[env_id] = self.restitution_coeffs[env_id]
            
        return props
    
    def _process_rigid_body_props(self, props, env_id):
    
        # TODO This section was modified because some indices changed when collapse_fixed_joints was set to False      
        if self.cfg.domain_rand.randomize_base_mass:
            mass_range = self.cfg.domain_rand.added_mass_range
            self.payloads[env_id, 0] = np.random.uniform(mass_range[0], mass_range[1])
            props[1].mass += self.payloads[env_id, 0] # TODO the default index was zero: props[0]
        
        if self.cfg.domain_rand.randomize_com_displacement:
            com_range = self.cfg.domain_rand.com_displacement_range
            self.com_displacements[env_id, :] = torch.rand(1, 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (com_range[1] - com_range[0]) + com_range[0]
            props[1].com += gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1], 
                                        self.com_displacements[env_id, 2])
                                          
        return props


    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        # delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = actions.to(device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        # print("#####################################")
        return super().step(actions)
    
    def compute_states(self):
        
        self.contact_foot_force = self.contact_forces[:, self.feet_indices, :]
        self.contact_foot_z_force = self.contact_forces[:, self.feet_indices, 2]
        self.contact_state = (torch.norm(self.contact_foot_force, dim=-1) > 1.0).int()


        self.contact_thigh_force = self.contact_forces[:, self.thigh_contact_indices, :]
        self.thigh_state = (torch.norm(self.contact_thigh_force, dim=-1) > 1.0).int()

        self.contact_calf_force = self.contact_forces[:, self.calf_contact_indices, :]
        self.calf_state = (torch.norm(self.contact_calf_force, dim=-1) > 1.0).int()



    def compute_observations(self):
        """ Compute observations for the quadruped robot. """
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1) #(sin, cos) + 3 (lin_vel_x, lin_vel_y, yaw_vel)
        
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        
        # TODO
        self.compute_states()

        # critic obs
        self.privileged_obs_buf = torch.cat(( self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                              self.base_ang_vel * self.obs_scales.ang_vel, # 3
                                              self.projected_gravity, # 3
                                              self.commands[:, :3] * self.commands_scale, # 3
                                              (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                                              self.dof_vel * self.obs_scales.dof_vel, # 12
                                              self.actions, # 12
                                              self.env_frictions, # 1
                                              self.env_restitution, # 1
                                              self.payloads, # 1
                                              self.rand_push_vel, # 2
                                              self.com_displacements, # 3
                                              self.contact_foot_z_force, #4
                                              self.contact_state, # 4
                                              self.thigh_state, # 4
                                              self.calf_state  # 4          
                                              ),dim=-1) # 72
         
        # print("########################################## friction", self.env_frictions)
        # print("########################################## restitution", self.env_restitution)    

        # print("privileged obs ", np.shape(self.privileged_obs_buf))
        
        # print("################################## push vel ", (self.rand_push_vel))
        # print("################################## env frictions  ", (self.env_frictions))
        # print("################################## env restitutions  ", (self.env_restitution))
        # print("################################## env playloads ", (self.payloads))
        # print("################################## env com sdisplacements ", (self.com_displacements))


        obs_buf = torch.cat((   self.base_lin_vel * self.obs_scales.lin_vel,
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,
                                self.commands[:, :3] * self.commands_scale,
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions
                                ),dim=-1) # 48
        
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        if self.add_noise:  
            # self.noise_scale_vec = self.noise_scale_vec.to('cuda:0')  # Move to GPU if not already
            obs_buf = obs_buf + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            

        self.obs_history.append(obs_buf.clone())
        # print('obs history:',np.shape(self.obs_history))
        self.critic_history.append(self.privileged_obs_buf.clone())
        # print('critic obs history:',np.shape(self.critic_history))

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

        ##### log errors for tracking
        linear_vel_x_error = abs(self.commands[:,0]) - abs(self.base_lin_vel[:, 0])
        linear_vel_y_error = abs(self.commands[:,1]) - abs(self.base_lin_vel[:, 1])
        angular_vel_yaw_error = abs(self.commands[:,2]) - abs(self.base_ang_vel[:, 2])

        self.error_linear_x.append((abs(linear_vel_x_error).sum())/self.num_envs)
        self.error_linear_y.append((abs(linear_vel_y_error).sum())/self.num_envs)
        self.error_angular_yaw.append((abs(angular_vel_yaw_error).sum())/self.num_envs)
        # self.actions_commands_all.append(self.actions_commands_vx[-1] + self.actions_commands_vy[-1] + 10 * self.actions_commands_yaw[-1])
        len = self.error_linear_x.__len__()
        
        if os.path.exists("LOG_DIR.txt"):
            with open("LOG_DIR.txt", "r") as file:
                self.log_dir = file.read().strip()

        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "error.txt"), "a") as f:
                f.write(f"{(sum(self.error_linear_x))/len}  {(sum(self.error_linear_y))/len}  {(sum(self.error_angular_yaw))/len} \n")
        #####

    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        #print('reset_id from iust_env function:',env_ids)

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards ================================================== #

    def _reward_lin_vel_z(self):
            # Penalize z axis base linear velocity
            return torch.square(self.base_lin_vel[:, 2])
        
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # print("self contact", contact)
        # print('self last contact ', self.last_contacts)
        # self contact forces  tensor([[ 0.0000,  0.0000,  0.0000, 34.8771]], device='cuda:0')
        # contact  tensor([[False, False, False,  True]], device='cuda:0')
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        # print('contact filt', contact_filt)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        # print('feet air time', first_contact)
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
            5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
