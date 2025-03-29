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

from sim2sim_leggedgym import LEGGED_GYM_ROOT_DIR, envs
from sim2sim_leggedgym.envs.base.legged_robot_config import LeggedRobotCfg
import isaacgym

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from isaacgym.terrain_utils import *

import torch
import matplotlib.pyplot as plt
from sim2sim_leggedgym.utils.math import quat_apply_yaw, wrap_to_pi

from sim2sim_leggedgym.envs.base.legged_robot import LeggedRobot
from sim2sim_leggedgym.utils.helpers import class_to_dict
from sim2sim_leggedgym.envs.base.curriculum import RewardThresholdCurriculum


import os
from sim2sim_leggedgym.utils.terrain import Terrain
import numpy as np
import time
from collections import deque



class GO1FreeEnv(LeggedRobot):
    '''
    Go1FreeEnv is a class that represents a custom environment for a legged robot.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # global sensors
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.error_linear_x = deque(maxlen=10)
        self.error_linear_y = deque(maxlen=10)
        self.error_angular_yaw = deque(maxlen=10)
        self.log_dir = None 
        self.compute_observations()

    def _push_robots(self):
        """ Random pushes the quadruped robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        # TODO using rand push vel as privileged obs
        self.rand_push_vel = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_vel
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        
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
        terrain_width = 200.
        terrain_length = 200.
        horizontal_scale = 0.1  # [m] resolution in x
        vertical_scale = 0.01  # [m] resolution in z
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_terrains*num_rows, num_cols), dtype=np.int16)
        
        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        heightfield[0:1*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.05, max_height=0.05, step=0.05, downsampled_scale=0.3).height_field_raw

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

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) #### gait
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # (0-3 position base, 3-7 quat base, 7-10 lin vel, 10-13 ang vel)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state) ### gait (num_envs * num_bodies, 13) , 13 = [px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, 0:3] #### gait
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = self.get_euler_xyz_tensor(self.base_quat)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10] #### gait it becomes (num_envs, num_feet, xyz axis)
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3] #### gait it becomes (num_envs, num_feet, xyz axis)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) ### gate
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) ### gate
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) ### gate
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) ### gate
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        
        ### gait
        # self._init_command_distribution(torch.arange(self.num_envs, device=self.device))
        ### gait

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))
        self.lag_buffer = deque(maxlen=self.cfg.domain_rand.lag_timesteps)
        
        ### gait
        for _ in range(self.cfg.domain_rand.lag_timesteps):
            self.lag_buffer.append(torch.zeros(
                self.num_envs, self.num_dof, dtype=torch.float, device=self.device))
        ### gait 
        self.phases_info = torch.zeros(self.num_envs, 11) # Frequency, phase, offset, bounds, duration  ### gait change hard code
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) ### gait change hard code
        ### gait
        # self.prev_foot_velocities = self.foot_velocities.clone() check!!
        ### gait


    def _init_privilaged(self):

        ## TODO: using for privileged observation
        self.rand_push_vel = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.env_restitution = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_damping =  torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_friction =  torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        ##
    
    #### gate
    def _init_custom_buffer(self):
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
    #### gate
        
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, dtype=torch.float, device=self.device, requires_grad=False)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level 
        noise_vec[3:6] = noise_scales.ang_vel * noise_level 
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = noise_scales.commands * noise_level # 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level 
        noise_vec[24:36] = noise_scales.dof_vel * noise_level 
        noise_vec[36:48] = noise_scales.actions * noise_level # 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[50:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        
        return noise_vec
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        if self.cfg.domain_rand.randomize_joint_damping:
            joint_damping_range = self.cfg.domain_rand.joint_damping_range
            self.joint_damping[env_id, 0] = np.random.uniform(joint_damping_range[0], joint_damping_range[1])  
            for i in range(len(props)): # len (props) = 12
                if self.joint_type[i] == gymapi.JOINT_REVOLUTE:
                    props['damping'][i] = self.joint_damping[env_id, 0]

        if self.cfg.domain_rand.randomize_joint_friction:
            joint_friction_range = self.cfg.domain_rand.joint_friction_range
            self.joint_friction[env_id, 0] = np.random.uniform(joint_friction_range[0], joint_friction_range[1])  
            for i in range(len(props)):
                if self.joint_type[i] == gymapi.JOINT_REVOLUTE:
                    props['friction'][i] = self.joint_friction[env_id, 0]  

        return props
    
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

    def pre_physics_step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = actions.to(device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        
        ### gait
        # self.prev_foot_velocities = self.foot_velocities.clone()
        ### gait

        return actions

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self.pre_physics_step(actions)

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render() 

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim) # This refresh should be inside the decimation loop because the compute_torques method relies on updated values.
        self.post_physics_step() # compute obs, rew, reset...

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards 
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3] ### gate
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = self.get_euler_xyz_tensor(self.base_quat)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10] #### gait
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3] #### gait

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids) # reset env***
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        

        self.last_last_actions[:] = self.last_actions[:] ### gate 
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:] #### gait
        self.last_joint_pos_target[:] = self.joint_pos_target[:] #### gait

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        ### gait
        # self._step_contact_targets()
        ### gait
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()



    def compute_contact_states(self):
        
        self.contact_foot_force = self.contact_forces[:, self.feet_indices, :]
        self.contact_foot_z_force = self.contact_forces[:, self.feet_indices, 2]
        self.contact_state = (torch.norm(self.contact_foot_force, dim=-1) > 1.0).int()


        self.contact_thigh_force = self.contact_forces[:, self.thigh_contact_indices, :]
        self.thigh_contact_state = (torch.norm(self.contact_thigh_force, dim=-1) > 1.0).int()

        self.contact_calf_force = self.contact_forces[:, self.calf_contact_indices, :]
        self.calf_contact_state = (torch.norm(self.contact_calf_force, dim=-1) > 1.0).int()



    def compute_observations(self):
        """ Compute observations for the quadruped robot. """

        
        # TODO
        self.compute_contact_states()

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
                                              #   self.joint_friction, # 1
                                              #   self.joint_damping, # 1
                                              self.contact_foot_z_force, #4
                                              self.contact_state, # 4
                                              self.thigh_contact_state, # 4
                                              self.calf_contact_state  # 4
                                              ),dim=-1) # 74
         
        # print("########################################## friction", self.joint_friction)
        # print("########################################## damping", self.joint_damping)    

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
            noise = ((torch.rand_like(obs_buf)-0.5) * 2) * self.noise_scale_vec * obs_buf.abs()
            obs_buf += noise
            

        self.obs_history.append(obs_buf.clone())
        self.critic_history.append(self.privileged_obs_buf.clone())

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

        len = self.error_linear_x.__len__()
        
        if os.path.exists("LOG_DIR.txt"):
            with open("LOG_DIR.txt", "r") as file:
                self.log_dir = file.read().strip()

        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "error.txt"), "a") as f:
                f.write(f"{(sum(self.error_linear_x))/len}  {(sum(self.error_linear_y))/len}  {(sum(self.error_angular_yaw))/len} \n")
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        ### gate
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        ### gate
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
            ### gate
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            ### gate

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        ### gate
        elif self.cfg.rewards.positive_rew_exp_negative_rew: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_negative)
        ### gate
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew ### gate
        
        ### gate
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1
        ### gate

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
        
        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                            for name in self.reward_scales.keys()}
        
        ### gate
        # command episode sums # check this !!! if does'nt necessary remove it
        self.command_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                            for name in list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", 
                                                                           "ang_vel_residual", "ep_timesteps"]}
        ### gate
        
    def _compute_torques(self, actions):
            """ Compute torques from actions.
                Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
                [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

            Args:
                actions (torch.Tensor): Actions

            Returns:
                [torch.Tensor]: Torques sent to the simulation
            """
            #pd controller
            actions_scaled = actions * self.cfg.control.action_scale

            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_buffer.append(actions_scaled.clone())
                self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
            else:
                self.joint_pos_target = actions_scaled + self.default_dof_pos

            control_type = self.cfg.control.control_type
            if control_type=="P":
                torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                raise NameError(f"Unknown controller type: {control_type}")
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
    

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)


        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset) # body names len = 23
        # body names: ['base', 'trunk', 'FL_hip', 'FL_thigh_shoulder', 'FL_thigh', 'FL_calf', 'FL_foot', 'FR_hip', 
        # 'FR_thigh_shoulder', 'FR_thigh', 'FR_calf', 'FR_foot', 'RL_hip', 'RL_thigh_shoulder', 
        # 'RL_thigh', 'RL_calf', 'RL_foot', 'RR_hip', 'RR_thigh_shoulder', 'RR_thigh', 'RR_calf', 'RR_foot', 'imu_link']
        self.dof_names = self.gym.get_asset_dof_names(robot_asset) 
        # print("############################################################################", self.dof_names)
        # dof names:['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 
        # 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        
        self.num_bodies = len(body_names) 
        self.num_dofs = len(self.dof_names) 
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # feet names: ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        # ['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        termination_contact_names = [] # ['base']
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        
        thigh_contact_names = ['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh']
        calf_contact_names = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        
        base_link_name = "base"  # Adjust according to the URDF link name
        trunk_link_name = "trunk"  # Adjust according to the URDF link name
        
        # Get indices of specific links based on the URDF definitions
        self.base_link_index = body_names.index("base") if "base" in body_names else None
        self.trunk_link_index = body_names.index("trunk") if "trunk" in body_names else None

        # Raise an error if any of these links are not found
        if self.base_link_index is None or self.trunk_link_index is None:
            raise ValueError("Base or Trunk link not found in the body names list.")

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        # TODO: initialization for new privileged i added
        self._init_privilaged()
        
        ### gait
        self._init_custom_buffer()
        ### gait
        
        self.joint_type = []

        for i in range(self.num_dofs):
             self.joint_type.append(self.gym.get_asset_joint_type(robot_asset, i))


        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # print("#########################rigid_shape_props", len(rigid_shape_props))
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            # print("#########################dof_props", len(dof_props))
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle) # body_props_len = 23
            # print("##################################### body_props", len(body_props))
            body_props = self._process_rigid_body_props(body_props, i)
            # print("#########################body_props", len(body_props))
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        
        
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        self.thigh_contact_indices = torch.zeros(len(thigh_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(thigh_contact_names)):
            self.thigh_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], thigh_contact_names[i])
        
        self.calf_contact_indices = torch.zeros(len(calf_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(calf_contact_names)):
            self.calf_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], calf_contact_names[i])
        
        
    def reset_idx(self, env_ids):

        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        

        # reset buffers
        self.last_actions[env_ids] = 0.
        ### gate
        self.last_last_actions[env_ids] = 0.
        ### gate
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # fill extras
        # The episode key is used in logging within the on-policy runner to track episode-related metrics
        # If you want to add information, try to include it under the episode key to keep all episode-related 
        # metrics organized. Otherwise, create a separate key in extras to store additional data without 
        # interfering with episode tracking.
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        ### gate
        for i in range(self.lag_buffer.maxlen):
            self.lag_buffer[i][env_ids] *= 0
        ### gate

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        ### gait
        self.curriculum_thresholds = class_to_dict(self.cfg.curriculum_thresholds)
        ### gait
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def get_euler_xyz_tensor(self, quat):
        r, p, w = get_euler_xyz(quat)
        # stack r, p, w in dim1
        euler_xyz = torch.stack((r, p, w), dim=1)
        euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
        return euler_xyz

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
