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
        print("base pos:########################################### ", self.base_pos)
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
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))
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
        self.prev_foot_velocities = self.foot_velocities.clone() ## check!!
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
        self.prev_foot_velocities = self.foot_velocities.clone()
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
        # print("#########################", self.base_pos)
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
        self._step_contact_targets()
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
                                              self.commands[:, 4:],
                                              (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                                              self.dof_vel * self.obs_scales.dof_vel, # 12
                                              self.actions, # 12
                                              self.env_frictions, # 1
                                              self.clock_inputs
                                              # self.env_restitution, # 1
                                              # self.payloads, # 1
                                              # self.rand_push_vel, # 2
                                              # self.com_displacements, # 3
                                              # self.joint_friction, # 1
                                              # self.joint_damping, # 1
                                              # self.contact_foot_z_force, #4
                                              # self.contact_state, # 4
                                              # self.thigh_contact_state, # 4
                                              # self.calf_contact_state  # 4
                                              ),dim=-1) # 74
          
        # print("########################################## command", self.commands)
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
                                self.commands[:, 4:],
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions,
                                self.clock_inputs
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
            # act_count = self.gym.get_actor_actuator_count(env_handle, actor_handle)
            # print("*******************************************************act_count:", act_count)
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
        
        ### gait
        self._init_custom_buffer()
        ### gait

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
    
    ###################################### gait

    def _resample_commands(self, env_ids): 
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        timesteps = int(self.cfg.commands.resampling_time / self.dt) # self.dt is sim_dt * decimation / timesteps = 500
        ep_len = min(self.max_episode_length, timesteps) ### ep_len = 500
        # print("#######################################3 envids", env_ids)
        # update curricula based on terminated environment bins and categories
        for i, curriculum in enumerate(self.curricula): ### i = [0, 1, 2, 3] and curriculum is 4 curriculum class that initialized in _init_command_distribution
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i ## self.env_command_categories contains all environment categories.
            # env_ids_in_category is a boolean array indicating which env_ids belong to a specific category.
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1: # env_ids_in_category has one boolean element loop like [true]
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue      
            # The two lines above ensure that env_ids_in_category contains at least one element. If it is empty, the loop will continue.
            
            env_ids_in_category = env_ids[env_ids_in_category]
            # At last, env_ids_in_category will be a torch tensor containing the indices of environments that belong to category i.
            
            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])
            
            # print("############################################ task rewards", task_rewards)
            # print("############################################ success_thresholds", success_thresholds) # note to env_ids_in_category
            # # success_thresholds [0.015999999642372132, 0.004999999888241291, 0.015999999642372132, 0.0479999989271164] always?

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]### wrong
            # print("################################################", old_bins) # This shows which bin is selected from all available bins. 
            # Each bin is a combination of all random commands.
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0]))
                                           
        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device) # it especified a number between 0 and 1 in number of env_ids length 

        probability_per_category = 1. / len(self.category_names) # probability of each category_ all has sam prob 
                
        # category_env_ids is a list containing len(self.category_names) torch tensors. Each tensor holds 
        # the indices of env_ids that, based on random_env_floats and probability_per_category, have been 
        # randomly assigned to that category.
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in range(len(self.category_names))]
                    

        # sample from new category curricula
        # print('###################################### env_ids_in_category', category_env_ids)
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue
            

            new_phas_info, new_bin_inds = curriculum.sample(batch_size=batch_size)
            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i
            self.phases_info = self.phases_info.to(self.device) 
            self.phases_info[env_ids_in_category, :] = torch.Tensor(new_phas_info[:, :11]).to(self.device)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
       
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        pronking_envs = env_ids[random_env_floats < 1/4]
        trotting_envs = env_ids[torch.logical_and(1/4 <= random_env_floats, random_env_floats < 2/4)]
        pacing_envs = env_ids[torch.logical_and(2/4 <= random_env_floats, random_env_floats < 3/4)]
        bounding_envs = env_ids[torch.logical_and(3/4 <= random_env_floats, random_env_floats < 4/4)]

     

        foot_0 = torch.zeros(self.num_envs).to(self.device)
        foot_1 = torch.zeros(self.num_envs).to(self.device)
        foot_2 = torch.zeros(self.num_envs).to(self.device)
        foot_3 = torch.zeros(self.num_envs).to(self.device)
     
        foot_0[bounding_envs] =  0.5
        foot_1[bounding_envs] =  0.5

        foot_0[pacing_envs] = 0.5
        foot_2[pacing_envs] = 0.5

        foot_0[trotting_envs] = 0.5
        foot_3[trotting_envs] = 0.5
        
       
        self.commands[env_ids, 4] = self.phases_info[env_ids, 0]
        self.commands[env_ids, 5] = foot_1[env_ids]
        self.commands[env_ids, 6] = foot_2[env_ids]
        self.commands[env_ids, 7] = foot_3[env_ids]

        # ADDED
        # There is a difference between self.command_sums and self.episode_sums initializations. self.episode_sums is initialized 
        # when an environment needs to reset due to a termination condition, while self.command_sums is initialized when the resample_command function 
        # is called—either because of termination or because the command resampling time has been reached.

        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
  

    def _step_contact_targets(self):
        frequencies = self.phases_info[:, 0]
        phases = self.phases_info[:, 1]
        # print(phases)
        offsets = self.phases_info[:, 2]
        bounds = self.phases_info[:, 3]
        durations = self.phases_info[:, 4]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0).to(self.device)

        if self.cfg.commands.pacing_offset:
            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + phases]

            # print(offsets)
            # print(bounds)
        else:
            foot_indices = [self.gait_indices + self.commands[:,5] + self.commands[:,6] + self.commands[:,7],
                            self.gait_indices + self.commands[:,5] ,
                            self.gait_indices + self.commands[:,6] ,
                            self.gait_indices + self.commands[:,7] ]
        # print("HERE")
        # print(phases)
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
        # print(self.foot_indices)
        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))
        
        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR
        # print(self.desired_contact_states[0, 0:4])
        # print("-------------------->>>>>>>>>>>>>>>>>>>")
        # print(self.foot_indices)
    
    def _init_command_distribution(self, envs_ids):
        # new style curriculum
        self.category_names = ['pronk', 'trot', 'pace', 'bound']
        CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass( seed = self.cfg.commands.curriculum_seed,
                                               
                                                gait_frequency = (self.cfg.commands.limit_gait_frequency[0],
                                                                 self.cfg.commands.limit_gait_frequency[1],
                                                                 self.cfg.commands.num_bins_gait_frequency),

                                                gait_phase = (self.cfg.commands.limit_gait_phase[0],
                                                              self.cfg.commands.limit_gait_phase[1],
                                                              self.cfg.commands.num_bins_gait_phase),

                                                gait_offset = (self.cfg.commands.limit_gait_offset[0],
                                                               self.cfg.commands.limit_gait_offset[1],
                                                               self.cfg.commands.num_bins_gait_offset),

                                                gait_bounds = (self.cfg.commands.limit_gait_bound[0],
                                                               self.cfg.commands.limit_gait_bound[1],
                                                               self.cfg.commands.num_bins_gait_bound),

                                                gait_duration = (self.cfg.commands.limit_gait_duration[0],
                                                                 self.cfg.commands.limit_gait_duration[1],
                                                                 self.cfg.commands.num_bins_gait_duration),

                                                footswing_height = (self.cfg.commands.limit_footswing_height[0],
                                                                    self.cfg.commands.limit_footswing_height[1],
                                                                    self.cfg.commands.num_bins_footswing_height),

                                                body_pitch = (self.cfg.commands.limit_body_pitch[0],
                                                              self.cfg.commands.limit_body_pitch[1],
                                                              self.cfg.commands.num_bins_body_pitch),

                                                body_roll = (self.cfg.commands.limit_body_roll[0],
                                                             self.cfg.commands.limit_body_roll[1],
                                                             self.cfg.commands.num_bins_body_roll),

                                                stance_width = (self.cfg.commands.limit_stance_width[0],
                                                                self.cfg.commands.limit_stance_width[1],
                                                                self.cfg.commands.num_bins_stance_width),

                                                stance_length = (self.cfg.commands.limit_stance_length[0],
                                                                 self.cfg.commands.limit_stance_length[1],
                                                                 self.cfg.commands.num_bins_stance_length),

                                                aux_reward_coef = (self.cfg.commands.limit_aux_reward_coef[0],
                                                                   self.cfg.commands.limit_aux_reward_coef[1],
                                                                   self.cfg.commands.num_bins_aux_reward_coef)
                                               )] # self.curricula self.curricala is a list with same size of self.category_names and contains
            # RewardThresholdCurriculum class. just init that
             
        self.env_command_bins = np.zeros(len(envs_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(envs_ids), dtype=np.int)
        low = np.array(
            [
             self.cfg.commands.gait_frequency_cmd_range[0],
             self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
             self.cfg.commands.body_roll_range[0], self.cfg.commands.stance_width_range[0],
             self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0],] )
        high = np.array(
            [
             self.cfg.commands.gait_frequency_cmd_range[1],
             self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1], 
             self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
             self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
             self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1],])

        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

###################################### gait

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
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
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
    

################################### gate    
    def _reward_jump(self):
        reference_heights = 0
        body_height = self.base_pos[:, 2] - reference_heights
        jump_height_target = self.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward

    def _init_custom_buffer__(self):
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
    
    def _init_command_distribution(self, env_ids):
        # new style curriculum

        self.category_names = ['pronk', 'trot', 'pace', 'bound']
        CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                               self.cfg.commands.limit_gait_frequency[1],
                                                               self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                           self.cfg.commands.limit_gait_phase[1],
                                                           self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                            self.cfg.commands.limit_gait_offset[1],
                                                            self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                            self.cfg.commands.limit_gait_bound[1],
                                                            self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                              self.cfg.commands.limit_gait_duration[1],
                                                              self.cfg.commands.num_bins_gait_duration),
                                                footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                 self.cfg.commands.limit_footswing_height[1],
                                                                 self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                           self.cfg.commands.limit_body_pitch[1],
                                                           self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.limit_body_roll[0],
                                                          self.cfg.commands.limit_body_roll[1],
                                                          self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.limit_stance_width[0],
                                                             self.cfg.commands.limit_stance_width[1],
                                                             self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.limit_stance_length[0],
                                                                self.cfg.commands.limit_stance_length[1],
                                                                self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                                self.cfg.commands.limit_aux_reward_coef[1],
                                                                self.cfg.commands.num_bins_aux_reward_coef)
                                               )]
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)
        low = np.array(
            [
             self.cfg.commands.gait_frequency_cmd_range[0],
             self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
             self.cfg.commands.body_roll_range[0], self.cfg.commands.stance_width_range[0],
             self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0],] )
        high = np.array(
            [
             self.cfg.commands.gait_frequency_cmd_range[1],
             self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1], 
             self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
             self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
             self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1],])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _reward_tracking_contacts_shaped_force(self): # TODO
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_new_contact_shape(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        shall_contact = self.foot_indices < 0.5
        in_contact = abs(foot_forces) > 0
        # print(shall_contact.size())
        # print(in_contact.size())
        wrong_contact = in_contact * ~shall_contact 
        wrong_contact2 = ~in_contact * shall_contact
        correct_contact1 = in_contact * shall_contact
        correct_contact2 = ~in_contact * ~shall_contact


        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        in_velocities = foot_velocities > 0.1
        shall_velocity = ~shall_contact
        wrong_velocity1 = in_velocities * ~shall_velocity
        wrong_velocity2 = ~in_velocities * shall_velocity
        true_velocity1 = in_velocities  * shall_velocity
        true_velocity2 = ~in_velocities * ~shall_velocity
        # print(-wrong_contact.sum())
        # return -wrong_contact.sum() - wrong_contact2.sum() + correct_contact1.sum() + correct_contact2.sum() - wrong_velocity1.sum() - wrong_velocity2.sum() + true_velocity1.sum() + true_velocity2.sum()
        return  - wrong_velocity1.sum() - wrong_velocity2.sum() #+ true_velocity1.sum() + true_velocity2.sum()

    def _reward_tracking_contacts_shaped_vel(self): # TODO
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        # print(foot_velocities.sum())
        # print(foot_velocities.size())
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))
        return reward / 4


    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.to(self.device).unsqueeze(1)

        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = self.phases_info[:, 8:9]
        desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        desired_stance_length = self.phases_info[:, 9:10]
        desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        
        frequencies = self.phases_info[:, 0]
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    def _reward_action_smoothness_1(self): # TODO
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, :self.num_actions] - self.last_joint_pos_target[:, :self.num_actions])
        diff = diff * (self.last_actions[:, :self.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):# TODO
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, :self.num_actions] - 2 * self.last_joint_pos_target[:, :self.num_actions] + self.last_last_joint_pos_target[:, :self.num_actions])
        diff = diff * (self.last_actions[:, :self.num_dof] != 0)  # ignore first step
        diff = diff * (self.last_last_actions[:, :self.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:3], dim=2).view(self.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel
    
    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        # print(phases)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        target_height = self.phases_info[:, 5].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.prev_foot_velocities[:, :, 2].view(self.num_envs, -1)
        contact_states = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)
    
    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.phases_info[:, 6:8]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)

        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

