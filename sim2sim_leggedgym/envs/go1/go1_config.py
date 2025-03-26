# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from sim2sim_leggedgym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO1sim2simCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        frame_stack = 15 #15
        c_frame_stack = 3 #3
        num_single_obs = 48
        num_observations = int(frame_stack * num_single_obs) # 48*15 = 720
        single_num_privileged_obs = 72 
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs) 
        num_actions = 12
        num_envs = 500
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm 
        episode_length_s = 20 # episode length in seconds
        use_ref_actions = False
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh or uneven
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
    
    
    ### gaits
    class curriculum_thresholds:
        c = {
        "tracking_lin_vel" : 0.8,  
        "tracking_ang_vel" : 0.5,
        "tracking_contacts_shaped_force" : 0.8,  
        "tracking_contacts_shaped_vel" : 0.8}
    ### gaits


    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error

        ## gaits
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1
        limit_gait_offset = [0, 1]
        limit_gait_bound = [0, 1]
        limit_gait_phase = [0, 1]
        limit_gait_frequency = [2, 4] # [2 4 ]
        limit_gait_duration = [0.5, 0.5]
        limit_footswing_height = [0.03, 0.35] #[0.06, 0.061] # if i change it i get error
        limit_body_pitch = [-0.4, 0.4]
        limit_body_roll = [-0.0, 0.0]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.10, 0.45] #[0.0, 0.01]        # if i change it i get error
        limit_stance_length =  [0.35, 0.45] # [0.0, 0.01]       # if i change it i get error
        curriculum_seed = 100
        gait_phase_cmd_range = [0.0, 1]
        gait_offset_cmd_range = [0.0, 1]
        gait_bound_cmd_range = [0.0, 1]
        gait_frequency_cmd_range = [2, 4]#[2.0, 4]
        gait_duration_cmd_range = [0.5, 0.5]
        footswing_height_range = [0.03, 0.35] #[0.06, 0.061]
        body_pitch_range = [0.0, 0.01]
        body_roll_range = [0.0, 0.01]
        aux_reward_coef_range = [0.0, 0.01]
        compliance_range = [0.0, 0.01]
        stance_width_range = [0.10, 0.45]# [0.0, 0.01]
        stance_length_range = [0.35, 0.45] # [0.0, 0.01]
        walkin_type = "trot"
        ### gaits

        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]  
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["trunk"]
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        num_buckets_friction = 64
        randomize_friction = True
        friction_range = [0.35, 1.4]

        randomize_base_mass = True
        added_mass_range = [-1.2, 1.2]

        num_buckets_restitution = 64
        randomize_restitution = True
        restitution_range = [0, 1.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_joint_damping = False
        joint_damping_range = [0.001, 0.05]

        randomize_joint_friction = False
        joint_friction_range = [0.05, 0.2]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        action_delay = 0.5
        action_noise = 0.024 # 0.04

        ### gaits
        lag_timesteps = 6
        randomize_lag_timesteps = True
        ### gaits

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

  
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.8
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits = -10.0

            ### gaits 
            # termination = -0.0
            # tracking_lin_vel = 1 #1.2
            # tracking_ang_vel =  0.5 # 0.7
            # lin_vel_z = -0.4 #-2.0
            # ang_vel_xy =  -0.
            # orientation =  -300
            # torques = -0.
            # dof_vel = -0.
            # dof_acc = -1e-7
            # base_height = 0
            # feet_air_time = 0 #1.0
            # collision = -1.
            # feet_stumble = -0.0 
            # action_rate =  -0.02
            # stand_still =  -0
            # jump =  10
            # tracking_contacts_shaped_force = 1# 4 #1
            # tracking_contacts_shaped_vel = 3
            # raibert_heuristic = -10
            # action_smoothness_1 = -0
            # action_smoothness_2 = -0
            # feet_clearance_cmd_linear = -5 #1
            # feet_impact_vel =  0
            # feet_slip = 0#-8e-4
            # orientation_control =  -10
            # new_contact_shape = 0.0
            ### gaits

        
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        cycle_time = 0.64

        ### gaits
        gait_force_sigma = 100
        kappa_gait_probs = 0.07
        gait_vel_sigma = 10
        # only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # only_positive_rewards_ji22_style = False
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.5
        max_contact_force = 100. # forces above this value are penalized
        # ADDED 
        boddy_height_range = [0.3, 0.5]
        sigma_rew_neg = 0.02
        # base_height_target = 0.5
        ### gaits


    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(LeggedRobotCfg.noise):
        add_noise = True # TODO False
        noise_level = 0.4 # scales other values
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            lin_vel = 1  #0.14
            ang_vel = 1  #0.2
            gravity = 1
            commands = 0
            dof_pos = 1 #0.01
            dof_vel = 1   #1.5
            actions = 0
            height_measurements = 0.1

class GO1sim2simCfgPPO( LeggedRobotCfgPPO ):
    seed = -1 # -1 for random
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1'
        Export_Policy_as_jit = True
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # "/home/mehtimans/sim2sim_leggedgym/logs/go1/Dec02_16-32-23_/"  # -1 = last run
        checkpoint = -1 #"3000" # -1 = last saved model checkpoint = "/home/mehtimans//sim2sim_leggedgym/logs/rough_iust/Nov04_18-38-25_/model_1500.pt"
        resume_path = None # updated from load_run and chkpt

  
