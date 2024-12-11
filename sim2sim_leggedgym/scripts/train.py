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

import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

import isaacgym
from sim2sim_leggedgym.envs import *
from sim2sim_leggedgym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg, env_cfg_dict = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, train_cfg_dict, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)   
    
    ####
    # # saving log_dir in a text file in scripts directory
    file_name = "LOG_DIR.txt"
    with open(file_name, "w") as file:
        file.write(log_dir)
    ####
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
    ########### saving configuration setup
    config_dict = {
        "environment configuration" : env_cfg_dict,
        "training configuration" : train_cfg_dict
    }
    
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config_dict, file)
    ###########

def plt_error():
    plt_iterations = []
    plt_linear_x_errors = []
    plt_linear_y_errors = []
    plt_angular_yaw_errors = []
    
    with open("LOG_DIR.txt", "r") as f:
            log_dir = f.read()

    with open(os.path.join(log_dir,'error.txt'), 'r') as f:
        
        for index, line in enumerate(f):
            plt_linear_x_error, plt_linear_y_error, plt_angular_yaw_error = (line.strip().split())
            plt_linear_x_error = float(plt_linear_x_error)
            plt_linear_y_error = float(plt_linear_y_error)
            plt_angular_yaw_error = float(plt_angular_yaw_error)
            plt_iterations.append(index + 1)
            plt_linear_x_errors.append(plt_linear_x_error)
            plt_linear_y_errors.append(plt_linear_y_error)
            plt_angular_yaw_errors.append(plt_angular_yaw_error)

    plt.figure(figsize=(10, 6))
    plt.plot(plt_iterations, plt_linear_x_errors, marker='', linestyle='-', color='b')
    plt.title('linear x error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir,'linear_x_error.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(plt_iterations, plt_linear_y_errors, marker='', linestyle='-', color='b')
    plt.title('linear y error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir,'linear_y_error.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(plt_iterations, plt_angular_yaw_errors, marker='', linestyle='-', color='b')
    plt.title('angular yaw error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir,'angular_yaw_error.png'))
    plt.close()

def plt_rewards():
    plt_iterations = []
    plt_topic = []
    rewards = []
    with open("LOG_DIR.txt", "r") as f:
            log_dir = f.read()
    
    with open(os.path.join(log_dir,'rewards.text'), 'r') as f:
        for index, line in enumerate(f):
            if index == 0:
                rewards_number = len(line.split(" "))
                for topic in line.split(" "):
                     plt_topic.append(topic)
            else:
                plt_iterations.append(index)
                rewards.append([])
                for value in line.split(" "):
                    try:
                        rewards[-1].append(float(value))
                    except:
                         pass
    rewards_by_topic = list(zip(*rewards))
    for i in range(rewards_number):
        plt.figure(figsize=(10, 6))
        plt.plot(plt_iterations, rewards_by_topic[i], marker='', linestyle='-', color='b')
        plt.title(plt_topic[i].replace("_", " "))
        plt.xlabel('Iteration')
        plt.ylabel('reward')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir,f'{plt_topic[i]}.png'))
        plt.close()

def remove_LOG_DIR():
    try:
      os.remove("LOG_DIR.txt")
    except:
      print("LOG_DIR.txt is not removed.")

  

if __name__ == '__main__':
    
    remove_LOG_DIR()
    args = get_args()
    train(args)
    plt_error()
    plt_rewards()
    remove_LOG_DIR()

