import matplotlib.pyplot as plt
import os
import numpy as np



# plot the mean reward
com_lin_x = []
com_lin_y = []
com_ang_yaw = []
vx = []
vy = []
yaw = []

iterations1 = []
iterations2 = []

with open('desired_command.txt', 'r') as f:
    for index1, line1 in enumerate(f):
            com_lin_xs, com_lin_ys, com_ang_yaws, vxs, vys, yaws = (line1.strip().split())
            # Use index + 1 as the iteration number (index starts from 0)
            com_lin_xs = float(com_lin_xs)
            com_lin_ys = float(com_lin_ys)
            com_ang_yaws = float(com_ang_yaws)
            vxs = float(vxs)
            vys = float(vys)
            yaws = float(yaws)

            iterations1.append(index1 + 1)
            com_lin_x.append(com_lin_xs)
            com_lin_y.append(com_lin_ys)
            com_ang_yaw.append(com_ang_yaws)
            vx.append(vxs)
            vy.append(vys)
            yaw.append(yaws)



iterations1 = [x * 0.005 for x in iterations1]


# 1          
plt.figure(figsize=(20, 10))
plt.plot(iterations1 , vx, label='command', color='r', linewidth=1)
plt.plot(iterations1 , com_lin_x, label='simulation', color='b', linewidth=3)
plt.xlabel('Time (s)', fontsize=14, fontname='serif')
plt.ylabel('Linear Velocity in x-direction (m/s)', fontsize=18, fontname='serif')
plt.legend(prop={'family': 'serif', 'size': 14})
plt.ylim(-1.5, 1.5)
plt.xticks(fontsize=14, fontname='serif')  # X-axis tick labels
plt.yticks(fontsize=14, fontname='serif')  # Y-axis tick labels
plt.grid(True)
plt.savefig('Linear Velocity in x-direction.png')

plt.figure(figsize=(20, 10))
plt.plot(iterations1 , vy, label='command', color='r', linewidth=1)
plt.plot(iterations1 , com_lin_y, label='simulation', color='b', linewidth=3)
plt.xlabel('Time (s)', fontsize=18, fontname='serif')
plt.ylabel('Linear Velocity in y-direction (m/s)', fontsize=18, fontname='serif')
plt.legend(prop={'family': 'serif', 'size': 14})
plt.ylim(-1.5, 1.5)
plt.xticks(fontsize=16, fontname='serif')  # X-axis tick labels
plt.yticks(fontsize=16, fontname='serif')  # Y-axis tick labels
plt.grid(True)
plt.savefig('Linear Velocity in y-direction.png')


plt.figure(figsize=(20, 10))
plt.plot(iterations1 , yaw, label='command', color='r', linewidth=1)
plt.plot(iterations1 , com_ang_yaw, label='simulation', color='b', linewidth=3)
plt.xlabel('Time (s)', fontsize=18, fontname='serif')
plt.ylabel('Yaw rate (rad/s)', fontsize=18, fontname='serif')
plt.legend(prop={'family': 'serif', 'size': 14})
plt.ylim(-5, 5)
plt.xticks(fontsize=16, fontname='serif')  # X-axis tick labels
plt.yticks(fontsize=16, fontname='serif')  # Y-axis tick labels
plt.grid(True)
plt.savefig('Yaw rate.png')



# 1          
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.plot(plt_iterations, plt_mean_rewards, marker='.', linestyle='-', color='b')
# plt.title('Mean Reward by Iteration')
# plt.xlabel('Iteration')
# plt.ylabel('Mean Reward')
# plt.grid(True)
# plt.subplot(1, 2, 2)
# plt.plot(plt_iterations, plt_mean_lens, marker='.', linestyle='-', color='b')
# plt.title('Mean length by Iteration')
# plt.xlabel('Iteration')
# plt.ylabel('Mean length')
# plt.grid(True)
#     # Save the plot to a file
# plt.savefig('mean_rewards_fig.png')

#     # Optionally, clear the figure to free memory
# plt.close()
