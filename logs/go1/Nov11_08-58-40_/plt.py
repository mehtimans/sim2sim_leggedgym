import matplotlib.pyplot as plt
import os


    # plot the mean reward
plt_iterations = []
plt_mean_rewards = []
plt_mean_lens = []
with open('mean_rewards.txt', 'r') as f:
    for index, line in enumerate(f):
            plt_mean_reward  = (line.strip())
            # Use index + 1 as the iteration number (index starts from 0)
            plt_mean_reward = float(plt_mean_reward)
           
            plt_iterations.append(index + 1)
            plt_mean_rewards.append(plt_mean_reward)
            
            
plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
plt.plot(plt_iterations, plt_mean_rewards, marker='.', linestyle='-', color='b')
plt.title('Mean Reward by Iteration')
plt.xlabel('Iteration')
plt.ylabel('Mean Reward')
plt.grid(True)
#plt.subplot(1, 2, 2)
#plt.plot(plt_iterations, plt_mean_lens, marker='.', linestyle='-', color='b')
#plt.title('Mean length by Iteration')
#plt.xlabel('Iteration')
#plt.ylabel('Mean length')
#plt.grid(True)
    # Save the plot to a file
plt.savefig('mean_rewards_fig.png')

    # Optionally, clear the figure to free memory
plt.close()
