import numpy as np
import torch
from matplotlib import pyplot as plt


def is_met(scale, l2_err, threshold):
    return (l2_err / scale) < threshold


def key_is_met(metric_cache, config, ep_len, target_key, env_id, threshold):
    # metric_cache[target_key][env_id] / ep_len
    scale = 1
    l2_err = 0
    return is_met(scale, l2_err, threshold)


class Curriculum:
    def set_to(self, low, high, value=1.0):
        inds = np.logical_and(
            self.grid >= low[:, None],
            self.grid <= high[:, None]
        ).all(axis=0) ## just true or false

        assert len(inds) != 0, "You are intializing your distribution with an empty domain!"

        self.weights[inds] = value ## just takes value 1 or 0
        # print("###################inds", inds)
        # print("###################self.weights", self.weights) # # The shape of self.weights and inds: (total combinations of all bins crossed together)
        # print("###################self.grid", np.shape(self.grid)) # The shape of self.grid: (number of initialization inputs,
        # total combinations of all bins crossed together)

    def __init__(self, seed, **key_ranges):
        self.rng = np.random.RandomState(seed)
        # print("###################inds", self.rng) # RandomState(MT19937)
        # print("################### key_ranges items", key_ranges.items()) # dict_items([('gait_frequency', (2, 4, 1)), ('gait_phase', (0, 1, 1)), ('gait_offset', (0, 1, 1)),
        # ('gait_bounds', (0, 1, 1)), ('gait_duration', (0.5, 0.5, 1)), ('footswing_height', (0.03, 0.35, 1)), ('body_pitch', (-0.4, 0.4, 1)), ('body_roll', (-0.0, 0.0, 1)), 
        # ('stance_width', (0.1, 0.45, 1)), ('stance_length', (0.35, 0.45, 1)), ('aux_reward_coef', (0.0, 0.01, 1))])

        self.cfg = cfg = {}
        self.indices = indices = {}
        for key, v_range in key_ranges.items():
            # print("################### v_range", type(v_range)) # a touple with three value for each key
            bin_size = (v_range[1] - v_range[0]) / v_range[2] # compute size of each bin
            cfg[key] = np.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2]) # Compute bin centers by adding bin_size/2 to low and high ranges
            indices[key] = np.linspace(0, v_range[2]-1, v_range[2]) # index of each bin
            # print("###################### bin_size", cfg)
        self.lows = np.array([range[0] for range in key_ranges.values()]) 
        self.highs = np.array([range[1] for range in key_ranges.values()])
        # size of lows and highs is np array (number of initialization inputs)


        # self.bin_sizes = {key: arr[1] - arr[0] for key, arr in cfg.items()}
        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()} # A dictionary containing the bin size for each initialization input.
        # print("###################################### self.bin_sizes", self.bin_sizes)

        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij')) # The shape (11, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) indicates that there are 11 initialization inputs. 
        # The first dimension represents the number of inputs, and each subsequent dimension corresponds to the number of bins (or center bins) for each respective input.

        # print("############################## self rew grid", np.shape(self._raw_grid))
        self._idx_grid = np.stack(np.meshgrid(*indices.values(), indexing='ij'))
        self.keys = [*key_ranges.keys()]
        self.grid = self._raw_grid.reshape([len(self.keys), -1]) # at last self._raw_grid become shape(number of initialization inputs, bin1 * bin2 * bin3 ....) . 
        # each column in self.grid represents one unique combination of all your parameters.
        self.idx_grid = self._idx_grid.reshape([len(self.keys), -1])
        # self.grid = np.stack([params.flatten() for params in raw_grid])

        self._l = l = len(self.grid[0]) # l is bin1 * bin2 * bin3 ....
        # print("############################ self._l", self._l)
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}

        self.weights = np.zeros(l)
        self.indices = np.arange(l)

    def __len__(self):

        return self._l

    def __getitem__(self, *keys):
        pass

    def update(self, **kwargs):
        # bump the envelop if
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """default to uniform"""
        if low is not None and high is not None: # if bounds given
            valid_inds = np.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)
            temp_weights = np.zeros_like(self.weights)
            temp_weights[valid_inds] = self.weights[valid_inds]
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else: # if no bounds given
            inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())

        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        bin_sizes = np.array([*self.bin_sizes.values()])
        low, high = centroids + bin_sizes / 2, centroids - bin_sizes / 2
        return self.rng.uniform(low, high)#.clip(self.lows, self.highs)

    def sample(self, batch_size, low=None, high=None):
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        return np.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds


class SumCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.success = np.zeros(len(self))
        self.trials = np.zeros(len(self))

    def update(self, bin_inds, l1_error, threshold):
        is_success = l1_error < threshold
        self.success[bin_inds[is_success]] += 1
        self.trials[bin_inds] += 1

    def success_rates(self, *keys):
        s_rate = self.success / (self.trials + 1e-6)
        s_rate = s_rate.reshape(list(self.ls.values()))
        marginals = tuple(i for i, key in enumerate(self.keys) if key not in keys)
        if marginals:
            return s_rate.mean(axis=marginals)
        return s_rate


class RewardThresholdCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.episode_reward_lin = np.zeros(len(self))
        self.episode_reward_ang = np.zeros(len(self))
        self.episode_lin_vel_raw = np.zeros(len(self))
        self.episode_ang_vel_raw = np.zeros(len(self))
        self.episode_duration = np.zeros(len(self))

    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = np.ones(self.grid.shape[0]) * ranges
        bin_inds = bin_inds.reshape(-1)

        adjacent_inds = np.logical_and(
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1, 1, 1),
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1, 1, 1)
        ).all(axis=0)

        # This method identifies the bins in the grid that are within a specified range of the given bin indices (bin_inds).
        # If 'ranges' is a float, it is applied uniformly across all grid dimensions. The method compares each bin's position
        # with the positions of the bins in bin_inds and returns a boolean array (adjacent_inds), where True indicates a bin is 
        # within the specified range of at least one of the bins in bin_inds.

        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):
        
        # print("#########################################", bin_inds) # previous bin index for terminated env
        # print("############################################ task rewards", task_rewards)

        is_success = 1.
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success = is_success * (task_reward > success_threshold).cpu()
            # print("################################### is_success", (task_reward > success_threshold))

        # Initialize is_success as 1 (indicating full success).
        # Iterate through each task's reward and success threshold.
        # For each task, compare the reward with its threshold (task_reward > success_threshold).
        # If the reward exceeds the threshold, the comparison returns True (1), otherwise False (0).
        # Multiply the current value of is_success by the result of each comparison.
        # If all tasks succeed, is_success remains 1; if any task fails, is_success becomes 0.
        #  is_success * (task_reward > success_threshold).cpu() is 0.0 or 1.0, when you multiply a float by a 
        # boolean, the boolean is automatically converted to a float
        
        if len(success_thresholds) == 0:
            is_success = np.array([False] * len(bin_inds))
        else:
            is_success = np.array(is_success.bool())
        
        # just convert to bool. an array that has a bool object

         
        # if len(is_success) > 0 and is_success.any():
        #     print("successes")

        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        # Update the weights of the bins that were successful (where is_success is True).
        # Add 0.2 to the current weight of the bin, ensuring the new value is clipped between 0 and 1.
        
        
        # Get the bins that are adjacent to the bins in bin_inds where the task was successful (is_success is True)
        # The method get_local_bins returns a boolean array indicating which bins are within the specified local_range.
        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        

        # Iterate through each adjacent bin (adjacent corresponds to each successful task's adjacent bins)
        for adjacent in adjacents:
            #print(adjacent)
            #print(self.grid[:, adjacent])
            # Find the indices of bins that are considered "adjacent" (where the value is True in the boolean array)
            adjacent_inds = np.array(adjacent.nonzero()[0])
            # Update the weights of the adjacent bins: Add 0.2 to their current weights and clip the result between 0 and 1
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)

    def log(self, bin_inds, lin_vel_raw=None, ang_vel_raw=None, episode_duration=None):
        self.episode_lin_vel_raw[bin_inds] = lin_vel_raw.cpu().numpy()
        self.episode_ang_vel_raw[bin_inds] = ang_vel_raw.cpu().numpy()
        self.episode_duration[bin_inds] = episode_duration.cpu().numpy()

if __name__ == '__main__':
    r = RewardThresholdCurriculum(100, x=(-1, 1, 5), y=(-1, 1, 2), z=(-1, 1, 11))

    assert r._raw_grid.shape == (3, 5, 2, 11), "grid shape is wrong: {}".format(r.grid.shape)

    low, high = np.array([-1.0, -0.6, -1.0]), np.array([1.0, 0.6, 1.0])

    # r.set_to(low, high, value=1.0)

    adjacents = r.get_local_bins(np.array([10, ]), range=0.5)
    for adjacent in adjacents:
        adjacent_inds = np.array(adjacent.nonzero()[0])
        print(adjacent_inds)
        r.update(bin_inds=adjacent_inds, lin_vel_rewards=np.ones_like(adjacent_inds),
                 ang_vel_rewards=np.ones_like(adjacent_inds), lin_vel_threshold=0.0, ang_vel_threshold=0.0,
                 local_range=0.5)

    samples, bins = r.sample(10_000)

    plt.scatter(*samples.T[:2])
    plt.show()
