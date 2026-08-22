# sim2sim leggedgym

`sim2sim_leggedgym` is a reinforcement learning pipeline for quadruped locomotion, developed primarily for the **Unitree Go1** and based on the [legged_gym](https://github.com/leggedrobotics/legged_gym.git) framework.

The project uses **NVIDIA Isaac Gym** for massively parallel robot simulation and **Proximal Policy Optimization (PPO)** for training locomotion policies. The training environment incorporates extensive **domain randomization** and other sim-to-real considerations to improve policy robustness against differences between simulation and real-world dynamics.

In addition to training in Isaac Gym, the repository provides an **Isaac Gym-to-MuJoCo sim-to-sim pipeline**, allowing trained policies to be deployed and evaluated in MuJoCo before moving toward real-robot deployment. This provides an additional validation stage for identifying differences in robot dynamics, control parameters, joint conventions, contact behavior, and simulation models.

The framework also includes support for the custom **IUST quadruped robot**, enabling the same training and sim-to-sim workflow to be applied to different quadruped platforms.

---
### Installation ###

1. Create a new virtual environment using **Python 3.6, 3.7, or 3.8**. Python **3.8 is recommended**.

2. Install a PyTorch version compatible with your CUDA version.
   - You can find previous PyTorch versions and their corresponding CUDA installation commands here: https://pytorch.org/get-started/previous-versions/

3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
   - The following tutorial is also helpful for setting up Isaac Gym on Ubuntu: https://learningreinforcementlearning.com/setting-up-isaac-gym-on-an-ubuntu-laptop-785b5a15e5a9

4. Install `sim2sim_leggedgym`
    - Clone this repository:
    - Then install the package in editable mode: `cd sim2sim_leggedgym && pip install -e .`

---
### Code Structure ###

1. **Robot environments and configurations:** Each robot has its own environment and configuration files:

   * Go1: `envs/go1/go1_env.py` and `envs/go1/go1_config.py`
   * IUST: `envs/iust/iust_env.py` and `envs/iust/iust_config.py`

   Both environments inherit common functionality from the base classes in `envs/base/`, mainly `legged_robot.py` and `legged_robot_config.py`. This keeps the shared locomotion, simulation, control, reward, and observation logic in one place while allowing robot-specific parameters to be defined separately.

2. **Inheritance-based design:** The environment and configuration classes use inheritance. Shared behavior is implemented in the base classes under `envs/base/`, while the Go1 and IUST classes extend or override these base implementations to define robot-specific behavior and parameters. This makes it easier to add new quadruped robots without duplicating the common training and simulation logic.

3. **Reward system:** The reward structure follows the original `legged_gym` design. Each enabled reward term in the configuration corresponds to a reward function implemented in the environment. The active reward terms are evaluated and summed during training to obtain the total reward.

4. **PPO training pipeline:** The PPO implementation is located in `sim2sim_leggedgym/algo/ppo/`. It contains the actor-critic network, PPO algorithm, rollout storage, and on-policy runner used for training locomotion policies.

5. **Task registration:** Each robot task is registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. The task registrations are defined in `envs/__init__.py`, allowing the desired robot environment and training configuration to be selected by task name.

6. **Training and evaluation scripts:** The main scripts are located in `sim2sim_leggedgym/scripts/`:

   * `train.py`: Trains a locomotion policy in NVIDIA Isaac Gym using PPO.
   * `play.py`: Loads and evaluates a trained policy in Isaac Gym.
   * `sim2sim.py`: Loads an Isaac Gym-trained policy and evaluates it in MuJoCo for sim-to-sim validation.

7. **Robot models and assets:** Robot models are stored under `resources/robots/`. Each robot directory contains the corresponding meshes and simulation model files:

   * Go1: `resources/robots/go1/`
   * IUST: `resources/robots/iust/`

   URDF files are used for Isaac Gym, while XML/MJCF models are used for MuJoCo-based sim-to-sim evaluation.

8. **Training logs and checkpoints:** Training results, experiment logs, and policy checkpoints are stored under `logs/`, with separate directories for Go1, IUST, and test experiments.

---
### Usage ###

1. **Train:** 
```python
python sim2sim_leggedgym/scripts/train.py --task=<task_name>
```
* To run on CPU add the following arguments:
  `--sim_device=cpu --rl_device=cpu`
  (Simulation on CPU and reinforcement learning on GPU is also possible.)

* To run headless (no rendering) add: `--headless`

* **Important:** To improve performance, once training starts press `v` to stop rendering. You can enable it later to check the training progress.

* The trained policy is saved in:

  `logs/<robot_name>/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`
  
  where `<experiment_name>` and `<run_name>` are defined in the training configuration.

* The following command line arguments override the values set in the config files:

  * `--task TASK`: Task name.
  * `--resume`: Resume training from a checkpoint.
  * `--experiment_name EXPERIMENT_NAME`: Name of the experiment to run or load.
  * `--run_name RUN_NAME`: Name of the run.
  * `--load_run LOAD_RUN`: Name of the run to load when `resume=True`. If `-1`, the last run is loaded.
  * `--checkpoint CHECKPOINT`: Saved model checkpoint number. If `-1`, the last checkpoint is loaded.
  * `--num_envs NUM_ENVS`: Number of environments to create.
  * `--seed SEED`: Random seed.
  * `--max_iterations MAX_ITERATIONS`: Maximum number of training iterations.

2. **Play a trained policy:**

```python
python sim2sim_leggedgym/scripts/play.py --task=<task_name>
```
* By default, the loaded policy is the last model of the last run in the experiment folder.
* Other runs or model iterations can be selected by setting `load_run` and `checkpoint` in the training configuration.

3. **Play sim-to-sim transfer in MuJoCo:**

```python
python sim2sim_leggedgym/scripts/sim2sim.py --task=<task_name>
```
* This script loads a policy trained in Isaac Gym and evaluates it in MuJoCo using the corresponding robot model.
* The MuJoCo models are located in:
  `resources/robots/<robot_name>/xml/`
---
 
### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`. It is also possible that you need to do `export LD_LIBRARY_PATH=/path/to/libpython/directory` / `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`(for conda user. Replace /path/to/ to the corresponding path.).


