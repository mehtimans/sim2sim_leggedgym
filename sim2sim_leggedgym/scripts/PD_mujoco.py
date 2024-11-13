import numpy as np
import mujoco, mujoco_viewer
# from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
# import mujoco_py
import copy
from sim2sim_leggedgym import LEGGED_GYM_ROOT_DIR

# Load the model and initialize simulation
model = mujoco.MjModel.from_xml_path("/home/mehtimans/sim2sim_leggedgym/resources/robots/go1/xml/go1.xml")
model.opt.timestep = 0.0001
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)




# Define PD gains for 12 joints
# Kp = np.array([0.0000] * 12)   # Set Kd to 0 for now

Kp = np.array([200] * 12)
Kd = np.array([0.1] * 12)   # Set Kd to 0 for now

# Desired positions for each of the 12 joints
# qdes = np.array([0.0] * 12)  # Example desired joint positions (adjust as needed)
# Desired positions for each of the 12 joints
qdes = np.array([0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 0.8, -1.5, -0.1, 1.0, -1.5])
pdes = np.array([0.0, 0.0, 0.32])
rotdes = [0.0, 0.0, 0.0, 1.0]

# Set initial joint positions directly to qdes
data.qpos[7:19] = qdes  # Modify qpos values for the DOFs you want to home

# Set initial base positions directly to qdes
data.qpos[0:3] = pdes  

# Set initial base orientation directly to qdes
data.qpos[3:7] = rotdes 

# Save the initial simulation state
initial_state = mujoco.MjData(model)
initial_state.qpos = data.qpos.copy()
initial_state.qvel = data.qvel.copy()

while True:
    # Reset to initial state at the start of each cycle
    data.qpos = initial_state.qpos.copy()
    data.qvel = initial_state.qvel.copy()

    for _ in range(1000000):
        # Current joint positions (qpos) and velocities (qvel) for 12 joints
        # print(sim.data.qpos.shape)

        # Get current joint positions and velocities for the first 12 actuated joints
        qpos = data.qpos[7:19]  # First 12 elements correspond to joint positions
        qvel = data.qvel[6:18]  # First 12 elements correspond to joint velocities


        # PD control law
        q_error = qdes - qpos       # Position error
        qvel_error = -qvel          # Velocity error (assuming desired velocity is zero)

        # Compute control torques using PD formula: torque = Kp * error + Kd * velocity_error
        torque = Kp * q_error + Kd * qvel_error

        # Apply the torques to the 12 joints
        data.ctrl[:12] = torque  # Apply torque to the 12 joints

        # Step the simulation
        mujoco.mj_step(model, data)
        viewer.render()
  
         

        # Print the current positions for debugging
        print("Current qpos:", qpos)

    if os.getenv('TESTING') is not None:
        break

    