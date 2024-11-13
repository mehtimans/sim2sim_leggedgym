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

# Desired positions for each of the 12 joints
# qdes = np.array([0.0] * 12)  # Example desired joint positions (adjust as needed)
# Desired positions for each of the 12 joints
qdes = np.array([0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 0.8, -1.5, -0.1, 1.0, -1.5])
pdes = np.array([0.0, 0.0, 0.32])# x y z
rotdes = [0.0, 0.0, 0.0, 1.0] # w x y z

# Set initial joint positions directly to qdes
data.qpos[7:19] = qdes  # Modify qpos values for the DOFs you want to home

# Set initial base positions directly to qdes
data.qpos[0:3] = pdes  

# Set initial base orientation directly to qdes
data.qpos[3:7] = rotdes 

print("############################ qvel length", len(data.qvel))
print("############################ qpos length", len(data.qpos))

# Print all joint names
print("Joint Names:")
# Check each joint's type and name to confirm
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_type = model.jnt_type[i]
    joint_qposadr = model.jnt_qposadr[i]  # qpos index for the joint
    joint_qveladr = model.jnt_dofadr[i]   # qvel index for the joint
    print(f"Joint {i}: Name = {joint_name}, Type = {joint_type}, qpos index: {joint_qposadr}, qvel index: {joint_qveladr}")

joint_names = [model.joint(j).name for j in range(model.njnt)]  # model.njnt gives the number of joints
print("Joint positions:", joint_names)

# Print all body names
print("Body Names:")
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"Body {i}: Name = {body_name}")

# Print all sensor names and types
print("Sensor Names:")
for i in range(model.nsensor):
    sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    sensor_type = model.sensor_type[i]  # Type of the sensor
    print(f"Sensor {i}: Name = {sensor_name}, Type = {sensor_type}")
          
# Apply changes by stepping once to update the simulation state
mujoco.mj_forward(model, data)  # Forward the physics to apply the changes

# Assuming `viewer` is already defined as an instance of MujocoViewer
while True:
    # Step the simulation
    mujoco.mj_step(model, data)
    
    # Render the viewer
    viewer.render()
    
    # Break the loop if the viewer window is closed
    if not viewer.is_alive:
        break

# Close the viewer after exiting the loop
viewer.close()