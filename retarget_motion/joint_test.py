import pybullet as p
import pybullet_data
import time
import numpy as np
import os

# Connect to pybullet with GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# Load plane and Go2 robot
plane = p.loadURDF("plane.urdf")
go2_urdf_path = os.path.join(os.path.dirname(os.getcwd()), "go2_description/urdf/go2_description.urdf")
print(f"Loading Go2 URDF from: {go2_urdf_path}")
robot = p.loadURDF(go2_urdf_path, [0, 0, 0.5])

# Print joint information
num_joints = p.getNumJoints(robot)
print(f"\nTotal number of joints: {num_joints}")

# Create a dictionary to store movable joints
movable_joints = {}

for i in range(num_joints):
    joint_info = p.getJointInfo(robot, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    
    type_name = "UNKNOWN"
    if joint_type == p.JOINT_REVOLUTE:
        type_name = "REVOLUTE"
        movable_joints[i] = joint_name
    elif joint_type == p.JOINT_PRISMATIC:
        type_name = "PRISMATIC"
        movable_joints[i] = joint_name
    elif joint_type == p.JOINT_SPHERICAL:
        type_name = "SPHERICAL"
        movable_joints[i] = joint_name
    elif joint_type == p.JOINT_PLANAR:
        type_name = "PLANAR"
        movable_joints[i] = joint_name
    elif joint_type == p.JOINT_FIXED:
        type_name = "FIXED"
    
    print(f"Joint {i}: {joint_name} - Type: {type_name}")

print("\nMovable joints:")
for joint_id, joint_name in movable_joints.items():
    print(f"Joint {joint_id}: {joint_name}")

# Set camera
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3])

# Try to identify toe and hip joints
print("\nTrying to identify toe and hip joints based on naming patterns:")
toe_candidates = []
hip_candidates = []

for joint_id, joint_name in movable_joints.items():
    if "foot" in joint_name.lower() or "toe" in joint_name.lower() or "ankle" in joint_name.lower():
        toe_candidates.append((joint_id, joint_name))
        print(f"Possible toe joint: {joint_id} - {joint_name}")
    
    if "hip" in joint_name.lower() or "shoulder" in joint_name.lower():
        hip_candidates.append((joint_id, joint_name))
        print(f"Possible hip joint: {joint_id} - {joint_name}")

# Move each joint one by one
print("\nTesting joints one by one...")
try:
    for joint_id in sorted(movable_joints.keys()):
        print(f"\nMoving joint {joint_id}: {movable_joints[joint_id]}")
        
        # Reset all joints to 0
        for j in movable_joints.keys():
            p.resetJointState(robot, j, 0)
        
        # Move the current joint
        for angle in np.linspace(0, 0.8, 50):
            p.resetJointState(robot, joint_id, angle)
            p.stepSimulation()
            time.sleep(0.01)
        
        # Hold for a moment
        for _ in range(50):
            p.stepSimulation()
            time.sleep(0.01)
        
        # Move back
        for angle in np.linspace(0.8, 0, 50):
            p.resetJointState(robot, joint_id, angle)
            p.stepSimulation()
            time.sleep(0.01)
except KeyboardInterrupt:
    print("\nJoint test interrupted.")

print("\nJoint test complete. Entering manual control mode.")
print("You can use the sliders to control joints.")

# Create sliders for manual control
sliders = {}
for joint_id in movable_joints.keys():
    sliders[joint_id] = p.addUserDebugParameter(
        f"Joint {joint_id}: {movable_joints[joint_id]}", 
        -3.14, 3.14, 0)

try:
    while True:
        # Update joint positions from sliders
        for joint_id, slider_id in sliders.items():
            angle = p.readUserDebugParameter(slider_id)
            p.resetJointState(robot, joint_id, angle)
            
        p.stepSimulation()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\nSimulation ended by user.")
    p.disconnect() 