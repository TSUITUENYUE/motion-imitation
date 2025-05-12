#!/usr/bin/env python3
"""
MuJoCo Runner for Go2 Policy

This script loads a trained policy model and runs it to control the Go2 robot in MuJoCo.
It handles observation processing, action execution, and visualization.
"""
import os
import numpy as np
import torch
import time
import argparse
import mujoco
import mujoco.viewer
from genesis_to_mujoco import Go2PolicyNetwork, observation_normalizer

# MuJoCo model path for Go2 robot
GO2_XML_PATH = "unitree_models/unitree_mujoco/unitree_robots/go2/scene.xml"

class MuJocoGo2Runner:
    """MuJoCo runner for the Go2 robot with the trained policy."""
    
    def __init__(self, model_path, xml_path=GO2_XML_PATH):
        """
        Initialize the MuJoCo runner.
        
        Args:
            model_path: Path to the policy model file
            xml_path: Path to the MuJoCo XML model file
        """
        self.model_path = model_path
        self.xml_path = xml_path
        
        # Load policy model
        self.policy = self.load_policy_model(model_path)
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Add ground plane
        self.add_ground_plane()
        
        # Set initial pose
        self.reset_robot()
        
        print(f"Initialized MuJoCo Go2 Runner with model from {model_path}")
        print(f"MuJoCo model: {xml_path}")
        print(f"Number of DoFs: {self.model.nv}")
        print(f"Number of joints: {self.model.njnt}")
        
    def add_ground_plane(self):
        """Add a ground plane to the MuJoCo scene."""
        # Simply try to load the scene XML which includes ground
        scene_xml_path = os.path.join(os.path.dirname(self.xml_path), "scene.xml")
        if os.path.exists(scene_xml_path) and "scene.xml" not in self.xml_path:
            print(f"Loading scene from {scene_xml_path} which includes a ground plane")
            self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
            self.data = mujoco.MjData(self.model)
        else:
            # Check if we're already using a scene.xml
            if "scene.xml" in self.xml_path:
                print("Using scene.xml which should include a ground plane")
            else:
                print("Warning: Could not find scene.xml with ground plane. Robot may fall.")
        
    def load_policy_model(self, model_path, input_dim=45, output_dim=12):
        """
        Load the policy model.
        
        Args:
            model_path: Path to the policy model file
            input_dim: Dimension of input observations
            output_dim: Dimension of output actions
            
        Returns:
            policy: Loaded policy model
        """
        # Create policy network
        policy = Go2PolicyNetwork(input_dim, output_dim)
        
        # Load state dict
        policy.load_state_dict(torch.load(model_path))
        
        # Set to evaluation mode
        policy.eval()
        
        return policy
    
    def reset_robot(self):
        """Reset the robot to the initial standing pose."""
        # Reset data
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial position (standing)
        self.data.qpos[:] = 0.0
        
        # Set initial height (z coordinate of the base)
        self.data.qpos[2] = 0.42  # Approximate Go2 standing height
        
        # Set initial joint positions for a standing pose
        # The exact indices will depend on the XML model
        # Format is usually: [root_pos(3), root_quat(4), joint_positions...]
        joint_indices = np.arange(7, 7+12)  # Adjust based on your model
        self.data.qpos[joint_indices] = 0.0  # Neutral position
        
        # Forward kinematics to update the robot state
        mujoco.mj_forward(self.model, self.data)
    
    def get_observation(self):
        """
        Get the current observation from MuJoCo.
        This should be formatted to match the inputs expected by the policy.
        
        The PPO policy expects 45-dimensional observations structured as:
        - Base position (3 values)
        - Base orientation quaternion (4 values)
        - Joint positions (12 values)
        - Joint velocities (12 values)
        - Base linear velocity (3 values)
        - Base angular velocity (3 values)
        - Additional proprioception (8 values) - padded with zeros
        
        Returns:
            observation: Observation tensor for the policy
        """
        # 1. Base position (3 values)
        base_pos = self.data.qpos[:3].copy()
        
        # 2. Base orientation quaternion (4 values)
        base_quat = self.data.qpos[3:7].copy()
        
        # 3. Joint positions (12 values)
        joint_pos = self.data.qpos[7:7+12].copy()
        
        # 4. Joint velocities (12 values)
        joint_vel = self.data.qvel[6:6+12].copy()
        
        # 5. Base linear velocity (3 values)
        base_lin_vel = self.data.qvel[:3].copy()
        
        # 6. Base angular velocity (3 values)
        base_ang_vel = self.data.qvel[3:6].copy()
        
        # Construct observation vector (items 1-6, total 37 values)
        observation = np.concatenate([
            base_pos, base_quat,
            joint_pos, joint_vel,
            base_lin_vel, base_ang_vel
        ])
        
        # 7. Pad with zeros to match the expected dimension of 45
        # In Genesis, these 8 additional values likely contained additional proprioceptive information
        padded_observation = np.zeros(45, dtype=np.float32)
        padded_observation[:observation.shape[0]] = observation
        
        # Convert to tensor
        observation_tensor = torch.tensor(padded_observation, dtype=torch.float32).unsqueeze(0)
        
        return observation_tensor
    
    def apply_action(self, action):
        """
        Apply the action to the robot in MuJoCo.
        
        Args:
            action: Action tensor from the policy
        """
        # Convert action tensor to numpy array
        action_np = action.squeeze().numpy()
        
        # Apply action to the robot's actuators
        self.data.ctrl[:] = action_np
        
    def step_simulation(self, n_steps=1):
        """
        Step the MuJoCo simulation forward.
        
        Args:
            n_steps: Number of simulation steps to take
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
    
    def run(self, duration=30.0, render=True):
        """
        Run the policy on the robot in MuJoCo.
        
        Args:
            duration: Duration to run in seconds
            render: Whether to render the simulation
        """
        if render:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Set camera parameters
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -20
                
                print(f"Running simulation for {duration} seconds...")
                
                # Simulation loop
                start_time = time.time()
                while time.time() - start_time < duration:
                    # Get observation
                    observation = self.get_observation()
                    
                    # Get action from policy
                    with torch.no_grad():
                        action = self.policy(observation)
                    
                    # Apply action
                    self.apply_action(action)
                    
                    # Step simulation (5 steps per render frame for smoother motion)
                    self.step_simulation(5)
                    
                    # Update viewer
                    viewer.sync()
                    
                    # Sleep to control simulation speed
                    time.sleep(0.01)
        else:
            # Run without rendering
            print(f"Running simulation for {duration} seconds...")
            
            # Simulation loop
            start_time = time.time()
            while time.time() - start_time < duration:
                # Get observation
                observation = self.get_observation()
                
                # Get action from policy
                with torch.no_grad():
                    action = self.policy(observation)
                
                # Apply action
                self.apply_action(action)
                
                # Step simulation
                self.step_simulation(5)
                
                # Sleep to control simulation speed
                time.sleep(0.01)
        
        print("Simulation complete!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MuJoCo Runner for Go2 Policy")
    parser.add_argument("--model", type=str, default="mujoco_models/go2_policy.pt",
                        help="Path to the trained policy model")
    parser.add_argument("--xml", type=str, default=GO2_XML_PATH,
                        help="Path to the MuJoCo XML model file")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration to run the simulation in seconds")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    args = parser.parse_args()
    
    # Create runner
    runner = MuJocoGo2Runner(args.model, args.xml)
    
    # Run simulation
    runner.run(args.duration, not args.no_render)


if __name__ == "__main__":
    main() 