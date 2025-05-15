#!/usr/bin/env python3
"""
Script for visualizing retargeted motion files in JSON format with Genesis.
Displays the Go2 robot executing the motion from canter.txt, trot.txt, etc.
"""
import os
import argparse
import json
import torch
import numpy as np
import genesis as gs
from go2_env import Go2Env
import time

# Initialize genesis
gs.init()

class RetargetedMotionVisualizer:
    """Class for visualizing JSON format retargeted motions on the Go2 robot."""
    
    def __init__(self):
        """Initialize the visualizer."""
        # Initialize device - we'll use the device set by genesis
        # after initialization
        gs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = None
        
        # Environment configurations for Go2 environment
        self.env_cfg = {
            "num_actions": 12,
            "default_joint_angles": {
                "FL_hip_joint": 0.1,
                "FL_thigh_joint": 0.8,
                "FL_calf_joint": -1.5,
                "FR_hip_joint": 0.1,
                "FR_thigh_joint": 0.8, 
                "FR_calf_joint": -1.5,
                "RL_hip_joint": 0.1,
                "RL_thigh_joint": 0.8,
                "RL_calf_joint": -1.5, 
                "RR_hip_joint": 0.1,
                "RR_thigh_joint": 0.8,
                "RR_calf_joint": -1.5
            },
            "joint_names": [
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
            ],
            "kp": 60.0,
            "kd": 5.0,
            "clip_actions": 1.0,
            "action_scale": 0.5,
            "episode_length_s": 20,
            "base_init_pos": [0.0, 0.0, 0.32],
            "base_init_quat": [0.0, 0.0, 0.0, 1.0],
            "termination_if_pitch_greater_than": 10.0,  # Increased to avoid early termination
            "termination_if_roll_greater_than": 10.0,   # Increased to avoid early termination
            "resampling_time_s": 10.0
        }
        
        # Configure observation and reward settings
        self.obs_cfg = {
            "num_obs": 45,
            "obs_scales": {
                "lin_vel": 2.0,
                "ang_vel": 0.25,
                "dof_pos": 1.0,
                "dof_vel": 0.05
            }
        }
        
        self.reward_cfg = {
            "reward_scales": {
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.5,
                "lin_vel_z": -2.0,
                "base_height": -2.0,
                "action_rate": -0.005,
                "similar_to_default": -0.01
            },
            "tracking_sigma": 0.25,
            "base_height_target": 0.32
        }
        
        self.command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [-1.0, 1.0],
            "lin_vel_y_range": [-1.0, 1.0],
            "ang_vel_range": [-1.0, 1.0]
        }

    def load_motion_file(self, filename):
        """
        Load JSON format motion data from file.
        
        Args:
            filename: Path to the motion file
            
        Returns:
            Motion data as dictionary
        """
        with open(filename, 'r') as f:
            motion_data = json.load(f)
        return motion_data
    
    def create_environment(self):
        """
        Create the Go2 environment for visualization.
        
        Returns:
            Go2Env instance
        """
        print("Creating environment...")
        # Create environment with 1 instance and enable viewer
        self.env = Go2Env(
            num_envs=1,
            env_cfg=self.env_cfg,
            obs_cfg=self.obs_cfg,
            reward_cfg=self.reward_cfg,
            command_cfg=self.command_cfg,
            show_viewer=True
        )
        
        # Adjust camera for better view
        self.env.scene.viewer.set_camera_pose(
            pos=(1.5, -1.5, 1.0),
            lookat=(0.0, 0.0, 0.3)
        )
        
        return self.env
    
    def visualize_motion(self, motion_data, duration=20, speed=1.0):
        """
        Visualize the motion data on the Go2 robot.
        
        Args:
            motion_data: Dictionary containing motion data
            duration: Duration of visualization in seconds
            speed: Playback speed multiplier
        """
        if self.env is None:
            self.create_environment()
        
        frames = motion_data["Frames"]
        frame_duration = motion_data["FrameDuration"]
        num_frames = len(frames)
        
        # Convert frames to torch tensors for easier processing
        tensor_frames = []
        for frame in frames:
            tensor_frames.append(torch.tensor(frame, device=gs.device))
            
        # Calculate joint velocities from consecutive frames to understand natural motion
        joint_velocities = []
        for i in range(1, len(tensor_frames)):
            # Extract joint angles from current and previous frame
            curr_joints = tensor_frames[i][7:19]  # Joint angles in frame
            prev_joints = tensor_frames[i-1][7:19]  # Joint angles in previous frame
            
            # Calculate velocity (joint change per frame)
            velocity = curr_joints - prev_joints
            joint_velocities.append(velocity)
        
        # Add first velocity (loop from last to first for cyclic motion)
        first_joints = tensor_frames[0][7:19]
        last_joints = tensor_frames[-1][7:19]
        first_velocity = first_joints - last_joints
        joint_velocities.insert(0, first_velocity)
        
        # Print velocity statistics
        if joint_velocities:
            all_velocities = torch.stack(joint_velocities)
            mean_velocity = torch.mean(torch.abs(all_velocities), dim=0)
            max_velocity = torch.max(torch.abs(all_velocities), dim=0)[0]
            
            print("\nJoint Velocity Analysis (per frame):")
            print(f"Mean absolute joint velocity: {mean_velocity}")
            print(f"Max absolute joint velocity: {max_velocity}")
            print(f"Overall mean velocity magnitude: {torch.mean(torch.abs(all_velocities)):.4f}")
            print(f"Overall max velocity magnitude: {torch.max(torch.abs(all_velocities)):.4f}")
            
            # Calculate expected forward velocity
            # Assuming X-axis motion corresponds to forward movement
            pos_diffs = []
            for i in range(1, len(tensor_frames)):
                pos_diff = tensor_frames[i][0] - tensor_frames[i-1][0]  # X-position difference
                pos_diffs.append(pos_diff)
            
            if pos_diffs:
                mean_x_velocity = torch.mean(torch.tensor(pos_diffs))
                print(f"\nForward Motion Analysis:")
                print(f"Mean X-axis velocity: {mean_x_velocity:.4f} units per frame")
                print(f"Estimated forward velocity: {mean_x_velocity/frame_duration:.4f} units per second")
        
        # Reset the environment to start with a clean slate
        self.env.reset()
        
        # Calculate ground height offset by analyzing the lowest point in animation
        # This helps ensure the dog stays on the ground
        min_height = float('inf')
        for frame in tensor_frames:
            root_pos = frame[:3]
            min_height = min(min_height, root_pos[2])
        
        # Add a small offset to ensure the dog is always above ground
        ground_offset = 0.32 - min_height + 0.01  # 0.32 is the typical Go2 height, add 1cm safety margin
        print(f"Calculated ground offset: {ground_offset:.4f}m")
        
        # Process frames to correct directions - this is now the default behavior
        print("Correcting motion direction...")
        # Create a copy of the frames with adjusted positions and orientations
        processed_frames = []
        
        # Find the maximum x position across all frames to determine the range of motion
        max_x = max([frame[0].item() for frame in tensor_frames])
        
        for frame in tensor_frames:
            new_frame = frame.clone()
            
            # 1. Negate the X position component (to reverse the direction)
            new_frame[0] = -frame[0] + max_x  # Reverse X and offset by max to keep it positive
            
            # 2. Rotate the quaternion by 180 degrees around the Z axis
            # Extract original quaternion [x,y,z,w]
            x, y, z, w = frame[3], frame[4], frame[5], frame[6]
            
            # For 180-degree rotation around Z axis
            new_frame[3] = -x  # Negate x
            new_frame[4] = -y  # Negate y
            new_frame[5] = z   # Keep z
            new_frame[6] = -w  # Negate w for proper quaternion
            
            # 3. Swap left and right legs for proper orientation
            # Check if we have joint angles in the frame
            if len(frame) >= 19:  # We expect 7 (pose) + 12 (joints)
                # Joints should be organized as:
                # [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, 
                #  RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
                
                # Swap front legs (FL and FR)
                temp_FL = frame[7:10].clone()  # FL joints
                temp_FR = frame[10:13].clone()  # FR joints
                new_frame[7:10] = temp_FR  # Set FL to FR
                new_frame[10:13] = temp_FL  # Set FR to FL
                
                # Swap rear legs (RL and RR)
                temp_RL = frame[13:16].clone()  # RL joints
                temp_RR = frame[16:19].clone()  # RR joints
                new_frame[13:16] = temp_RR  # Set RL to RR
                new_frame[16:19] = temp_RL  # Set RR to RL
            
            processed_frames.append(new_frame)
        
        # Replace the original frames with the corrected ones
        tensor_frames = processed_frames
        print("Motion direction corrected and leg positions swapped for proper orientation")
        
        # Determine the total number of iterations based on duration and frame_duration
        adjusted_frame_duration = frame_duration / speed
        total_iterations = int(duration / adjusted_frame_duration)
        
        print(f"Visualizing motion with {num_frames} frames over {duration} seconds")
        print(f"Frame duration: {frame_duration} seconds, speed multiplier: {speed}")
        print(f"Press Ctrl+C to exit")
        
        # Main visualization loop
        last_time = time.time()
        
        for i in range(total_iterations):
            current_time = time.time()
            elapsed = current_time - last_time
            
            # Sleep to maintain correct playback rate
            if elapsed < adjusted_frame_duration:
                time.sleep(adjusted_frame_duration - elapsed)
            
            # Get the appropriate frame index, wrapping around if needed
            frame_idx = i % num_frames
            frame = tensor_frames[frame_idx]
            
            # Extract root position (first 3 values) and add ground offset
            root_pos = frame[:3].clone()
            root_pos[2] += ground_offset  # Apply ground offset to Z-coordinate
            
            # Extract root orientation quaternion (next 4 values) - format [x, y, z, w]
            root_quat = frame[3:7].clone()
            
            # Extract joint angles (remaining values, should be 12 values)
            joint_angles = frame[7:].clone()
            
            # Set the robot's position and orientation
            self.env.robot.set_pos(root_pos.unsqueeze(0), zero_velocity=False)
            self.env.robot.set_quat(root_quat.unsqueeze(0), zero_velocity=False)
            
            # Correctly apply joint angles - directly set the joint positions
            self.env.robot.set_dofs_position(
                position=joint_angles.unsqueeze(0),
                dofs_idx_local=self.env.motors_dof_idx,
                zero_velocity=False
            )
            
            # Step the simulation to update the visualization
            self.env.scene.step()
            
            # Print frame info for debugging
            if frame_idx == 0 or frame_idx % 10 == 0:
                print(f"Frame {frame_idx+1}/{num_frames}")
                print(f"Root position: {root_pos}")
                print(f"Root quaternion: {root_quat}")
                print(f"Joint angles: {joint_angles}")
            
            last_time = time.time()

def main():
    parser = argparse.ArgumentParser(description="Visualize retargeted motion files")
    parser.add_argument("--file", type=str, required=True, help="Path to the motion file")
    parser.add_argument("--time", type=float, default=20.0, help="Duration of visualization in seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()
    
    visualizer = RetargetedMotionVisualizer()
    motion_data = visualizer.load_motion_file(args.file)
    
    try:
        visualizer.visualize_motion(motion_data, args.time, args.speed)
    except KeyboardInterrupt:
        print("Visualization stopped by user")

if __name__ == "__main__":
    main() 