#!/usr/bin/env python3
"""
Script for visualizing retargeted motion files in Genesis.
Simply loads and plays back .npy or .pt motion files on the Go2 robot.
"""
import os
import argparse
import torch
import numpy as np
import genesis as gs
from go2_env import Go2Env

class MotionVisualizer:
    """Class for visualizing retargeted motions on the Go2 robot."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = None
        
        # Environment configurations for Go2 environment
        self.env_cfg = {
            "num_actions": 12,
            "default_joint_angles": {
                "FL_hip_joint": 0.0,
                "FR_hip_joint": 0.0,
                "RL_hip_joint": 0.0,
                "RR_hip_joint": 0.0,
                "FL_thigh_joint": 0.7,
                "FR_thigh_joint": 0.7,
                "RL_thigh_joint": 0.9,
                "RR_thigh_joint": 0.9,
                "FL_calf_joint": -1.4,
                "FR_calf_joint": -1.4,
                "RL_calf_joint": -1.4,
                "RR_calf_joint": -1.4,
            },
            "joint_names": [
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            ],
            "kp": 12.0,
            "kd": 0.3,
            "termination_if_roll_greater_than": 120.0,
            "termination_if_pitch_greater_than": 120.0,
            "base_init_pos": [0.0, 0.0, 0.6],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 60.0,
            "resampling_time_s": 10.0,
            "action_scale": 0.15,
            "simulate_action_latency": False,
            "clip_actions": 100.0,
        }
        
        self.obs_cfg = {
            "num_obs": 45,
            "obs_scales": {
                "lin_vel": 2.0,
                "ang_vel": 0.25,
                "dof_pos": 1.0,
                "dof_vel": 0.05,
            },
        }
        
        self.reward_cfg = {
            "tracking_sigma": 0.25,
            "base_height_target": 0.3,
            "reward_scales": {},
        }
        
        self.command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.0, 0.0],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [0.0, 0.0],
        }
    
    def load_motion(self, motion_file):
        """Load a retargeted motion file (.npy or .pt)."""
        print(f"Loading motion from {motion_file}")
        try:
            # Try loading as torch file first (common for retargeted motions)
            try:
                data = torch.load(motion_file)
                if isinstance(data, dict):
                    joint_angles = data.get('joint_angles')
                    if joint_angles is None:
                        # Try other common keys
                        for key in data:
                            if isinstance(data[key], torch.Tensor) and len(data[key].shape) == 2 and data[key].shape[1] == 12:
                                joint_angles = data[key]
                                print(f"Found joint angles in key: {key}")
                                break
                else:
                    joint_angles = data
            except:
                # Try numpy format if torch fails
                if motion_file.endswith('.npy'):
                    joint_angles = np.load(motion_file)
                    joint_angles = torch.tensor(joint_angles, device=self.device)
                else:
                    raise ValueError("Could not load file as PyTorch or Numpy")
            
            # Ensure we found joint angles
            if joint_angles is None:
                raise ValueError("Could not find joint angles in the file")
                
            # Ensure tensor is on the correct device
            joint_angles = joint_angles.to(device=self.device)
            
            print(f"Loaded motion with {joint_angles.shape[0]} frames")
            return joint_angles
        except Exception as e:
            print(f"Error loading motion file: {e}")
            return None
    
    def visualize_motion(self, joint_angles, playback_time=15.0, loop=True, speed_factor=1.0):
        """Visualize the motion on the Go2 robot."""
        if joint_angles is None:
            print("No motion data to visualize!")
            return
        
        # Initialize environment if not already done
        if self.env is None:
            self.env = Go2Env(
                num_envs=1,
                env_cfg=self.env_cfg,
                obs_cfg=self.obs_cfg,
                reward_cfg=self.reward_cfg,
                command_cfg=self.command_cfg,
                show_viewer=True,
            )
        
        # Reset the environment
        obs, _ = self.env.reset()
        
        # Play back the motion
        num_frames = joint_angles.shape[0]
        print(f"Visualizing {num_frames} frames of motion...")
        
        # Visualization parameters
        frames_per_second = 30  # approximate fps of simulation
        total_frames = int(playback_time * frames_per_second)
        
        print(f"Playing motion for approximately {playback_time} seconds at {speed_factor}x speed...")
        print(f"Press Ctrl+C to stop playback")
        
        try:
            # Play the motion continuously if looping is enabled
            for i in range(total_frames):
                if loop:
                    # Get frame index with looping
                    frame_idx = (int(i * speed_factor) % num_frames)
                else:
                    # Without looping, stop at the end
                    frame_idx = min(int(i * speed_factor), num_frames - 1)
                    if frame_idx >= num_frames - 1 and i > 0:
                        print("Reached end of motion sequence")
                        break
                
                # Get the joint angles for this frame
                action = joint_angles[frame_idx].clone()
                
                # Ensure action has the correct shape
                if len(action.shape) == 1:
                    action = action.unsqueeze(0)  # Add batch dimension
                
                # Step the environment
                try:
                    obs, reward, done, info = self.env.step(action)
                    
                    # Break if done
                    if done:
                        print(f"  Simulation terminated at frame {i+1}/{total_frames}")
                        break
                        
                except Exception as e:
                    print(f"Error on frame {i}: {e}")
                    break
                
                # Progress indicator (don't print too often)
                if i % 30 == 0 or i == total_frames - 1:
                    current_time = i / frames_per_second
                    print(f"  Time: {current_time:.1f}s - Frame {i+1}/{total_frames} - Motion frame: {frame_idx+1}/{num_frames}")
            
            print("Visualization complete")
        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")
    
    def compare_motions(self, motion_files, labels=None):
        """Load and compare multiple motion files."""
        if not motion_files:
            print("No motion files provided for comparison")
            return
        
        motions = []
        for i, file in enumerate(motion_files):
            print(f"Loading motion {i+1}/{len(motion_files)}: {file}")
            motion = self.load_motion(file)
            if motion is not None:
                label = labels[i] if labels and i < len(labels) else f"Motion {i+1}"
                motions.append((motion, label))
        
        if not motions:
            print("No valid motion files loaded")
            return
        
        print("\nMotion comparison:")
        for i, (motion, label) in enumerate(motions):
            print(f"{i+1}. {label}: {motion.shape[0]} frames")
        
        # Ask which motion to visualize
        while True:
            try:
                choice = int(input("\nEnter the number of the motion to visualize (0 to quit): "))
                if choice == 0:
                    break
                elif 1 <= choice <= len(motions):
                    motion, label = motions[choice-1]
                    print(f"\nVisualizing {label}...")
                    self.visualize_motion(motion)
                else:
                    print("Invalid choice. Please enter a number between 1 and", len(motions))
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nComparison interrupted by user")
                break


def main():
    """Main function to visualize retargeted motion."""
    parser = argparse.ArgumentParser(description="Motion visualization for Go2 robot")
    parser.add_argument("--file", type=str, help="Motion file to visualize")
    parser.add_argument("--time", type=float, default=15.0, help="Playback time in seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor")
    parser.add_argument("--no-loop", action="store_true", help="Disable looping of motion")
    parser.add_argument("--compare", nargs='+', help="Compare multiple motion files")
    parser.add_argument("--labels", nargs='+', help="Labels for comparison (must match number of files)")
    
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init()
    
    # Create visualizer
    visualizer = MotionVisualizer()
    
    # Handle comparison mode
    if args.compare:
        visualizer.compare_motions(args.compare, args.labels)
        return
    
    # Handle single file visualization
    if args.file:
        joint_angles = visualizer.load_motion(args.file)
        if joint_angles is not None:
            visualizer.visualize_motion(joint_angles, args.time, not args.no_loop, args.speed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 