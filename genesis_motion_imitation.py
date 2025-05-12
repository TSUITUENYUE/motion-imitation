#!/usr/bin/env python3
"""
Genesis Motion Imitation with PyTorch for Go2 robot.
Based on go2_train.py structure for better compatibility with Genesis ecosystem.
Incorporates PPO from rsl-rl-lib with joint matching rewards.
"""
import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import torch
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from go2_env import Go2Env
from RL_training.multi_dog_walk_retarget import MultiDogMotionRetargeter, MOCAP_MOTIONS

class MotionImitationEnv(Go2Env):
    """Extension of Go2Env with motion imitation rewards."""
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, reference_motion=None, show_viewer=False):
        # Call parent constructor
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=show_viewer)
        
        # Reference motion data
        self.reference_motion = None
        self.motion_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_motion_frames = 0
        
        # Setup reference motion if provided
        if reference_motion is not None:
            if isinstance(reference_motion, np.ndarray):
                # Convert numpy array to tensor and move to the device
                self.reference_motion = torch.tensor(reference_motion, device=self.device, dtype=torch.float32)
                print(f"Converted numpy reference motion to tensor on device: {self.reference_motion.device}")
            elif isinstance(reference_motion, torch.Tensor):
                # Ensure tensor is on the correct device
                self.reference_motion = reference_motion.to(device=self.device)
                print(f"Moved reference motion tensor to device: {self.reference_motion.device}")
            else:
                print(f"Warning: Reference motion type not recognized: {type(reference_motion)}")
                self.reference_motion = None
                
            if self.reference_motion is not None:
                self.max_motion_frames = self.reference_motion.shape[0]
                print(f"Loaded reference motion with {self.max_motion_frames} frames on device: {self.reference_motion.device}")
        
        # Add joint matching rewards if we have reference motion
        if self.reference_motion is not None:
            print("Adding joint matching rewards")
            self.reward_scales["joint_pose_matching"] = 1.0
            self.reward_functions["joint_pose_matching"] = self._reward_joint_pose_matching
        
        # Reset obs buffer to match env configuration
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
    
    def _reward_joint_pose_matching(self):
        """Reward for matching joint angles to reference motion."""
        if self.reference_motion is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Get target joint positions from reference motion
        ref_indices = self.motion_frame_idx % self.max_motion_frames
        
        # Ensure indices are on the same device as the reference motion
        if ref_indices.device != self.reference_motion.device:
            ref_indices = ref_indices.to(self.reference_motion.device)
            
        target_joint_pos = self.reference_motion[ref_indices]
        
        # Ensure target and dof_pos are on same device
        if target_joint_pos.device != self.dof_pos.device:
            target_joint_pos = target_joint_pos.to(self.dof_pos.device)
        
        # Calculate joint position error (MSE)
        joint_error = torch.sum(torch.square(target_joint_pos - self.dof_pos), dim=1)
        
        # Use negative exponential to convert error to reward
        return torch.exp(-joint_error / 1.0)  # Sigma = 1.0
    
    def step(self, actions):
        # Call parent step method
        obs, rew_buf, reset_buf, extras = super().step(actions)
        
        # Update motion frame index for each environment
        self.motion_frame_idx += 1
        
        # Reset motion frame on environment reset
        if torch.any(reset_buf):
            reset_idx = reset_buf.nonzero(as_tuple=False).flatten()
            self.motion_frame_idx[reset_idx] = 0
        
        return obs, rew_buf, reset_buf, extras

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict

def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            # joint_pose_matching will be added if reference motion is provided
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def load_reference_motion(motion_file):
    """Load reference motion file if provided."""
    print(f"Loading reference motion from {motion_file}")
    if not os.path.exists(motion_file):
        # Check if it's a named motion
        walk_motion = None
        for motion in MOCAP_MOTIONS:
            if motion[0] == motion_file:
                walk_motion = motion
                break
        
        if walk_motion:
            print(f"Using predefined motion: {walk_motion[0]}")
            retargeter = MultiDogMotionRetargeter()
            motion_name, file_path, frame_start, frame_end = walk_motion
            retargeter.load_motion_data(file_path, frame_start, frame_end)
            reference_motion, _ = retargeter.retarget_sequence()
            return reference_motion
        else:
            print(f"Motion file {motion_file} not found, continuing without reference")
            return None
    else:
        try:
            # Check if it's an NPY file
            if motion_file.lower().endswith('.npy'):
                print(f"Loading NPY file: {motion_file}")
                data_np = np.load(motion_file)
                # Convert numpy array to torch tensor and move to the proper device
                reference_motion = torch.tensor(data_np, device=gs.device, dtype=torch.float32)
                print(f"Loaded NPY motion data with shape: {reference_motion.shape}")
            else:
                # Try loading as a PyTorch file
                print(f"Loading PyTorch file: {motion_file}")
                data = torch.load(motion_file)
                if isinstance(data, dict):
                    reference_motion = data['joint_angles']
                else:
                    reference_motion = data
                
                # Ensure tensor is on the correct device
                reference_motion = reference_motion.to(device=gs.device)
            
            print(f"Loaded reference motion with {reference_motion.shape[0]} frames, device: {reference_motion.device}")
            return reference_motion
        except Exception as e:
            print(f"Error loading motion file: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-motion-imitation")
    parser.add_argument("-B", "--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--visualize", action="store_true", default=False, 
                        help="Enable visualization (default: False)")
    parser.add_argument("--motion_file", type=str, default="walk", 
                        help="Reference motion file or name (e.g., walk, trot)")
    parser.add_argument("--joint_reward_weight", type=float, default=1.0,
                        help="Weight for joint pose matching reward")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    # Initialize Genesis with minimal logging
    gs.init(logging_level="warning")

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set up log directory
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Load reference motion if provided
    reference_motion = None
    if args.motion_file:
        reference_motion = load_reference_motion(args.motion_file)
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    # Adjust reward weights for joint matching if needed
    if reference_motion is not None and "joint_pose_matching" not in reward_cfg["reward_scales"]:
        reward_cfg["reward_scales"]["joint_pose_matching"] = args.joint_reward_weight
    
    # Save configurations
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Create environment
    print(f"\nCreating {args.num_envs} environments...")
    env = MotionImitationEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        reference_motion=reference_motion,
        show_viewer=args.visualize
    )
    print("Environment created successfully")

    # Create and start PPO runner
    print(f"\nInitializing PPO runner...")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    print(f"\n{'='*50}")
    print(f"Starting training for {args.max_iterations} iterations")
    print(f"{'='*50}\n")
    
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 