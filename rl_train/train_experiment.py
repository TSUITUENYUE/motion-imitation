#!/usr/bin/env python3
"""
Enhanced Motion Imitation with Time-dependent Rewards and Domain Randomization for Go2 robot.
Based on train.py with additions for:
1. Time-dependent reward weighting
2. Continuous motion synthesis
3. Domain randomization
4. Gait transfer learning
"""
import argparse
import os
import pickle
import shutil
from importlib import metadata
import glob
import re

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

# Add wandb import with try/except to handle cases where it's not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb is not installed. W&B logging will be disabled.")
    print("To enable W&B logging, install with: pip install wandb")

from go2_env import Go2Env

class EnhancedMotionImitationEnv(Go2Env):
    """
    Extension of Go2Env with motion imitation rewards focused on joint angles, velocity profile
    and end-effector positions, with domain randomization for robustness.
    """
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, reference_motion=None, reference_velocities=None, show_viewer=False, motion_filename=''):
        # Call parent constructor
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=show_viewer)
        
        # Reference motion data
        self.reference_motion = None
        self.reference_joint_velocities = None
        self.motion_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_motion_frames = 0
        self.motion_filename = motion_filename
        
        # For tracking joint velocities in real-time
        self.last_dof_pos = None
        self.current_joint_velocities = None
        
        # For tracking absolute position progress
        self.last_base_position = None
        self.absolute_position_history = None
        self.position_progress_tracker = torch.zeros(self.num_envs, device=self.device)  # Tracks cumulative progress
        
        # For increasing speed over time
        self.target_speed = torch.ones(self.num_envs, device=self.device) * 0.5  # Initial target speed
        self.speed_progression_rate = env_cfg.get("speed_progression_rate", 0.0002)  # Speed increase per step (slowed down)
        self.max_target_speed = env_cfg.get("max_target_speed", 1.2)  # Maximum target speed (reduced)
        self.straight_motion_tolerance = env_cfg.get("straight_motion_tolerance", 0.05)  # Y-movement tolerance
        
        # Domain randomization config
        self.domain_rand_cfg = env_cfg.get("domain_rand", {
            "enabled": True,
            "friction_range": [0.8, 1.2],
            "restitution_range": [0.0, 0.1],
            "gravity_range": [-10.2, -9.4],
            "mass_range": [0.9, 1.1],
            "randomize_on_reset": True,
        })
        
        # Curriculum learning
        self.use_curriculum = env_cfg.get("use_curriculum", True)
        self.curriculum_step = 0
        self.curriculum_success_threshold = env_cfg.get("curriculum_success_threshold", 0.8)
        self.curriculum_eval_period = env_cfg.get("curriculum_eval_period", 50)
        self.curriculum_success_counter = 0
        self.curriculum_stages = env_cfg.get("curriculum_stages", [
            {"terrain_roughness": 0.0, "push_force": 0.0},
            {"terrain_roughness": 0.05, "push_force": 0.0},
            {"terrain_roughness": 0.05, "push_force": 10.0},
            {"terrain_roughness": 0.1, "push_force": 20.0},
        ])
        self.current_curriculum = self.curriculum_stages[0].copy()
        
        # Gait analysis
        self.gait_period = 0
        self.gait_pattern = None
        
        # Setup reference motion if provided
        if reference_motion is not None:
            if isinstance(reference_motion, np.ndarray):
                self.reference_motion = torch.tensor(reference_motion, device=self.device, dtype=torch.float32)
            elif isinstance(reference_motion, torch.Tensor):
                self.reference_motion = reference_motion.to(device=self.device)
            else:
                print(f"Warning: Reference motion type not recognized: {type(reference_motion)}")
                self.reference_motion = None
                
            if self.reference_motion is not None:
                self.max_motion_frames = self.reference_motion.shape[0]
                print(f"Loaded reference motion with {self.max_motion_frames} frames on device: {self.reference_motion.device}")
                
                # Set reference joint velocities if provided
                if reference_velocities is not None:
                    if isinstance(reference_velocities, np.ndarray):
                        self.reference_joint_velocities = torch.tensor(reference_velocities, device=self.device, dtype=torch.float32)
                    elif isinstance(reference_velocities, torch.Tensor):
                        self.reference_joint_velocities = reference_velocities.to(device=self.device)
                    print(f"Using provided reference joint velocities for training")
                
                # Calculate joint velocities if not provided
                if self.reference_joint_velocities is None:
                    self._calculate_reference_joint_velocities()
                
                self._detect_gait_pattern()
        
        # Initialize tracking of current joint velocities
        self.last_dof_pos = self.dof_pos.clone()
        self.current_joint_velocities = torch.zeros_like(self.dof_pos)
        
        # Initialize position tracking for absolute progress
        self.last_base_position = self.base_pos.clone()
        self.absolute_position_history = torch.zeros((self.num_envs, 10, 3), device=self.device)  # Store last 10 positions
        self.position_history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Initialize reward functions and scales - SIMPLIFIED to just core rewards
        # Joint Pose Matching (Primary)
        self.reward_functions = {}
        self.episode_sums = {}
        
        if self.reference_motion is not None:
            print("Initializing joint_pose_matching reward")
            self.reward_functions["joint_pose_matching"] = self._reward_joint_pose_matching
            self.episode_sums["joint_pose_matching"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

            # Joint Velocity Matching
            print("Initializing joint_velocity_matching reward")
            self.reward_functions["joint_velocity_matching"] = self._reward_joint_velocity_matching
            self.episode_sums["joint_velocity_matching"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

            # End-effector position matching
            print("Initializing end_effector_matching reward")
            self.reward_functions["end_effector_matching"] = self._reward_end_effector_matching
            self.episode_sums["end_effector_matching"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            
            # Ground contact reward
            print("Initializing ground_contact reward")
            self.reward_functions["ground_contact"] = self._reward_ground_contact
            self.episode_sums["ground_contact"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            
            # Forward progression reward 
            print("Initializing forward_progression reward")
            self.reward_functions["forward_progression"] = self._reward_forward_progression
            self.episode_sums["forward_progression"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            
            # Absolute position progress reward
            print("Initializing absolute_position_progress reward")
            self.reward_functions["absolute_position_progress"] = self._reward_absolute_position_progress
            self.episode_sums["absolute_position_progress"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            
            # Straight-line accelerating motion reward
            print("Initializing straight_line_motion reward")
            self.reward_functions["straight_line_motion"] = self._reward_straight_line_motion
            self.episode_sums["straight_line_motion"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            
            # Front leg movement reward
            print("Initializing front_leg_movement reward")
            self.reward_functions["front_leg_movement"] = self._reward_front_leg_movement
            self.episode_sums["front_leg_movement"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Base stability (anti-bouncing)
        print("Initializing stability reward")
        self.reward_functions["stability"] = self._reward_stability
        self.episode_sums["stability"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        
        # Check for reward scales specified in the config but not initialized here
        if hasattr(self, 'reward_scales'):
            for reward_name in self.reward_scales.keys():
                if reward_name not in self.reward_functions:
                    print(f"Creating stub for unused reward: {reward_name}")
                    
                    # Create a dynamic stub method without print spam
                    def make_stub_method(name):
                        def stub_method(self):
                            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
                        return stub_method
                    
                    # Add the stub method to the class and the reward_functions dict
                    stub = make_stub_method(reward_name)
                    setattr(self.__class__, f"_reward_{reward_name}", stub)
                    self.reward_functions[reward_name] = getattr(self, f"_reward_{reward_name}")
                    self.episode_sums[reward_name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Reset obs buffer to match env configuration
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        
        # Debug monitoring enabled for bouncing detection
        self.enable_debug = True
        self.debug_foot_positions = []
        self.bouncing_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Apply physics parameters
        self._setup_physics_parameters()
    
    def _calculate_reference_joint_velocities(self):
        """Calculate reference joint velocities from joint angles for cyclic motion."""
        if self.reference_motion is None or self.max_motion_frames < 2:
            self.reference_joint_velocities = None
            return
            
        print("Calculating joint velocities from reference motion")
        
        # Initialize joint velocities tensor
        joint_velocities = torch.zeros_like(self.reference_motion)
        
        # Calculate velocities as joint angle differences between consecutive frames
        # First frame is special case - it gets the velocity as if continuing from last frame
        joint_velocities[1:] = self.reference_motion[1:] - self.reference_motion[:-1]
        
        # For cyclic motion: calculate first frame velocity as if looping from last to first frame
        joint_velocities[0] = self.reference_motion[0] - self.reference_motion[-1]
        
        # Store for later use
        self.reference_joint_velocities = joint_velocities
        
        # Report velocity magnitude statistics
        vel_magnitudes = torch.norm(joint_velocities, dim=1)
        print(f"Calculated joint velocities (min/max magnitude): {torch.min(vel_magnitudes):.4f} / {torch.max(vel_magnitudes):.4f}")
    
    def _reward_joint_pose_matching(self):
        """Reward for joint angle matching with reference motion."""
        if self.reference_motion is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Get target positions with continuous motion handling
        target_joint_pos = self._get_target_joint_positions()
        
        # Weight different joint types differently - give much more importance to front leg joints
        # Front leg indices
        fr_hip_joint = 0
        fr_thigh_joint = 1
        fr_calf_joint = 2
        fl_hip_joint = 3
        fl_thigh_joint = 4
        fl_calf_joint = 5
        
        # Rear leg indices
        rr_hip_joint = 6
        rr_thigh_joint = 7
        rr_calf_joint = 8
        rl_hip_joint = 9
        rl_thigh_joint = 10
        rl_calf_joint = 11
        
        # Create joint weights tensor
        joint_weights = torch.ones_like(self.dof_pos)
        
        # Front legs - much higher weights
        joint_weights[:, fr_hip_joint] = 3.5   # Increased from default
        joint_weights[:, fr_thigh_joint] = 3.0 # Increased from default
        joint_weights[:, fr_calf_joint] = 2.5  # Increased from default
        joint_weights[:, fl_hip_joint] = 3.5   # Increased from default
        joint_weights[:, fl_thigh_joint] = 3.0 # Increased from default
        joint_weights[:, fl_calf_joint] = 2.5  # Increased from default
        
        # Rear legs - normal weights
        joint_weights[:, rr_hip_joint] = 2.0
        joint_weights[:, rr_thigh_joint] = 1.5
        joint_weights[:, rr_calf_joint] = 1.2
        joint_weights[:, rl_hip_joint] = 2.0
        joint_weights[:, rl_thigh_joint] = 1.5
        joint_weights[:, rl_calf_joint] = 1.2
        
        # Calculate error per joint with higher precision - ensure we're truly matching angles
        joint_errors = torch.square(target_joint_pos - self.dof_pos)
        weighted_joint_error = torch.sum(joint_weights * joint_errors, dim=1)
        
        # Sharper exponential reward for more precise matching
        return torch.exp(-weighted_joint_error / 0.20) # Even sharper falloff for more precise matching
    
    def _reward_joint_velocity_matching(self):
        """Reward for matching joint velocities from reference motion for cyclic gait."""
        if self.reference_joint_velocities is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)
            
        # Update current joint velocities
        if self.last_dof_pos is None:
            self.current_joint_velocities = torch.zeros_like(self.dof_pos)
        else:
            self.current_joint_velocities = self.dof_pos - self.last_dof_pos
        
        # Get target velocity from reference
        ref_indices = torch.clamp(self.motion_frame_idx, 0, self.max_motion_frames - 1)
        target_joint_velocities = self.reference_joint_velocities[ref_indices].to(self.dof_pos.device)
        
        # Calculate velocity matching error - more importance on hip joints
        joint_vel_errors = torch.square(target_joint_velocities - self.current_joint_velocities)
        
        # Weight different joint types differently for velocity matching
        hip_joint_indices = [0, 3, 6, 9]    # FR, FL, RR, RL Hip
        thigh_joint_indices = [1, 4, 7, 10] # Thigh
        calf_joint_indices = [2, 5, 8, 11]  # Calf
        
        joint_weights = torch.ones_like(self.dof_pos)
        joint_weights[:, hip_joint_indices] = 2.5  # Increased emphasis on hip joints (was 1.8)
        joint_weights[:, thigh_joint_indices] = 1.5 # Increased weight on thigh (was 1.2)
        joint_weights[:, calf_joint_indices] = 1.0
        
        weighted_vel_error = torch.sum(joint_weights * joint_vel_errors, dim=1)
        
        # Sharper exponential reward for more precise velocity matching
        return torch.exp(-weighted_vel_error / 0.15) # Was 0.2, sharper falloff
    
    def _reward_end_effector_matching(self):
        """Reward for matching end-effector positions."""
        if self.reference_motion is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)
            
        # Get foot positions from current state
        current_foot_positions = self._get_foot_positions()
        
        # Get target joints with continuous motion handling
        target_joint_pos = self._get_target_joint_positions()
        
        # Focus more on leg extension joints (thigh and calf) which affect foot position
        thigh_calf_indices = [1, 2, 4, 5, 7, 8, 10, 11]  # Thigh and calf joints
        
        # Compute error only on joints most affecting end effector
        joint_errors = torch.square(target_joint_pos[:, thigh_calf_indices] - self.dof_pos[:, thigh_calf_indices])
        weighted_error = torch.sum(joint_errors, dim=1)
        return torch.exp(-weighted_error / 0.3)
    
    def _reward_stability(self):
        """Reward for stable movement without bouncing or falling."""
        # Penalize vertical velocity (anti-bouncing)
        vertical_vel_abs = torch.abs(self.base_lin_vel[:, 2])
        bounce_penalty = torch.exp(-vertical_vel_abs / 0.03) # Sharper penalty for bounce
        
        # Penalize angular velocity (anti-rolling/pitching) except around Z axis
        ang_vel_xy = torch.square(self.base_ang_vel[:, 0]) + torch.square(self.base_ang_vel[:, 1])
        ang_stability = torch.exp(-ang_vel_xy / 0.1)
        
        # Height stability - reward being at appropriate height
        target_height = self.reward_cfg.get("base_height_target", 0.28)
        height_error = torch.abs(self.base_pos[:, 2] - target_height)
        height_reward = torch.exp(-height_error / 0.05)
        
        # Combine components
        return (bounce_penalty + ang_stability + height_reward) / 3.0
    
    def _reward_front_leg_movement(self):
        """
        Reward for encouraging active front leg movement.
        Focuses on hip, thigh, and calf joints of both front legs.
        """
        # Front leg joint indices
        fr_leg_indices = [0, 1, 2]  # FR leg: hip, thigh, calf
        fl_leg_indices = [3, 4, 5]  # FL leg: hip, thigh, calf
        
        # Get joint velocities 
        if self.last_dof_pos is None:
            return torch.ones(self.num_envs, device=self.device)  # Neutral reward on first call
            
        joint_velocities = self.dof_pos - self.last_dof_pos
            
        # Calculate magnitude of front leg joint velocities
        fr_leg_velocity = torch.sum(torch.abs(joint_velocities[:, fr_leg_indices]), dim=1)
        fl_leg_velocity = torch.sum(torch.abs(joint_velocities[:, fl_leg_indices]), dim=1)
        
        # Combine both legs, with more emphasis on lateral movement (hip joints)
        hip_velocity = torch.abs(joint_velocities[:, 0]) + torch.abs(joint_velocities[:, 3])
        combined_velocity = (fr_leg_velocity + fl_leg_velocity) + hip_velocity * 2.5  # Increased emphasis on hip motion
        
        # Reward is higher when there's more front leg movement
        # Scale the reward to be between 0 and 1, with lower threshold to encourage more movement
        min_expected_velocity = 0.005  # Reduced minimum threshold to encourage more movement
        max_expected_velocity = 0.2
        normalized_velocity = torch.clamp((combined_velocity - min_expected_velocity) / 
                                         (max_expected_velocity - min_expected_velocity), 
                                         0.0, 1.0)
        
        # Add stronger bonus for alternating leg movements (walking/running gait)
        leg_coordination_bonus = torch.where(
            torch.sign(joint_velocities[:, 0]) != torch.sign(joint_velocities[:, 3]),
            torch.ones_like(fr_leg_velocity) * 0.5,  # Increased bonus for proper coordination
            torch.zeros_like(fr_leg_velocity)
        )
        
        # Add extra bonus for any front leg movement at all
        any_movement_bonus = torch.where(
            combined_velocity > 0.01,
            torch.ones_like(fr_leg_velocity) * 0.3,  # Bonus just for having movement
            torch.zeros_like(fr_leg_velocity)
        )
        
        # Final reward combines normalized velocity with coordination bonus
        return normalized_velocity + leg_coordination_bonus + any_movement_bonus
    
    def _reward_absolute_position_progress(self):
        """
        Reward for making progress in absolute position (penalizes not moving).
        Tracks the robot's position over time and rewards consistent forward progress.
        Increasingly penalizes side-to-side movement as forward progress increases.
        """
        # Get current position
        current_position = self.base_pos.clone()
        
        # If we don't have a previous position, use current as reference
        if self.last_base_position is None:
            self.last_base_position = current_position.clone()
            return torch.ones(self.num_envs, device=self.device)  # Neutral reward on first call
        
        # Calculate change in position, focus on forward (x-axis) movement
        position_change = current_position - self.last_base_position
        forward_progress = position_change[:, 0]  # X-axis movement
        
        # Update position progress tracker - accumulates forward progress for speed progression
        progress_increase = torch.clamp(forward_progress, 0.0, 0.05)  # Only count positive progress
        self.position_progress_tracker += progress_increase
        
        # Calculate a progress-dependent penalty multiplier for y-direction movement
        # As progress increases, the penalty for side movement becomes much stronger
        # Start with base penalty and increase based on accumulated progress
        progress_penalty_multiplier = 1.0 + torch.clamp(self.position_progress_tracker * 1.0, 0.0, 9.0)  # Increased scaling and max value
        
        # Strong penalty for any side-to-side movement (y-axis), increasing with progress
        side_movement = torch.abs(position_change[:, 1])  # Y-axis movement
        side_movement_penalty = torch.where(
            side_movement > 0.02,  # Reduced tolerance threshold for side movement (was 0.03)
            torch.ones_like(side_movement) * -0.4 * (side_movement / 0.02) * progress_penalty_multiplier,  # Doubled base penalty (-0.2 to -0.4)
            torch.zeros_like(side_movement)
        )
        
        # Update position history buffer
        idx = self.position_history_idx % self.absolute_position_history.shape[1]
        self.absolute_position_history[:, idx] = current_position
        self.position_history_idx += 1
        
        # Calculate longer-term progress (over last ~1 second)
        # Get oldest stored position with a valid index
        history_size = min(self.position_history_idx[0].item(), self.absolute_position_history.shape[1])
        if history_size > 5:  # Only compute if we have enough history
            old_idx = (self.position_history_idx - history_size) % self.absolute_position_history.shape[1]
            old_position = torch.zeros_like(current_position)
            for i in range(self.num_envs):
                old_position[i] = self.absolute_position_history[i, old_idx[i]]
            
            # Calculate medium-term progress
            medium_term_progress = current_position[:, 0] - old_position[:, 0]
            
            # Calculate medium-term side movement - with stronger progress-dependent penalties
            medium_term_side = torch.abs(current_position[:, 1] - old_position[:, 1])
            medium_side_penalty = torch.where(
                medium_term_side > 0.08,  # Reduced tolerance threshold (was 0.1)
                torch.ones_like(medium_term_side) * -0.8 * progress_penalty_multiplier,  # Doubled penalty (was -0.4)
                torch.zeros_like(medium_term_side)
            )
            
            # Strong penalty for lack of forward progress over time
            # We want significant progress over the past 1 second
            min_expected_progress = 0.3  # At least 0.3 meters progress expected
            progress_deficit = torch.clamp(min_expected_progress - medium_term_progress, 0.0, float('inf'))
            
            # Penalty increases exponentially with lack of progress
            progress_penalty = torch.exp(-progress_deficit / 0.1) - 1.0
            
            # Additional penalty for moving backward over the longer term
            backward_penalty = torch.where(medium_term_progress < 0.0, 
                                           torch.ones_like(medium_term_progress) * -0.5, 
                                           torch.zeros_like(medium_term_progress))
        else:
            # Not enough history yet, use short-term only
            progress_penalty = torch.zeros_like(forward_progress)
            backward_penalty = torch.zeros_like(forward_progress)
            medium_side_penalty = torch.zeros_like(forward_progress)
        
        # Short-term immediate penalty for not moving or moving backward
        immediate_penalty = torch.where(forward_progress < 0.01,
                                       torch.ones_like(forward_progress) * -0.2,
                                       torch.zeros_like(forward_progress))
        
        # Update last position for next time
        self.last_base_position = current_position.clone()
        
        # Final reward combines penalties (negative values) with neutral baseline
        return 1.0 + progress_penalty + backward_penalty + immediate_penalty + side_movement_penalty + medium_side_penalty
    
    def _reward_straight_line_motion(self):
        """
        Reward for maintaining straight-line motion with speed based on position progress.
        Only increases speed after position progress has been made.
        Heavily penalizes any side movement.
        """
        # Only update target speed if position progress has been made
        # Use the position_progress_tracker to determine speed increase
        progress_threshold = 1.0  # Meters of cumulative forward progress needed for speed increase
        
        # Apply speed increase only when progression threshold is reached
        increase_speed_mask = self.position_progress_tracker > progress_threshold
        if torch.any(increase_speed_mask):
            # Reset counters for envs that reached the threshold
            self.position_progress_tracker[increase_speed_mask] = 0.0
            
            # Increase target speed for those envs
            self.target_speed[increase_speed_mask] += self.speed_progression_rate * progress_threshold * 100
            self.target_speed = torch.clamp(self.target_speed, 0.5, self.max_target_speed)
        
        # Get current velocities
        forward_vel = self.base_lin_vel[:, 0]  # X-axis velocity
        side_vel = self.base_lin_vel[:, 1]     # Y-axis velocity
        
        # Calculate progression-dependent side movement penalty factor
        # Higher target speed = stricter penalty
        side_penalty_factor = 1.0 + (self.target_speed - 0.5) * 4.0  # Doubled scaling factor (was 2.0)
        
        # Calculate forward velocity matching component - match current target
        vel_diff = torch.abs(forward_vel - self.target_speed)
        forward_match_reward = torch.exp(-vel_diff / 0.3)
        
        # Much stronger penalty for side-to-side motion, increasing with target speed
        side_motion_penalty = torch.where(
            torch.abs(side_vel) > self.straight_motion_tolerance,  # Reduced tolerance (was 2x tolerance)
            (torch.exp(-torch.abs(side_vel) / 0.03) - 1.0) * side_penalty_factor,  # Sharper falloff (was 0.05)
            torch.zeros_like(side_vel)
        )
        
        # Strong penalty for angular velocity around vertical axis (yaw)
        # Also scales with target speed
        yaw_vel = self.base_ang_vel[:, 2]  # Z-axis angular velocity
        yaw_penalty = torch.where(
            torch.abs(yaw_vel) > 0.15,  # Reduced tolerance (was 0.2)
            (torch.exp(-torch.abs(yaw_vel) / 0.07) - 1.0) * side_penalty_factor,  # Sharper falloff (was 0.1)
            torch.zeros_like(yaw_vel)
        )
        
        # Combine all components
        return forward_match_reward + side_motion_penalty + yaw_penalty
    
    def _detect_gait_pattern(self):
        """
        Analyze reference motion to extract gait pattern for continuation.
        Detects gait cycle period and phase relationships between legs.
        """
        if self.reference_motion is None or self.max_motion_frames < 10:
            self.gait_period = 20  # Default
            return
            
        # Simple detection of gait period by looking at leg patterns
        hip_indices = [0, 3, 6, 9]  # FR, FL, RR, RL hip joints
        hip_angles = self.reference_motion[:, hip_indices]
        
        # Find cycle period using autocorrelation
        max_period = min(100, self.max_motion_frames // 2)
        periods = []
        
        for leg_idx in range(4):
            signal = hip_angles[:, leg_idx]
            autocorr = torch.zeros(max_period, device=self.device)
            
            for lag in range(max_period):
                if lag >= self.max_motion_frames:
                    continue
                signal1 = signal[:self.max_motion_frames-lag]
                signal2 = signal[lag:self.max_motion_frames]
                autocorr[lag] = torch.sum(signal1 * signal2)
            
            # Find peaks in autocorrelation
            is_peak = torch.zeros_like(autocorr, dtype=torch.bool)
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    is_peak[i] = True
            
            peak_indices = torch.where(is_peak)[0]
            if len(peak_indices) > 0 and peak_indices[0] > 5:  # Ignore very short periods
                periods.append(peak_indices[0].item())
        
        # Use the median period if available
        if periods:
            self.gait_period = int(sorted(periods)[len(periods)//2])
            print(f"Detected gait period: {self.gait_period} frames")
        else:
            # Default to a reasonable period if detection fails
            self.gait_period = min(20, self.max_motion_frames // 2)
            print(f"Using default gait period: {self.gait_period} frames")
            
        # Optionally, analyze phase relationships between legs (for quadruped gaits)
        # This would store phase offsets between legs for gait generation
        self.gait_pattern = {
            "period": self.gait_period,
            "detected": len(periods) > 0
        }
    
    def _get_target_joint_positions(self):
        """
        Get target joint positions with continuous looping motion.
        Always loops reference motion for perfect cyclic behavior.
        """
        if self.reference_motion is None or self.max_motion_frames == 0:
            return self.dof_pos.clone()  # Fallback to current position
            
        # Always use modulo indexing for perfect looping of cyclic motion
        looped_indices = (self.motion_frame_idx % self.max_motion_frames)
        target_joint_pos = self.reference_motion[looped_indices].clone()
        
        return target_joint_pos
    
    def _get_foot_positions(self):
        """
        Estimate the world positions of the four feet using base position and joint angles.
        Uses a simplified kinematic model based on approximate leg segment lengths.
        """
        # This method remains the same as in the original MotionImitationEnv class
        base_pos = self.base_pos # Shape: (num_envs, 3)
        base_quat = self.base_quat # Shape: (num_envs, 4)
        dof_pos = self.dof_pos # Shape: (num_envs, 12)

        # Approximate leg segment lengths
        l_hip_offset = 0.08
        l_thigh = 0.21
        l_calf = 0.21

        foot_positions_local = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # Joint indices for each leg (hip_y, thigh, calf)
        leg_joint_indices = [
            [0, 1, 2],  # FR leg: FR_hip_y, FR_thigh, FR_calf
            [3, 4, 5],  # FL leg: FL_hip_y, FL_thigh, FL_calf
            [6, 7, 8],  # RR leg: RR_hip_y, RR_thigh, RR_calf
            [9, 10, 11] # RL leg: RL_hip_y, RL_thigh, RL_calf
        ]

        for leg_idx in range(4):
            hip_abduction_angle = dof_pos[:, leg_joint_indices[leg_idx][0]]
            thigh_pitch_angle = dof_pos[:, leg_joint_indices[leg_idx][1]]
            calf_pitch_angle = dof_pos[:, leg_joint_indices[leg_idx][2]]

            # Local X (forward/backward in leg plane)
            local_x = l_thigh * torch.sin(thigh_pitch_angle) + l_calf * torch.sin(thigh_pitch_angle + calf_pitch_angle)
            
            # Local Z (downward/upward in leg plane)
            local_z = - (l_thigh * torch.cos(thigh_pitch_angle) + l_calf * torch.cos(thigh_pitch_angle + calf_pitch_angle))

            # Position hip joints relative to base
            x_hip_body_offset = 0.19 # distance from base center to hip cluster (front/rear)
            y_hip_body_offset = 0.08 # half distance between left/right hips

            hip_origin_in_base_frame = torch.zeros(self.num_envs, 3, device=self.device)
            if leg_idx == 0: # FR
                hip_origin_in_base_frame[:, 0] = x_hip_body_offset
                hip_origin_in_base_frame[:, 1] = y_hip_body_offset
            elif leg_idx == 1: # FL
                hip_origin_in_base_frame[:, 0] = x_hip_body_offset
                hip_origin_in_base_frame[:, 1] = -y_hip_body_offset
            elif leg_idx == 2: # RR
                hip_origin_in_base_frame[:, 0] = -x_hip_body_offset
                hip_origin_in_base_frame[:, 1] = y_hip_body_offset
            elif leg_idx == 3: # RL
                hip_origin_in_base_frame[:, 0] = -x_hip_body_offset
                hip_origin_in_base_frame[:, 1] = -y_hip_body_offset

            # Position foot in base frame
            foot_positions_local[:, leg_idx, 0] = hip_origin_in_base_frame[:, 0] + local_x
            foot_positions_local[:, leg_idx, 1] = hip_origin_in_base_frame[:, 1]
            foot_positions_local[:, leg_idx, 2] = hip_origin_in_base_frame[:, 2] + local_z
        
        # Rotate to world frame (simplified)
        foot_positions_world = torch.zeros_like(foot_positions_local)
        for i in range(self.num_envs):
            q = base_quat[i]
            # Normalize quaternion
            q = q / torch.norm(q)
            
            # Vector part and scalar part of quaternion
            u = q[0:3] 
            s = q[3]
            
            for j in range(4): # For each foot
                # Simplified transformation
                foot_positions_world[i,j,2] = base_pos[i,2] + foot_positions_local[i,j,2]
                foot_positions_world[i,j,0] = base_pos[i,0] + foot_positions_local[i,j,0]
                foot_positions_world[i,j,1] = base_pos[i,1] + foot_positions_local[i,j,1]

        return foot_positions_world 

    def _update_curriculum(self):
        """Update training curriculum based on progress."""
        if not self.use_curriculum:
            return
            
        # Track success rate (e.g., staying upright for X% of episode)
        if hasattr(self, 'episode_length_buf') and hasattr(self, 'progress_buf'):
            success_rate = torch.mean((self.progress_buf / self.episode_length_buf) > 0.8).item()
            
            self.curriculum_success_counter += 1
            if self.curriculum_success_counter >= self.curriculum_eval_period:
                if success_rate > self.curriculum_success_threshold and self.curriculum_step < len(self.curriculum_stages) - 1:
                    self.curriculum_step += 1
                    self.current_curriculum = self.curriculum_stages[self.curriculum_step].copy()
                    print(f"Advancing to curriculum stage {self.curriculum_step+1}: {self.current_curriculum}")
                
                self.curriculum_success_counter = 0
                
                # Apply curriculum changes to environment
                self._apply_curriculum_changes()
    
    def _apply_curriculum_changes(self):
        """Apply changes from current curriculum stage to the environment."""
        # Apply terrain roughness if available
        if "terrain_roughness" in self.current_curriculum:
            roughness = self.current_curriculum["terrain_roughness"]
            if roughness > 0 and hasattr(self, 'scene') and hasattr(self.scene, 'grounds'):
                # This implementation depends on Genesis API for terrain modification
                # For now, just log the change
                print(f"Would apply terrain roughness: {roughness} (API implementation needed)")
                
        # Apply push forces if in curriculum
        if "push_force" in self.current_curriculum:
            self.push_force_magnitude = self.current_curriculum["push_force"]
            if self.push_force_magnitude > 0:
                print(f"Will apply random push forces with magnitude: {self.push_force_magnitude} N")
                
    def _apply_random_push_forces(self):
        """Apply random push forces to the robot if enabled in curriculum."""
        if not hasattr(self, 'push_force_magnitude') or self.push_force_magnitude <= 0:
            return
            
        # Apply pushes at random intervals, to random environments
        if torch.rand(1).item() < 0.01:  # ~1% chance per step
            # Select random environments to push (up to 10% of envs)
            num_to_push = max(1, int(0.1 * self.num_envs))
            push_indices = torch.randint(0, self.num_envs, (num_to_push,), device=self.device)
            
            # Random force direction (in horizontal plane)
            force_angles = torch.rand(num_to_push, device=self.device) * 2 * 3.14159
            force_x = self.push_force_magnitude * torch.cos(force_angles)
            force_y = self.push_force_magnitude * torch.sin(force_angles)
            
            # Apply forces to selected environments
            if hasattr(self, 'apply_force'):
                for i, idx in enumerate(push_indices):
                    # Apply force at robot center of mass, slightly above ground
                    force_pos = self.base_pos[idx].clone()
                    force_pos[2] += 0.1  # Apply slightly above ground
                    self.apply_force(
                        force=[force_x[i].item(), force_y[i].item(), 0],
                        position=force_pos,
                        env_idx=idx.item()
                    )
    
    def _setup_physics_parameters(self):
        """Configure physics parameters with randomization for robustness."""
        if not hasattr(self, 'scene') or self.scene is None:
            return
            
        # Apply base physics parameters
        if hasattr(self.scene, 'physics'):
            try:
                # Set base parameters from env_cfg
                if hasattr(self.scene.physics, 'solver_iterations'):
                    self.scene.physics.solver_iterations = 8
                if hasattr(self.scene.physics, 'enable_stabilization'):
                    self.scene.physics.enable_stabilization = True
                
                # Apply domain randomization if enabled
                if self.domain_rand_cfg.get("enabled", True):
                    self._randomize_physics()
            except Exception as e:
                print(f"Warning: Could not set physics parameters: {e}")
    
    def _randomize_physics(self):
        """Apply domain randomization to physics parameters."""
        if not self.domain_rand_cfg.get("enabled", True):
            return
            
        # Base parameter values
        base_friction = self.env_cfg.get("ground_friction", 1.0)
        base_restitution = self.env_cfg.get("ground_restitution", 0.0)
        base_gravity = -9.81
        
        # Get randomization ranges from config
        friction_range = self.domain_rand_cfg.get("friction_range", [0.8, 1.2])
        restitution_range = self.domain_rand_cfg.get("restitution_range", [0.0, 0.1])
        gravity_range = self.domain_rand_cfg.get("gravity_range", [-10.2, -9.4])
        
        # Apply randomization per environment
        if hasattr(self.scene, 'grounds'):
            friction_rand = torch.rand(self.num_envs, device=self.device)
            friction_values = base_friction * (friction_range[0] + friction_rand * (friction_range[1] - friction_range[0]))
            
            restitution_rand = torch.rand(self.num_envs, device=self.device)
            restitution_values = restitution_range[0] + restitution_rand * (restitution_range[1] - restitution_range[0])
            
            # Apply to ground properties
            # Implementation depends on Genesis API
            for i in range(self.num_envs):
                try:
                    # This is a placeholder - actual implementation depends on Genesis API
                    if hasattr(self.scene.grounds[i], 'set_friction'):
                        self.scene.grounds[i].set_friction(friction_values[i].item())
                    if hasattr(self.scene.grounds[i], 'set_restitution'):
                        self.scene.grounds[i].set_restitution(restitution_values[i].item())
                except:
                    pass  # Handle API compatibility
        
        # Apply randomized gravity
        if hasattr(self.scene, 'physics') and hasattr(self.scene.physics, 'gravity'):
            gravity_rand = torch.rand(1).item()
            gravity_value = gravity_range[0] + gravity_rand * (gravity_range[1] - gravity_range[0])
            try:
                self.scene.physics.gravity = (0, 0, gravity_value)
                print(f"Randomized gravity to {gravity_value:.2f}")
            except Exception as e:
                print(f"Could not set gravity: {e}")
    
    def reset(self):
        """Reset the environment with better initialization for forward motion."""
        # Store the original DOF state before resetting
        original_dof_pos = None
        if hasattr(self, 'dof_pos') and self.dof_pos is not None:
            original_dof_pos = self.dof_pos.clone()
        
        # Reset motion frame indices - use random starting frames for variety
        self.motion_frame_idx = torch.randint(0, self.max_motion_frames, (self.num_envs,), device=self.device)
        
        # Reset target speed and position progress tracker
        self.target_speed = torch.ones(self.num_envs, device=self.device) * 0.5
        self.position_progress_tracker = torch.zeros(self.num_envs, device=self.device)
        
        # Reset curriculum parameters if needed
        if self.use_curriculum:
            self._update_curriculum()
        
        # Apply domain randomization before reset if enabled
        if self.domain_rand_cfg.get("enabled", True) and self.domain_rand_cfg.get("randomize_on_reset", True):
            self._randomize_physics()
            
        # Call parent reset method with careful initialization
        obs = super().reset()
        
        # Apply reference motion's first frame for joint positions
        if self.reference_motion is not None and self.max_motion_frames > 0:
            # Use the motion frame index we just set for variety
            start_joint_pos = self.reference_motion[self.motion_frame_idx]
            
            if hasattr(self, 'dof_pos') and self.dof_pos is not None:
                # Apply the joint positions
                self.dof_pos[:] = start_joint_pos
                if hasattr(self, 'dof_targets'): self.dof_targets[:] = self.dof_pos[:]
                if hasattr(self, 'default_dof_targets'): self.default_dof_targets[:] = self.dof_pos[:]
                if hasattr(self, 'dof_vel'): self.dof_vel[:] = torch.zeros_like(self.dof_vel)
        
        # Reset base position and velocity
        if hasattr(self, 'root_states'):
            # Set position
            self.root_states[:, 0] = 0.0  # X position centered
            self.root_states[:, 1] = 0.0  # Y position centered
            self.root_states[:, 2] = self.env_cfg["base_init_pos"][2]  # Z height from config
            
            # Set velocity - set to match the target velocity from the current frame
            if self.reference_motion is not None:
                adaptive_vel = self._extract_reference_forward_velocity()
                self.root_states[:, 7] = adaptive_vel  # Initial X velocity matched to reference motion
            else:
                self.root_states[:, 7] = 0.6  # Default initial velocity
                
            self.root_states[:, 8:13] = 0.0  # Zero other velocities
        
        # Reset position history for reward calculations
        if hasattr(self, 'base_pos') and self.base_pos is not None:
            self.position_history = self.base_pos.clone()
        
        # Physics stabilization
        if hasattr(self, 'scene') and hasattr(self.scene, 'physics'):
            try:
                if hasattr(self.scene.physics, 'solver_iterations'): self.scene.physics.solver_iterations = 8
                if hasattr(self.scene.physics, 'enable_stabilization'): self.scene.physics.enable_stabilization = True
            except Exception as e:
                print(f"Warning: Could not set advanced physics parameters: {e}")
        
        # Run a few steps to stabilize
        if hasattr(self, 'scene'):
            try:
                for _ in range(10):
                    if hasattr(self.scene, 'step_no_render'): self.scene.step_no_render()
                    elif hasattr(self.scene, 'simulate'): self.scene.simulate(0.01)
            except Exception as e:
                print(f"Warning: Could not run stabilization steps: {e}")
        
        # Update observations
        if hasattr(self, '_compute_observations'):
            try:
                self._compute_observations()
            except: 
                pass
        
        return obs
    
    def _sync_gait_with_commands(self):
        """
        Synchronize the robot's gait pattern with locomotion commands.
        This improves coordination between joint poses and forward motion.
        """
        if not hasattr(self, 'commands'):
            return
            
        # Get forward velocity command
        forward_command = self.commands[:, 0]  # X velocity command
        
        # Determine if we need to sync the gait with command
        needs_sync = torch.any(forward_command > 0.1)
        
        if needs_sync and self.reference_motion is not None and self.max_motion_frames > 0:
            # For environments with low forward velocity, give them a boost
            envs_to_boost = self.base_lin_vel[:, 0] < 0.2
            
            if torch.any(envs_to_boost):
                # Increment the motion frame more quickly for these environments
                # This helps them break out of static positions
                boost_idx = envs_to_boost.nonzero().flatten()
                self.motion_frame_idx[boost_idx] += 1
    
    def step(self, actions):
        """Perform one step with curriculum progression and gait synchronization."""
        # Store previous joint positions for velocity calculation
        self.last_dof_pos = self.dof_pos.clone()

        # Apply random push forces if enabled in curriculum
        self._apply_random_push_forces()
        
        # Sync gait with locomotion commands for better coordination
        self._sync_gait_with_commands()

        # Parent step implementation
        obs, rew_buf, reset_buf, extras = super().step(actions)
        
        # Update motion frame index
        self.motion_frame_idx += 1
        
        # Handle resets
        if torch.any(reset_buf):
            reset_idx = reset_buf.nonzero(as_tuple=False).flatten()
            self.motion_frame_idx[reset_idx] = 0
            
            # Reset joint velocity tracking
            if hasattr(self, 'last_dof_pos'):
                self.last_dof_pos[reset_idx] = self.dof_pos[reset_idx].clone() 
            
            # Reset position tracking for absolute progress
            if hasattr(self, 'last_base_position'):
                self.last_base_position[reset_idx] = self.base_pos[reset_idx].clone()
                self.position_history_idx[reset_idx] = 0
            
        # Update curriculum if enabled
        if self.use_curriculum:
            self._update_curriculum()

        return obs, rew_buf, reset_buf, extras
    
    def close(self):
        """Close the environment, primarily for the visualizer if it's open."""
        if hasattr(self, 'scene') and self.scene is not None:
            if hasattr(self.scene, 'close_viewer') and callable(self.scene.close_viewer):
                try:
                    self.scene.close_viewer()
                    print("Closed scene viewer.")
                except Exception as e:
                    print(f"Exception while closing viewer: {e}")
            # Alternatively, the scene might be closed by its __del__ or a similar mechanism 

    def _reward_ground_contact(self):
        """Reward for ensuring all four feet have proper ground contact during the gait cycle."""
        # Get the foot positions
        foot_positions = self._get_foot_positions()  # Shape: [num_envs, 4, 3]
        
        # Get foot heights (z-coordinate)
        foot_heights = foot_positions[:, :, 2]  # Shape: [num_envs, 4]
        
        # Determine which feet should be on the ground based on gait phase
        # For cyclic gaits, typically diagonal legs contact together
        if self.reference_motion is not None and self.max_motion_frames > 0:
            frame_idx = self.motion_frame_idx % self.max_motion_frames
            
            # Create phase-dependent contact pattern
            # For a canter-like gait:
            # 1. FL and RR should contact in first half of cycle
            # 2. FR and RL should contact in second half
            phase = frame_idx.float() / self.max_motion_frames
            
            # Create target height pattern for each foot based on phase
            # Lower value = closer to ground = better contact
            target_heights = torch.ones_like(foot_heights) * 0.05  # Small clearance for all feet
            
            # Front feet should be on ground for higher reward
            front_feet_idx = [0, 1]  # FR, FL 
            target_heights[:, front_feet_idx] = 0.01  # Very close to ground
            
            # Calculate distance to target height
            height_error = torch.abs(foot_heights - target_heights)
            
            # Stronger penalty for front feet being too high
            height_error[:, front_feet_idx] *= 2.0
            
            # Overall ground contact score
            contact_score = torch.exp(-torch.sum(height_error, dim=1) / 0.1)
            
            return contact_score
        
        # Fallback behavior if no reference motion
        return torch.ones(self.num_envs, device=self.device)
    
    def _extract_reference_forward_velocity(self):
        """
        Calculate the adaptive forward velocity directly from reference motion frames.
        This makes velocity target match exactly with the frame-to-frame motion.
        """
        if self.reference_motion is None or self.max_motion_frames < 2:
            # Default velocity if no reference motion
            return 0.7
            
        # Calculate frame-to-frame velocity directly from joint changes
        # Use current frame index to get a time-appropriate velocity
        current_frame = self.motion_frame_idx % self.max_motion_frames
        next_frame = (current_frame + 1) % self.max_motion_frames
        
        # Focus on hip joint changes which most directly correlate with forward motion
        hip_indices = [0, 3, 6, 9]  # Hip joints
        
        # Get the relevant frames for velocity calculation
        current_hip_angles = self.reference_motion[current_frame][:, hip_indices]
        next_hip_angles = self.reference_motion[next_frame][:, hip_indices]
        
        # Calculate the magnitude of hip angle changes
        angle_diffs = torch.abs(next_hip_angles - current_hip_angles)
        angle_change_magnitude = torch.sum(angle_diffs, dim=1)
        
        # Convert angle change to appropriate velocity
        # Scale factor determined empirically based on gait patterns
        # Higher angle changes = higher velocity
        velocity_scale = 5.0  # Scaling factor to convert angle change to m/s
        adaptive_velocity = angle_change_magnitude * velocity_scale
        
        # Apply reasonable limits
        min_velocity = 0.4  # Minimum forward velocity
        max_velocity = 1.2  # Maximum forward velocity
        adaptive_velocity = torch.clamp(adaptive_velocity, min_velocity, max_velocity)
        
        return adaptive_velocity
    
    def _reward_forward_progression(self):
        """Reward for moving forward based on reference motion velocity."""
        # Get linear velocity in X direction (forward)
        forward_vel = self.base_lin_vel[:, 0]  # X-axis velocity
        
        # Get adaptive target velocity from reference motion
        target_vel = self._extract_reference_forward_velocity()
        
        # Calculate reward based on how close we are to target velocity
        vel_diff = torch.abs(forward_vel - target_vel)
        vel_rew = torch.exp(-vel_diff / 0.3)
        
        # Add stronger bonus specifically for any positive forward motion
        # This helps overcome the initial static position
        forward_bonus = torch.zeros_like(forward_vel)
        forward_bonus = torch.where(forward_vel > 0.1, 
                                  torch.ones_like(forward_vel), 
                                  torch.zeros_like(forward_vel))
        
        # Add extra bonus for getting closer to target velocity
        close_to_target = torch.where(vel_diff < 0.3,
                                    torch.ones_like(forward_vel),
                                    torch.zeros_like(forward_vel))
        
        # Penalize standing still or moving backward
        stationary_penalty = torch.where(forward_vel < 0.05,
                                       torch.ones_like(forward_vel) * 0.5,
                                       torch.zeros_like(forward_vel))
        
        # Penalize side motion (y-velocity)
        side_vel = self.base_lin_vel[:, 1]
        side_vel_rew = torch.exp(-torch.abs(side_vel) / 0.1)
        
        # Penalize rotation except around z-axis (RSL-style)
        ang_vel_xy = torch.square(self.base_ang_vel[:, 0]) + torch.square(self.base_ang_vel[:, 1])
        ang_vel_rew = torch.exp(-ang_vel_xy / 0.05)
        
        # Combine components with more emphasis on forward motion
        return 0.5 * vel_rew + 0.3 * forward_bonus + 0.15 * close_to_target - 0.5 * stationary_penalty + 0.03 * side_vel_rew + 0.02 * ang_vel_rew

    def visualize_rewards(self):
        """Visualize reward components for debugging."""
        with torch.no_grad():
            # Calculate each reward component
            pose_reward = self._reward_joint_pose_matching()
            velocity_reward = self._reward_joint_velocity_matching()
            end_effector_reward = self._reward_end_effector_matching()
            stability_reward = self._reward_stability()
            forward_reward = self._reward_forward_progression()
            ground_contact_reward = self._reward_ground_contact()
            
            # Get scales from reward config
            pose_scale = self.reward_cfg["reward_scales"].get("joint_pose_matching", 0.0)
            velocity_scale = self.reward_cfg["reward_scales"].get("joint_velocity_matching", 0.0)
            end_effector_scale = self.reward_cfg["reward_scales"].get("end_effector_matching", 0.0)
            stability_scale = self.reward_cfg["reward_scales"].get("stability", 0.0)
            forward_scale = self.reward_cfg["reward_scales"].get("forward_progression", 0.0)
            ground_contact_scale = self.reward_cfg["reward_scales"].get("ground_contact", 0.0)
            
            # Calculate weighted rewards
            weighted_pose = pose_reward * pose_scale
            weighted_velocity = velocity_reward * velocity_scale
            weighted_end_effector = end_effector_reward * end_effector_scale
            weighted_stability = stability_reward * stability_scale
            weighted_forward = forward_reward * forward_scale
            weighted_ground_contact = ground_contact_reward * ground_contact_scale
            
            # Calculate total reward
            total_reward = (weighted_pose + weighted_velocity + weighted_end_effector + 
                           weighted_stability + weighted_forward + weighted_ground_contact)
            
            # Print reward breakdown for the first environment
            print("\nReward Components:")
            print(f"  Forward Progression:  {forward_reward[0]:.4f} × {forward_scale:.1f} = {weighted_forward[0]:.4f}")
            print(f"  Ground Contact:       {ground_contact_reward[0]:.4f} × {ground_contact_scale:.1f} = {weighted_ground_contact[0]:.4f}")
            print(f"  Joint Pose Matching:  {pose_reward[0]:.4f} × {pose_scale:.1f} = {weighted_pose[0]:.4f}")
            print(f"  Joint Velocity Match: {velocity_reward[0]:.4f} × {velocity_scale:.1f} = {weighted_velocity[0]:.4f}")
            print(f"  End Effector Match:   {end_effector_reward[0]:.4f} × {end_effector_scale:.1f} = {weighted_end_effector[0]:.4f}")
            print(f"  Stability:            {stability_reward[0]:.4f} × {stability_scale:.1f} = {weighted_stability[0]:.4f}")
            print(f"  Total Reward:         {total_reward[0]:.4f}")
            
            # Print some physics stats
            print("\nPhysics Stats:")
            print(f"  Target Velocity:      {self._extract_reference_forward_velocity():.4f} m/s")
            print(f"  Current Velocity:     {self.base_lin_vel[0, 0]:.4f} m/s")
            print(f"  Base Height:          {self.base_pos[0, 2]:.4f} m")
            
            # Get foot positions to debug ground contact
            foot_positions = self._get_foot_positions()
            print("\nFoot Heights:")
            foot_names = ["FR", "FL", "RR", "RL"]
            for i, name in enumerate(foot_names):
                print(f"  {name} foot:            {foot_positions[0, i, 2]:.4f} m")
            
            print("\n")

class WandbCallback:
    """Callback for logging to Weights & Biases during training."""
    
    def __init__(self, runner, project_name="go2-motion-imitation", experiment_name=None, config=None):
        """Initialize the WandbCallback.
        
        Args:
            runner: The training runner instance
            project_name: Wandb project name
            experiment_name: Name of the experiment
            config: Configuration dictionary to log
        """
        self.runner = runner
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.wandb = None
        
        # Initialize wandb if available
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize wandb run
            wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                resume=config.get("resuming", False) if config else False
            )
            print(f"Initialized wandb logging for project: {self.project_name}, experiment: {self.experiment_name}")
        except ImportError:
            print("Wandb not available. Install with `pip install wandb` for experiment tracking.")
            self.wandb = None
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            self.wandb = None

    def log_iteration(self, iteration, infos):
        """Log metrics to wandb."""
        if self.wandb is None:
            return
        
        # Extract metrics from infos
        metrics = {
            "iteration": iteration,
            "fps": infos["fps"],
            "time": infos["time"],
            "total_time": infos["total_time"],
            "reward/mean": infos["episode_rewards"].mean(),
            "reward/std": infos["episode_rewards"].std(),
            "episode/mean_length": infos["episode_lengths"].mean(),
            "episode/num_completed": infos["num_episodes"],
            "losses/value_loss": infos["value_loss"],
            "losses/policy_loss": infos["policy_loss"],
            "losses/entropy": infos["entropy"],
            "losses/approx_kl": infos["approx_kl"],
            "losses/clip_frac": infos["clip_frac"]
        }
        
        # Add reward component breakdown if available
        if "reward_components" in infos:
            for key, value in infos["reward_components"].items():
                metrics[f"reward_components/{key}"] = value
        
        # Log to wandb
        self.wandb.log(metrics)
    
    def on_iteration_end(self, iteration, infos):
        """Called by the runner at the end of each training iteration."""
        self.log_iteration(iteration, infos)

    def close(self):
        """Close the wandb run."""
        if self.wandb is not None:
            self.wandb.finish()

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

def get_cfgs(with_domain_randomization=True, with_curriculum=True):
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.75, # Slightly more bent to ensure feet can reach ground if base is low
            "FR_thigh_joint": 0.75,
            "RL_thigh_joint": 0.75, 
            "RR_thigh_joint": 0.75,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 50.0, # Moderate stiffness 
        "kd": 0.8,  # Moderate damping
        "termination_if_roll_greater_than": 25, 
        "termination_if_pitch_greater_than": 25,
        "base_init_pos": [0.0, 0.0, 0.26],  # Lowered from 0.28 to ensure better ground contact
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0, 
        "action_scale": 0.25, # Reduced for finer control
        "simulate_action_latency": False,
        "clip_actions": 0.75, # Tighter clipping
        # Physics for ground contact
        "ground_friction": 1.0,
        "ground_restitution": 0.0, # No bounce from ground
        "joint_friction": 0.03,
        "foot_friction": 1.0,
        "enable_stabilizer": True,
        # Disable time-dependent reward weighting
        "use_time_dependent_rewards": False,
    }
    
    # Add domain randomization if enabled
    if with_domain_randomization:
        env_cfg["domain_rand"] = {
            "enabled": True,
            "friction_range": [0.9, 1.1],      # Reduced range for more consistent behavior
            "restitution_range": [0.0, 0.05],  # Reduced range to minimize bouncing
            "gravity_range": [-9.9, -9.7],     # Closer to normal gravity 
            "mass_range": [0.95, 1.05],        # Reduced range for more predictable dynamics
            "randomize_on_reset": True,
        }
    
    # Add curriculum learning if enabled
    if with_curriculum:
        env_cfg["use_curriculum"] = True
        env_cfg["curriculum_success_threshold"] = 0.8
        env_cfg["curriculum_eval_period"] = 50
        env_cfg["curriculum_stages"] = [
            {"terrain_roughness": 0.0, "push_force": 0.0},
            {"terrain_roughness": 0.0, "push_force": 10.0},
            {"terrain_roughness": 0.05, "push_force": 15.0},
            {"terrain_roughness": 0.1, "push_force": 20.0},
        ]
    
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
        "base_height_target": 0.28, # Keep for reference but won't use height reward
        "reward_scales": {
            "tracking_lin_vel": 0.0,  # Set to 0 as we'll use our custom velocity_profile instead
            "tracking_ang_vel": 0.0,  # Not using the default angular velocity tracking
            "lin_vel_z": -2.0,        # Keep strong penalty for vertical velocity (anti-bounce)
            "action_rate": -0.015,    # Small penalty for rapid action changes
            "similar_to_default": 0.0, # Not using default pose matching
            # Custom rewards will be set in main
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.9, 0.9], # Increased target forward speed to 0.9 m/s (was 0.7)
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0], 
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def load_reference_motion(motion_file):
    """Load reference motion file (.npy or .pt)."""
    print(f"Loading reference motion from {motion_file}")
    if not os.path.exists(motion_file):
        print(f"Motion file {motion_file} not found, continuing without reference")
        return None, None
    
    try:
        # Try loading as torch file first
        try:
            data = torch.load(motion_file)
            if isinstance(data, dict):
                # Extract motion data from dictionary
                if 'joint_angles' in data:
                    joint_angles = data['joint_angles']
                    joint_velocities = data.get('joint_velocities', None)
                    print(f"Loaded joint angles and velocity data from dictionary format")
                else:
                    # Try to find tensor data that looks like joint angles
                    joint_angles = None
                    joint_velocities = None
                    
                    for key in data:
                        if isinstance(data[key], torch.Tensor) and len(data[key].shape) == 2 and data[key].shape[1] == 12:
                            joint_angles = data[key]
                            print(f"Found joint angles in key: {key}")
                            break
                    
                    if joint_angles is None:
                        raise ValueError("Could not find joint angles in the file")
            else:
                # Direct tensor - assume it's just joint angles (old format)
                joint_angles = data
                joint_velocities = None
                print("Loaded legacy format with joint angles only")
        except:
            # If torch loading fails, try numpy
            if motion_file.lower().endswith('.npy'):
                data_np = np.load(motion_file, allow_pickle=True)
                
                # Check if data is a dictionary (new format) or array (old format)
                if isinstance(data_np, np.ndarray) and not isinstance(data_np.item(), dict):
                    # Old format - just joint angles
                    joint_angles = torch.tensor(data_np, device=gs.device, dtype=torch.float32)
                    joint_velocities = None
                    print(f"Loaded NPY motion data with shape: {joint_angles.shape} (old format)")
                else:
                    # New format - dictionary with joint angles and velocities
                    data_dict = data_np.item()
                    joint_angles = torch.tensor(data_dict['joint_angles'], device=gs.device, dtype=torch.float32)
                    
                    # Get joint velocities if available
                    if 'joint_velocities' in data_dict:
                        joint_velocities = torch.tensor(data_dict['joint_velocities'], device=gs.device, dtype=torch.float32)
                        print(f"Loaded NPY motion data with shape: {joint_angles.shape} (new format with joint velocities)")
                    else:
                        joint_velocities = None
                        print(f"Loaded NPY motion data with shape: {joint_angles.shape} (new format without joint velocities)")
            else:
                raise ValueError("Could not load file as PyTorch or Numpy")
        
        # Ensure tensors are on the correct device
        joint_angles = joint_angles.to(device=gs.device)
        if joint_velocities is not None:
            joint_velocities = joint_velocities.to(device=gs.device)
        
        print(f"Loaded reference motion with {joint_angles.shape[0]} frames, device: {joint_angles.device}")
        return joint_angles, joint_velocities
    except Exception as e:
        print(f"Error loading motion file: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Train Go2 robot with motion imitation techniques")
    parser.add_argument("--file", type=str, default="data/canter.npy", 
                        help="Path to retargeted motion file (.npy or .pt file)")
    parser.add_argument("--envs", type=int, default=256,
                        help="Number of parallel training environments (default: 256)")
    parser.add_argument("--iters", type=int, default=1000, 
                        help="Maximum number of training iterations (default: 1000)")
    parser.add_argument("--viz", action="store_true", default=False, 
                        help="Enable visualization during training")
    parser.add_argument("--no-wandb", action="store_true", default=False,
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="go2-motion-imitation-enhanced",
                        help="W&B project name (default: go2-motion-imitation-enhanced)")
    parser.add_argument("--no-domain-rand", action="store_true", default=False,
                        help="Disable domain randomization")
    parser.add_argument("--no-curriculum", action="store_true", default=False,
                        help="Disable curriculum learning")
    # Add resume training arguments
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from a checkpoint")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directory of the run to resume (e.g., logs/go2-paper-rewards-canter)")
    parser.add_argument("--checkpoint", type=int, default=-1,
                        help="Checkpoint iteration to load (-1 for latest)")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Handle resuming training
    resuming = False
    resume_run_dir = None
    resume_checkpoint = None
    
    if args.resume:
        # Try to find the run directory and checkpoint
        if args.run_dir is not None:
            # Using provided run directory
            resume_run_dir = args.run_dir
            if not os.path.isdir(resume_run_dir):
                # Try prepending logs/ if just the name was provided
                alt_run_dir = os.path.join("logs", resume_run_dir)
                if os.path.isdir(alt_run_dir):
                    resume_run_dir = alt_run_dir
                else:
                    # Try finding a directory that matches in logs/
                    log_dirs = glob.glob(os.path.join("logs", f"{resume_run_dir}*"))
                    if log_dirs:
                        resume_run_dir = log_dirs[0]
                    else:
                        print(f"Error: Could not find run directory: {args.run_dir}")
                        return
        else:
            # Try to find latest run for the given motion
            motion_basename = os.path.basename(args.file).split('.')[0]
            run_pattern = f"logs/go2-paper-rewards-{motion_basename}*"
            matching_dirs = glob.glob(run_pattern)
            
            if matching_dirs:
                # Sort by modification time (most recent first)
                matching_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                resume_run_dir = matching_dirs[0]
                print(f"Found most recent run directory: {resume_run_dir}")
            else:
                print(f"Error: Could not find any run directories matching {run_pattern}")
                return
        
        # Determine checkpoint to resume from
        if args.checkpoint >= 0:
            # Specific checkpoint requested
            resume_checkpoint = args.checkpoint
        else:
            # Find latest checkpoint
            checkpoints = glob.glob(os.path.join(resume_run_dir, "model_*.pt"))
            if not checkpoints:
                print(f"Error: No checkpoints found in {resume_run_dir}")
                return
                
            # Extract iteration numbers
            iter_numbers = []
            for ckpt in checkpoints:
                match = re.search(r'model_(\d+)\.pt', os.path.basename(ckpt))
                if match:
                    iter_numbers.append((int(match.group(1)), ckpt))
            
            if not iter_numbers:
                print(f"Error: Could not parse checkpoint filenames in {resume_run_dir}")
                return
                
            # Get highest iteration
            iter_numbers.sort(reverse=True)
            resume_checkpoint = iter_numbers[0][0]
            
        print(f"Resuming training from run {resume_run_dir}, checkpoint {resume_checkpoint}")
        resuming = True
            
        # Load the existing configs
        cfg_path = os.path.join(resume_run_dir, "cfgs.pkl")
        if not os.path.exists(cfg_path):
            print(f"Error: Configuration file {cfg_path} not found")
            return
            
        try:
            with open(cfg_path, 'rb') as f:
                env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
                print(f"Loaded configuration from {cfg_path}")
                
                # Update the training iterations
                train_cfg["runner"]["resume"] = True
                train_cfg["runner"]["load_run"] = -1  # This means same directory
                train_cfg["runner"]["checkpoint"] = resume_checkpoint
                train_cfg["runner"]["max_iterations"] = args.iters
                
                # Override run directory
                log_dir = resume_run_dir
                exp_name = train_cfg["runner"]["experiment_name"]
                
                # Use loaded motion filename if available
                if "motion_filename" in env_cfg and os.path.exists(env_cfg["motion_filename"]):
                    args.file = env_cfg["motion_filename"]
                    print(f"Using motion file from config: {args.file}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return
            
        # Load the reference motion
        reference_motion, joint_velocities = load_reference_motion(args.file)
        if reference_motion is None:
            print(f"Error: Could not load motion file {args.file}. Aborting training.")
            return
    else:
        # Standard training setup (not resuming)
        motion_basename = os.path.basename(args.file).split('.')[0]
        exp_name = f"go2-enhanced-{motion_basename}"
        log_dir = f"logs/{exp_name}"
        if os.path.exists(log_dir) and not os.listdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        reference_motion, joint_velocities = load_reference_motion(args.file)
        if reference_motion is None:
            print(f"Error: Could not load motion file {args.file}. Aborting training.")
            return
        
        # Get configurations with options for domain randomization and curriculum
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(
            with_domain_randomization=not args.no_domain_rand,
            with_curriculum=not args.no_curriculum
        )
        train_cfg = get_train_cfg(exp_name, args.iters)

        # Store motion_filename in env_cfg before saving cfgs.pkl
        env_cfg["motion_filename"] = args.file
        
        # Disable time-dependent rewards
        env_cfg["use_time_dependent_rewards"] = False
        
        # Add speed progression parameters
        env_cfg["speed_progression_rate"] = 0.0002  # Speed increase per step (slowed down)
        env_cfg["max_target_speed"] = 1.2  # Maximum target speed (reduced)
        env_cfg["straight_motion_tolerance"] = 0.05  # Y-movement tolerance

        # Configure reward scales to focus on forward motion and joint matching
        reward_cfg["reward_scales"]["forward_progression"] = 7.0       # Keep as is
        reward_cfg["reward_scales"]["joint_pose_matching"] = 12.0      # Significantly increased (was 8.0)
        reward_cfg["reward_scales"]["joint_velocity_matching"] = 9.0   # Significantly increased (was 6.0)
        reward_cfg["reward_scales"]["end_effector_matching"] = 5.0     # Significantly increased (was 3.5)
        reward_cfg["reward_scales"]["stability"] = 0.3                 # Keep as is
        reward_cfg["reward_scales"]["ground_contact"] = 0.5            # Keep as is
        reward_cfg["reward_scales"]["absolute_position_progress"] = 10.0  # Keep as highest priority
        reward_cfg["reward_scales"]["straight_line_motion"] = 3.0      # Keep as is
        reward_cfg["reward_scales"]["front_leg_movement"] = 9.0        # Keep as is
        
        # Remove all other rewards
        for key in list(reward_cfg["reward_scales"].keys()):
            if key not in [
                "joint_pose_matching", "joint_velocity_matching", "forward_progression", 
                "end_effector_matching", "stability", "ground_contact", 
                "absolute_position_progress", "straight_line_motion", "front_leg_movement"
            ]:
                del reward_cfg["reward_scales"][key]
        
        print("Final Reward Scales:", reward_cfg["reward_scales"])

        # Save full configuration
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

    print(f"\nCreating {args.envs} environments with enhanced features...")
    
    # Create stub methods for missing reward functions if resuming training
    if resuming:
        # Check for reward functions that might be required by checkpoint but missing in EnhancedMotionImitationEnv
        for reward_name in ["chassis_height", "leg_symmetry", "gait_continuation"]:
            method_name = f"_reward_{reward_name}"
            if hasattr(EnhancedMotionImitationEnv, method_name):
                continue  # Skip if method already exists
            
            print(f"Adding stub for missing reward method: {method_name}")
            # Define a stub method for the missing reward
            def make_stub_method(name):
                def stub_method(self):
                    # Silent stub to avoid console spam
                    return torch.zeros(self.num_envs, device=self.device)
                return stub_method
            
            # Bind the method to the class
            setattr(EnhancedMotionImitationEnv, method_name, make_stub_method(method_name))
    
    env = EnhancedMotionImitationEnv(
        num_envs=args.envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        reference_motion=reference_motion,
        reference_velocities=joint_velocities,
        show_viewer=args.viz,
        motion_filename=args.file
    )
    
    # Set up viewer for better visualization if enabled
    if args.viz and hasattr(env, 'scene') and hasattr(env.scene, 'viewer'):
        try:
            # Set camera for better viewing angle
            env.scene.viewer.set_camera_pose(
                pos=(1.5, -1.5, 1.0),  # Position camera behind and to the side
                lookat=(0.0, 0.0, 0.3)  # Look at robot's approximate center
            )
            
            # Set window size and position
            if hasattr(env.scene.viewer, 'set_window_size'):
                env.scene.viewer.set_window_size(1280, 720)
            
            # Position window in a good spot on screen
            if hasattr(env.scene.viewer, 'set_window_pos'):
                env.scene.viewer.set_window_pos(50, 50)
                
            print("Viewer configured for better visualization")
        except Exception as e:
            print(f"Warning: Could not fully configure viewer: {e}")
    
    print("Environment created successfully")

    print(f"\nInitializing PPO runner...")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    config = {
        "motion_file": args.file,
        "num_envs": args.envs,
        "max_iterations": args.iters,
        "seed": seed,
        "reward_scales": reward_cfg["reward_scales"],
        "env_cfg": env_cfg, 
        "policy": train_cfg["policy"],
        "algorithm": train_cfg["algorithm"],
        "domain_randomization": env_cfg.get("domain_rand", {}).get("enabled", False),
        "curriculum_learning": env_cfg.get("use_curriculum", False),
        "resuming": resuming,
        "resume_checkpoint": resume_checkpoint if resuming else None
    }
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    wandb_callback = WandbCallback(
        runner=runner,
        project_name=args.wandb_project,
        experiment_name=exp_name,
        config=config
    ) if use_wandb else None
    
    print(f"\n{'='*50}")
    if resuming:
        print(f"Resuming training from checkpoint {resume_checkpoint} for {args.iters} additional iterations")
    else:
        print(f"Starting training for {args.iters} iterations")
    print(f"Motion file: {args.file}")
    print(f"Log directory: {log_dir}")
    print(f"Domain randomization: {'enabled' if env_cfg.get('domain_rand', {}).get('enabled', False) else 'disabled'}")
    print(f"Curriculum learning: {'enabled' if env_cfg.get('use_curriculum', False) else 'disabled'}")
    print(f"Time-dependent rewards: {'enabled' if env_cfg.get('use_time_dependent_rewards', False) else 'disabled'}")
    if use_wandb:
        print(f"W&B logging enabled for project: {args.wandb_project}")
    else:
        if args.no_wandb:
            print("W&B logging disabled via --no-wandb flag")
        elif not WANDB_AVAILABLE:
            print("W&B logging disabled (package not installed)")
    print(f"{'='*50}\n")
    
    if use_wandb and wandb_callback:
        runner.custom_callback = wandb_callback.on_iteration_end
    
    try:
        # Start learning
        runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Clean up wandb
        if wandb_callback:
            wandb_callback.close()
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 