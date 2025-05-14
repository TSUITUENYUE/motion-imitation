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
    Extension of Go2Env with enhanced motion imitation rewards.
    Adds time-dependent reward weighting, continuous motion synthesis,
    domain randomization, and gait pattern analysis.
    """
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, reference_motion=None, show_viewer=False, motion_filename=''):
        # Call parent constructor
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=show_viewer)
        
        # Reference motion data
        self.reference_motion = None
        self.motion_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_motion_frames = 0
        self.motion_filename = motion_filename
        self.reference_velocities = None
        
        # Domain randomization config
        self.domain_rand_cfg = env_cfg.get("domain_rand", {
            "enabled": True,
            "friction_range": [0.8, 1.2],
            "restitution_range": [0.0, 0.1],
            "gravity_range": [-10.2, -9.4],
            "mass_range": [0.9, 1.1],
            "randomize_on_reset": True,
        })
        
        # Dynamic reward weighting parameters
        self.use_time_dependent_rewards = env_cfg.get("use_time_dependent_rewards", True)
        self.imitation_decay_rate = env_cfg.get("imitation_decay_rate", 5.0)  # Higher = faster decay
        self.robustness_rise_rate = env_cfg.get("robustness_rise_rate", 8.0)  # Higher = faster rise
        
        # Time-dependent reward weights
        self.imitation_weight = torch.ones((self.num_envs, 1), device=self.device)
        self.robustness_weight = torch.zeros((self.num_envs, 1), device=self.device)
        
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
                self._estimate_reference_velocities()
                self._detect_gait_pattern()
        
        # Initialize reward functions and scales
        # Joint Pose Matching (Primary)
        if self.reference_motion is not None:
            print("Initializing joint_pose_matching reward")
            self.reward_functions["joint_pose_matching"] = self._reward_joint_pose_matching
            self.episode_sums["joint_pose_matching"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

            # Velocity Profile Matching
            print("Initializing velocity_profile reward")
            self.reward_functions["velocity_profile"] = self._reward_velocity_profile
            self.episode_sums["velocity_profile"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

            # Leg Symmetry
            print("Initializing leg_symmetry reward")
            self.reward_functions["leg_symmetry"] = self._reward_leg_symmetry
            self.episode_sums["leg_symmetry"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

            # End-effector position matching
            print("Initializing end_effector_matching reward")
            self.reward_functions["end_effector_matching"] = self._reward_end_effector_matching
            self.episode_sums["end_effector_matching"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Forward Motion
        print("Initializing forward_motion reward")
        self.reward_functions["forward_motion"] = self._reward_forward_motion
        self.episode_sums["forward_motion"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.position_history = self.base_pos.clone()

        # Chassis Height
        print("Initializing chassis_height reward")
        self.reward_functions["chassis_height"] = self._reward_chassis_height
        self.episode_sums["chassis_height"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        
        # Ground Contact
        print("Initializing ground_contact reward")
        self.reward_functions["ground_contact"] = self._reward_ground_contact
        self.episode_sums["ground_contact"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Gait Continuation
        print("Initializing gait_continuation reward")
        self.reward_functions["gait_continuation"] = self._reward_gait_continuation
        self.episode_sums["gait_continuation"] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Reset obs buffer to match env configuration
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        
        # Debug monitoring enabled for bouncing detection
        self.enable_debug = True
        self.debug_foot_positions = []
        self.bouncing_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Apply physics parameters
        self._setup_physics_parameters()
        
        # Initial reward weights computation
        self._compute_reward_weights()
    
    def _compute_reward_weights(self):
        """
        Compute dynamic reward weights based on current progress in motion sequence.
        Early phase: emphasize imitation
        Late phase: emphasize robust locomotion
        """
        if not self.use_time_dependent_rewards:
            self.imitation_weight = torch.ones((self.num_envs, 1), device=self.device)
            self.robustness_weight = torch.ones((self.num_envs, 1), device=self.device)
            return
            
        # Calculate progress through reference motion (0 to 1)
        if self.reference_motion is not None and self.max_motion_frames > 0:
            # Divide by 2x motion length to extend the imitation phase longer
            progress = torch.clamp(self.motion_frame_idx.float() / (self.max_motion_frames * 2), 0, 1)
            
            # Use a more gradual transition (linear instead of exponential)
            imitation_weight = 1.0 - (progress * 0.7)  # Only decay to 0.3, not all the way to 0
            robustness_weight = 0.3 + (progress * 0.7)  # Start at 0.3, rise to 1.0
            
            # Create weight tensors for all environments
            self.imitation_weight = imitation_weight.unsqueeze(1)  # Shape for broadcasting
            self.robustness_weight = robustness_weight.unsqueeze(1)
        else:
            # No reference motion, focus on robustness
            self.imitation_weight = torch.zeros((self.num_envs, 1), device=self.device)
            self.robustness_weight = torch.ones((self.num_envs, 1), device=self.device)
    
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
    
    def _estimate_reference_velocities(self):
        if self.reference_motion is None or self.max_motion_frames < 2:
            self.reference_velocities = None
            return
            
        velocities = torch.zeros(self.max_motion_frames, device=self.device)
        # Simplified: assume hip joints reflect forward movement
        hip_indices = [0, 3, 6, 9] # FR, FL, RR, RL hip
        for i in range(1, self.max_motion_frames):
            prev_frame_hips = self.reference_motion[i-1, hip_indices]
            curr_frame_hips = self.reference_motion[i, hip_indices]
            hip_diff = torch.mean(torch.abs(curr_frame_hips - prev_frame_hips))
            velocities[i-1] = hip_diff * 5.0 # Heuristic scaling factor
        
        velocities[-1] = velocities[-2] # Extrapolate last frame
        
        # Smooth and scale
        kernel_size = 3
        if self.max_motion_frames > kernel_size:
            smoothed = torch.zeros_like(velocities)
            for i in range(self.max_motion_frames):
                start = max(0, i - kernel_size // 2)
                end = min(self.max_motion_frames, i + kernel_size // 2 + 1)
                smoothed[i] = torch.mean(velocities[start:end])
            velocities = smoothed
        
        min_vel, max_vel = torch.min(velocities), torch.max(velocities)
        if max_vel > min_vel:
             # Scale to a typical walking speed range, e.g., 0.3 to 0.8 m/s for this reference
            self.reference_velocities = 0.3 + (velocities - min_vel) * (0.5 / (max_vel - min_vel))
        else:
            self.reference_velocities = torch.full_like(velocities, 0.5) # Default if no variation
        print(f"Estimated reference velocity profile (min/max): {torch.min(self.reference_velocities):.2f} / {torch.max(self.reference_velocities):.2f} m/s") 

    def _get_target_joint_positions(self):
        """
        Get target joint positions with motion synthesis after reference motion ends.
        Either loops reference motion or generates synthetic continuation based on gait pattern.
        """
        if self.reference_motion is None or self.max_motion_frames == 0:
            return self.dof_pos.clone()  # Fallback to current position
            
        # For environments within reference motion length, use actual reference
        ref_indices = torch.clamp(self.motion_frame_idx, 0, self.max_motion_frames - 1)
        target_joint_pos = self.reference_motion[ref_indices].clone()
        
        # For environments that have exceeded reference motion length
        exceeded_mask = (self.motion_frame_idx >= self.max_motion_frames).unsqueeze(1)
        if torch.any(exceeded_mask):
            # Loop the reference motion (better for cyclic gaits)
            if self.gait_pattern is not None and self.gait_pattern["detected"]:
                # Use detected gait period for more precise looping
                adjusted_indices = ((self.motion_frame_idx - self.max_motion_frames) % self.gait_period) + (self.max_motion_frames - self.gait_period)
                looped_positions = self.reference_motion[adjusted_indices]
            else:
                # Simple looping if no gait detected
                looped_indices = (self.motion_frame_idx % self.max_motion_frames)
                looped_positions = self.reference_motion[looped_indices]
            
            # Apply the looped positions where needed
            target_joint_pos = torch.where(exceeded_mask, looped_positions, target_joint_pos)
        
        return target_joint_pos
    
    def _reward_joint_pose_matching(self):
        """Reward for joint angle matching with reference motion, weighted by time progress."""
        if self.reference_motion is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Get target positions with continuous motion handling
        target_joint_pos = self._get_target_joint_positions()
        
        hip_joint_indices = [0, 3, 6, 9]    # FR, FL, RR, RL Hip
        thigh_joint_indices = [1, 4, 7, 10] # Thigh
        calf_joint_indices = [2, 5, 8, 11]  # Calf
        
        joint_weights = torch.ones_like(self.dof_pos)
        joint_weights[:, hip_joint_indices] = 1.5 # Emphasize hip joints
        joint_weights[:, thigh_joint_indices] = 1.2
        joint_weights[:, calf_joint_indices] = 1.0
        
        joint_errors = torch.square(target_joint_pos - self.dof_pos)
        weighted_joint_error = torch.sum(joint_weights * joint_errors, dim=1)
        
        # Add a stronger penalty for large errors to prevent accumulating deviation
        cumulative_error_threshold = 0.8
        cumulative_error_penalty = torch.where(
            weighted_joint_error > cumulative_error_threshold,
            -2.0 * (weighted_joint_error - cumulative_error_threshold),
            torch.zeros_like(weighted_joint_error)
        )
        
        # Base reward combines exponential part and error penalty
        base_reward = torch.exp(-weighted_joint_error / 0.4) + cumulative_error_penalty
        
        # Scale by imitation weight if time-dependent rewards are enabled
        return base_reward * self.imitation_weight.squeeze()
    
    def _reward_velocity_profile(self):
        """Reward for matching velocity profile from reference, weighted by time progress."""
        if self.reference_velocities is None:
            return torch.ones(self.num_envs, device=self.device) # Neutral reward
            
        ref_indices = torch.clamp(self.motion_frame_idx, 0, self.max_motion_frames - 1)
        target_velocity = self.reference_velocities[ref_indices].to(self.base_lin_vel.device)
        current_velocity = self.base_lin_vel[:, 0] # Forward velocity (x-axis)
        
        vel_error = torch.abs(current_velocity - target_velocity)
        base_reward = torch.exp(-vel_error / 0.25) # Adjusted sigma
        
        # Scale by imitation weight
        return base_reward * self.imitation_weight.squeeze()
    
    def _reward_leg_symmetry(self):
        """Reward for maintaining appropriate leg symmetry, weighted by time progress."""
        if self.reference_motion is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)

        # Get target positions with continuous motion handling
        target_joint_pos = self._get_target_joint_positions()

        target_front_hip_diff = target_joint_pos[:, 0] - target_joint_pos[:, 3]
        current_front_hip_diff = self.dof_pos[:, 0] - self.dof_pos[:, 3]
        front_hip_error = torch.square(current_front_hip_diff - target_front_hip_diff)

        target_rear_hip_diff = target_joint_pos[:, 6] - target_joint_pos[:, 9]
        current_rear_hip_diff = self.dof_pos[:, 6] - self.dof_pos[:, 9]
        rear_hip_error = torch.square(current_rear_hip_diff - target_rear_hip_diff)
        
        symmetry_error = front_hip_error + rear_hip_error
        base_reward = torch.exp(-symmetry_error / 0.15) # Adjusted sigma
        
        # Scale by imitation weight
        return base_reward * self.imitation_weight.squeeze()
    
    def _reward_end_effector_matching(self):
        """Reward for matching end-effector positions, weighted by time progress."""
        if self.reference_motion is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)
            
        # Get foot positions from current state
        current_foot_positions = self._get_foot_positions()
        
        # We need to compute feet positions for target joint angles
        # This is complex and would require duplicating FK logic
        # For simplicity, we'll use a proxy reward:
        
        # Get target joints with continuous motion handling
        target_joint_pos = self._get_target_joint_positions()
        
        # Focus more on leg extension joints (thigh and calf) which affect foot position
        thigh_calf_indices = [1, 2, 4, 5, 7, 8, 10, 11]  # Thigh and calf joints
        
        # Compute error only on joints most affecting end effector
        joint_errors = torch.square(target_joint_pos[:, thigh_calf_indices] - self.dof_pos[:, thigh_calf_indices])
        weighted_error = torch.sum(joint_errors, dim=1)
        base_reward = torch.exp(-weighted_error / 0.3)  # Adjusted sigma
        
        # Scale by imitation weight
        return base_reward * self.imitation_weight.squeeze()
        
    def _reward_forward_motion(self):
        """Reward for forward motion, weighted more as reference motion ends."""
        forward_vel = self.base_lin_vel[:, 0]
        vertical_vel_abs = torch.abs(self.base_lin_vel[:, 2])
        target_fwd_vel = self.command_cfg["lin_vel_x_range"][0]
        
        # Reward for achieving or maintaining forward velocity
        # Positive reward scaled by how close to target, penalty for being slow/backwards
        fwd_progress_reward = torch.where(forward_vel > 0.05, 
                                        1.0 - torch.clamp(torch.abs(forward_vel - target_fwd_vel) / target_fwd_vel, 0, 1),
                                        -1.0 + forward_vel * 10) # Penalize no/reverse motion

        # Strong penalty for vertical velocity (bouncing)
        bounce_penalty_val = vertical_vel_abs
        bounce_penalty_reward = torch.exp(-bounce_penalty_val / 0.03) # Sharper penalty for bounce
        
        # Combine: prioritize forward progress, strongly penalize bouncing
        base_reward = fwd_progress_reward * 0.6 + (bounce_penalty_reward -1.0) * 0.4
        
        # Scale by robustness weight - more important as reference motion progresses
        return base_reward * self.robustness_weight.squeeze()

    def _reward_chassis_height(self):
        """Reward for maintaining appropriate chassis height, weighted by time progress."""
        base_height = self.base_pos[:, 2]
        target_height = self.reward_cfg["base_height_target"] 
        height_error = torch.abs(base_height - target_height)
        
        # Allow small deviations, penalize larger ones. Sigma of 0.03 means +-3cm is tolerable.
        height_reward = torch.exp(-height_error / 0.03)
        # Add a stronger penalty if significantly below target (e.g. > 5cm below)
        too_low_penalty = torch.where(base_height < (target_height - 0.05), -1.0, 0.0)
        base_reward = height_reward + too_low_penalty
        
        # Scale by robustness weight - more important as reference motion progresses
        return base_reward * self.robustness_weight.squeeze()

    def _reward_ground_contact(self):
        """Reward for appropriate foot contact patterns, weighted by time progress."""
        foot_positions = self._get_foot_positions() # Shape: (num_envs, 4, 3)
        foot_heights_z = foot_positions[:, :, 2]
        
        # Average foot height error from ground
        avg_foot_height_error = torch.mean(torch.abs(foot_heights_z), dim=1)
        # Reward for feet being close to ground
        foot_on_ground_reward = torch.exp(-avg_foot_height_error / 0.02)

        # Penalize if any foot is too high
        max_foot_height = torch.max(foot_heights_z, dim=1)[0]
        floating_penalty = torch.where(max_foot_height > 0.05, -1.0 * (max_foot_height - 0.05) / 0.1, 0.0)
        
        # Penalize vertical base velocity (anti-bouncing)
        base_vertical_vel_abs = torch.abs(self.base_lin_vel[:, 2])
        stability_penalty = -2.0 * base_vertical_vel_abs

        # Combine rewards
        base_reward = foot_on_ground_reward + floating_penalty + stability_penalty
        
        # Scale by robustness weight - more important as reference motion progresses
        return base_reward * self.robustness_weight.squeeze()
    
    def _reward_gait_continuation(self):
        """Reward for continuing the gait pattern after reference motion ends."""
        if self.reference_motion is None or self.max_motion_frames == 0 or self.gait_period == 0:
            return torch.zeros(self.num_envs, device=self.device)
            
        # Only apply this reward when beyond reference motion
        beyond_ref = (self.motion_frame_idx >= self.max_motion_frames)
        
        if not torch.any(beyond_ref):
            return torch.zeros(self.num_envs, device=self.device)
        
        # For each environment that's beyond reference, check if the current motion
        # maintains the gait frequency and pattern
        
        # Simple approach: reward left-right alternation at appropriate frequency
        left_hip_pos = self.dof_pos[:, 3] # Front left hip
        right_hip_pos = self.dof_pos[:, 0] # Front right hip
        
        # Current frame within gait cycle
        cycle_phase = ((self.motion_frame_idx - self.max_motion_frames) % self.gait_period) / self.gait_period
        
        # Approximation: in quadruped gaits, opposite legs are often 180° out of phase
        # If we're in the first half of the gait cycle, left should be ahead of right, and vice versa
        phase_appropriate = torch.where(
            cycle_phase < 0.5,
            left_hip_pos > right_hip_pos,  # First half: left forward
            right_hip_pos > left_hip_pos   # Second half: right forward
        )
        
        # Simple binary reward for now
        gait_reward = torch.where(phase_appropriate, 0.5, -0.1)
        
        # Apply only to environments beyond reference motion
        masked_reward = torch.where(beyond_ref, gait_reward, torch.zeros_like(gait_reward))
        
        # Always apply full weight to this reward (it only triggers after reference motion ends)
        return masked_reward

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
        """Reset the environment with domain randomization and initial motion frame."""
        # Store the original DOF state before resetting
        original_dof_pos = None
        if hasattr(self, 'dof_pos') and self.dof_pos is not None:
            original_dof_pos = self.dof_pos.clone()
        
        # Reset motion frame indices
        self.motion_frame_idx.zero_()
        
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
            first_frame = self.reference_motion[0].clone()
            if hasattr(self, 'dof_pos') and self.dof_pos is not None:
                self.dof_pos[:] = first_frame.unsqueeze(0).repeat(self.num_envs, 1)
                if hasattr(self, 'dof_targets'): self.dof_targets[:] = self.dof_pos[:]
                if hasattr(self, 'default_dof_targets'): self.default_dof_targets[:] = self.dof_pos[:]
                if hasattr(self, 'dof_vel'): self.dof_vel[:] = torch.zeros_like(self.dof_vel)
        
        # Reset base position and velocity
        if hasattr(self, 'root_states'):
            self.root_states[:, 2] = self.env_cfg["base_init_pos"][2]
            self.root_states[:, 7:13] = 0.0
        
        # Reset position history for reward calculations
        if hasattr(self, 'base_pos') and self.base_pos is not None:
            self.position_history = self.base_pos.clone()
        
        # Reset initial reward weights
        self._compute_reward_weights()
        
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
    
    def step(self, actions):
        """Perform one step with time-dependent rewards and curriculum progression."""
        # Store previous base position for forward motion reward
        if hasattr(self, 'base_pos') and self.base_pos is not None:
            self.position_history = self.base_pos.clone()
        else:
            if hasattr(self, 'root_states') and self.root_states is not None:
                 self.position_history = self.root_states[:, 0:3].clone()

        # Apply random push forces if enabled in curriculum
        self._apply_random_push_forces()

        # Parent step implementation
        obs, rew_buf, reset_buf, extras = super().step(actions)
        
        # Update motion frame index and compute new reward weights
        self.motion_frame_idx += 1
        self._compute_reward_weights()
        
        # Handle resets
        if torch.any(reset_buf):
            reset_idx = reset_buf.nonzero(as_tuple=False).flatten()
            self.motion_frame_idx[reset_idx] = 0
            if hasattr(self, 'position_history') and self.base_pos is not None: 
                self.position_history[reset_idx] = self.base_pos[reset_idx].clone()
            elif hasattr(self, 'root_states') and self.root_states is not None: 
                self.position_history[reset_idx] = self.root_states[reset_idx, 0:3].clone()
            
            # Reset reward weights for reset environments
            self._compute_reward_weights()
            
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
        "base_init_pos": [0.0, 0.0, 0.28],  # Start low to encourage ground contact
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
        # Time-dependent reward weighting
        "use_time_dependent_rewards": True,
        "imitation_decay_rate": 1.5,  # Reduced from 5.0 (slower decay = better motion matching)
        "robustness_rise_rate": 2.0,  # Reduced from 8.0 (slower rise = less focus on walking)
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
        "base_height_target": 0.28, # Target height for walking
        "reward_scales": {
            "tracking_lin_vel": 0.4, 
            "tracking_ang_vel": 0.2, 
            "lin_vel_z": -2.0, # Strong penalty for vertical base velocity (anti-bounce)
            "action_rate": -0.015, 
            "similar_to_default": -0.03, 
            # Custom rewards will be scaled in main
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5], # Target moderate forward speed
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0], 
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def load_reference_motion(motion_file):
    """Load reference motion file (.npy or .pt)."""
    print(f"Loading reference motion from {motion_file}")
    if not os.path.exists(motion_file):
        print(f"Motion file {motion_file} not found, continuing without reference")
        return None
    
    try:
        # Try loading as torch file first
        try:
            data = torch.load(motion_file)
            if isinstance(data, dict):
                # Extract joint angles from dictionary
                if 'joint_angles' in data:
                    reference_motion = data['joint_angles']
                    print(f"Loaded joint angles from key 'joint_angles'")
                else:
                    # Try to find tensor data that looks like joint angles
                    reference_motion = None
                    for key in data:
                        if isinstance(data[key], torch.Tensor) and len(data[key].shape) == 2 and data[key].shape[1] == 12:
                            reference_motion = data[key]
                            print(f"Found joint angles in key: {key}")
                            break
                    
                    if reference_motion is None:
                        raise ValueError("Could not find joint angles in the file")
            else:
                # Direct tensor
                reference_motion = data
        except:
            # If torch loading fails, try numpy
            if motion_file.lower().endswith('.npy'):
                data_np = np.load(motion_file)
                reference_motion = torch.tensor(data_np, device=gs.device, dtype=torch.float32)
                print(f"Loaded NPY motion data with shape: {reference_motion.shape}")
            else:
                raise ValueError("Could not load file as PyTorch or Numpy")
        
        # Ensure tensor is on the correct device
        reference_motion = reference_motion.to(device=gs.device)
        
        print(f"Loaded reference motion with {reference_motion.shape[0]} frames, device: {reference_motion.device}")
        return reference_motion
    except Exception as e:
        print(f"Error loading motion file: {e}")
        return None

# Custom TensorBoard callback that logs to both TensorBoard and W&B if enabled
class WandbCallback:
    def __init__(self, runner, project_name="go2-motion-imitation", experiment_name=None, config=None):
        self.runner = runner
        self.use_wandb = WANDB_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config or {},
                sync_tensorboard=True,
                monitor_gym=False,
            )
            print(f"W&B logging initialized for project: {project_name}")
    
    def log_iteration(self, iteration, infos):
        if not self.use_wandb:
            return
            
        metrics = {}
        if "episode" in infos:
            for k, v in infos["episode"].items():
                if isinstance(v, (int, float)):
                    metrics[f"episode/{k}"] = v
        if "losses" in infos:
            for k, v in infos["losses"].items():
                if isinstance(v, (int, float)):
                    metrics[f"loss/{k}"] = v
        if "learning_rate" in infos:
            metrics["train/learning_rate"] = infos["learning_rate"]
        metrics["train/iteration"] = iteration
        wandb.log(metrics)
    
    def on_iteration_end(self, iteration, infos):
        """Called by the runner at the end of each training iteration."""
        self.log_iteration(iteration, infos)

    def close(self):
        if self.use_wandb:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train Go2 robot with enhanced motion imitation techniques")
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
    # New arguments for motion fidelity
    parser.add_argument("--imitation-decay", type=float, default=1.5,
                        help="Imitation reward decay rate (default: 1.5, lower = better matching)")
    parser.add_argument("--robustness-rise", type=float, default=2.0,
                        help="Robustness reward rise rate (default: 2.0, lower = slower transition)")
    parser.add_argument("--motion-focus", action="store_true", default=False,
                        help="Prioritize motion matching over robustness (increases imitation rewards)")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    motion_basename = os.path.basename(args.file).split('.')[0]
    exp_name = f"go2-enhanced-{motion_basename}"
    log_dir_basename = f"go2-enhanced-{motion_basename}"
    log_dir = f"logs/{log_dir_basename}"
    if os.path.exists(log_dir) and not os.listdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    reference_motion = load_reference_motion(args.file)
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
    
    # Apply command-line arguments for motion fidelity parameters
    env_cfg["imitation_decay_rate"] = args.imitation_decay
    env_cfg["robustness_rise_rate"] = args.robustness_rise

    # Configure reward scales with an emphasis on blending imitation and robustness
    if args.motion_focus:
        # Further increase imitation rewards for extreme motion fidelity
        reward_cfg["reward_scales"]["joint_pose_matching"] = 7.0
        reward_cfg["reward_scales"]["end_effector_matching"] = 6.0 
        reward_cfg["reward_scales"]["leg_symmetry"] = 3.0
        print("Motion focus mode enabled: Increased imitation reward weights")
    else:
        # Use the already-increased values from earlier edit
        reward_cfg["reward_scales"]["joint_pose_matching"] = 5.0
        reward_cfg["reward_scales"]["ground_contact"] = 3.5
        reward_cfg["reward_scales"]["forward_motion"] = 2.0
        reward_cfg["reward_scales"]["chassis_height"] = 1.0
        reward_cfg["reward_scales"]["velocity_profile"] = 1.0
        reward_cfg["reward_scales"]["leg_symmetry"] = 2.0
        reward_cfg["reward_scales"]["end_effector_matching"] = 4.0
        reward_cfg["reward_scales"]["gait_continuation"] = 1.5

    # Remove default base_height if custom chassis_height is used
    if "base_height" in reward_cfg["reward_scales"]:
         del reward_cfg["reward_scales"]["base_height"]
    
    print("Final Reward Scales:", reward_cfg["reward_scales"])
    
    # Save full configuration
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    print(f"\nCreating {args.envs} environments with enhanced features...")
    env = EnhancedMotionImitationEnv(
        num_envs=args.envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        reference_motion=reference_motion,
        show_viewer=args.viz,
        motion_filename=args.file
    )
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
        "domain_randomization": not args.no_domain_rand,
        "curriculum_learning": not args.no_curriculum,
    }
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    wandb_callback = WandbCallback(
        runner=runner,
        project_name=args.wandb_project,
        experiment_name=exp_name,
        config=config
    ) if use_wandb else None
    
    print(f"\n{'='*50}")
    print(f"Starting enhanced training for {args.iters} iterations")
    print(f"Motion file: {args.file}")
    print(f"Log directory: {log_dir}")
    print(f"Domain randomization: {'enabled' if not args.no_domain_rand else 'disabled'}")
    print(f"Curriculum learning: {'enabled' if not args.no_curriculum else 'disabled'}")
    print(f"Time-dependent rewards: enabled")
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