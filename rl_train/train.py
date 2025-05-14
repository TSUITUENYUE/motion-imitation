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

# Add wandb import with try/except to handle cases where it's not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb is not installed. W&B logging will be disabled.")
    print("To enable W&B logging, install with: pip install wandb")

from go2_env import Go2Env

class MotionImitationEnv(Go2Env):
    """Extension of Go2Env with motion imitation rewards."""
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, reference_motion=None, show_viewer=False, motion_filename=''):
        # Call parent constructor
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=show_viewer)
        
        # Reference motion data
        self.reference_motion = None
        self.motion_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_motion_frames = 0
        self.motion_filename = motion_filename
        self.reference_velocities = None
        
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
        
        # Initialize reward functions and scales
        # These will be populated based on the main function's reward_cfg
        
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

        # Reset obs buffer to match env configuration
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        
        # Debug monitoring enabled for bouncing detection
        self.enable_debug = True
        self.debug_foot_positions = []
        self.bouncing_detected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Apply physics parameters
        self._setup_physics_parameters()
    
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
            # Consider average change in hip angles as a proxy for forward velocity
            # This is a heuristic and might need tuning or a more sophisticated approach
            # A positive change could mean swinging forward, a negative backward.
            # We're interested in the magnitude of change that contributes to forward propulsion.
            # Taking mean of absolute differences might be too simplistic.
            # Let's consider the *sum* of changes that would typically result in forward motion
            # For now, a simpler proxy: average absolute difference scaled.
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

    def _reward_joint_pose_matching(self):
        if self.reference_motion is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)
        
        ref_indices = self.motion_frame_idx % self.max_motion_frames
        target_joint_pos = self.reference_motion[ref_indices].to(self.dof_pos.device)
        
        hip_joint_indices = [0, 3, 6, 9]    # FR, FL, RR, RL Hip
        thigh_joint_indices = [1, 4, 7, 10] # Thigh
        calf_joint_indices = [2, 5, 8, 11]  # Calf
        
        joint_weights = torch.ones_like(self.dof_pos)
        joint_weights[:, hip_joint_indices] = 1.5 # Emphasize hip joints
        joint_weights[:, thigh_joint_indices] = 1.2
        joint_weights[:, calf_joint_indices] = 1.0
        
        joint_errors = torch.square(target_joint_pos - self.dof_pos)
        weighted_joint_error = torch.sum(joint_weights * joint_errors, dim=1)
        return torch.exp(-weighted_joint_error / 0.4) # Sigma adjusted for error scale

    def _reward_velocity_profile(self):
        if self.reference_velocities is None:
            return torch.ones(self.num_envs, device=self.device) # Neutral reward
            
        ref_indices = self.motion_frame_idx % self.max_motion_frames
        target_velocity = self.reference_velocities[ref_indices].to(self.base_lin_vel.device)
        current_velocity = self.base_lin_vel[:, 0] # Forward velocity (x-axis)
        
        vel_error = torch.abs(current_velocity - target_velocity)
        return torch.exp(-vel_error / 0.25) # Adjusted sigma

    def _reward_leg_symmetry(self):
        if self.reference_motion is None or self.max_motion_frames == 0:
            return torch.zeros(self.num_envs, device=self.device)

        ref_indices = self.motion_frame_idx % self.max_motion_frames
        target_joint_pos = self.reference_motion[ref_indices].to(self.dof_pos.device)

        target_front_hip_diff = target_joint_pos[:, 0] - target_joint_pos[:, 3]
        current_front_hip_diff = self.dof_pos[:, 0] - self.dof_pos[:, 3]
        front_hip_error = torch.square(current_front_hip_diff - target_front_hip_diff)

        target_rear_hip_diff = target_joint_pos[:, 6] - target_joint_pos[:, 9]
        current_rear_hip_diff = self.dof_pos[:, 6] - self.dof_pos[:, 9]
        rear_hip_error = torch.square(current_rear_hip_diff - target_rear_hip_diff)
        
        symmetry_error = front_hip_error + rear_hip_error
        return torch.exp(-symmetry_error / 0.15) # Adjusted sigma

    def _reward_forward_motion(self):
        forward_vel = self.base_lin_vel[:, 0]
        vertical_vel_abs = torch.abs(self.base_lin_vel[:, 2])
        target_fwd_vel = self.command_cfg["lin_vel_x_range"][0]
        
        # Reward for achieving or maintaining forward velocity
        # Positive reward scaled by how close to target, penalty for being slow/backwards
        fwd_progress_reward = torch.where(forward_vel > 0.05, 
                                        1.0 - torch.clamp(torch.abs(forward_vel - target_fwd_vel) / target_fwd_vel, 0, 1),
                                        -1.0 + forward_vel * 10) # Penalize no/reverse motion

        # Strong penalty for vertical velocity (bouncing)
        # Sigma for vertical velocity: 0.05 m/s is already some bounce
        bounce_penalty_val = vertical_vel_abs
        bounce_penalty_reward = torch.exp(-bounce_penalty_val / 0.03) # Sharper penalty for bounce
        
        # Combine: prioritize forward progress, strongly penalize bouncing
        return fwd_progress_reward * 0.6 + (bounce_penalty_reward -1.0) * 0.4 # Ensure bounce penalty is negative

    def _reward_chassis_height(self):
        base_height = self.base_pos[:, 2]
        target_height = self.reward_cfg["base_height_target"] 
        height_error = torch.abs(base_height - target_height)
        
        # Allow small deviations, penalize larger ones. Sigma of 0.03 means +-3cm is tolerable.
        # Penalize more for being too low than too high if not perfectly at target.
        height_reward = torch.exp(-height_error / 0.03)
        # Add a stronger penalty if significantly below target (e.g. > 5cm below)
        too_low_penalty = torch.where(base_height < (target_height - 0.05), -1.0, 0.0)
        return height_reward + too_low_penalty

    def _reward_ground_contact(self):
        """Reward for feet being close to the ground and stable contact."""
        foot_positions = self._get_foot_positions() # Shape: (num_envs, 4, 3)
        foot_heights_z = foot_positions[:, :, 2]
        
        # Target foot height is z=0 (ground). Deviations are penalized.
        # Average foot height error from ground
        avg_foot_height_error = torch.mean(torch.abs(foot_heights_z), dim=1)
        # Reward for feet being close to ground (sigma 0.02 means 2cm error is significant)
        foot_on_ground_reward = torch.exp(-avg_foot_height_error / 0.02)

        # Penalize if any foot is too high (e.g., > 5cm, indicating floating or undesirable leg lift)
        max_foot_height = torch.max(foot_heights_z, dim=1)[0]
        floating_penalty = torch.where(max_foot_height > 0.05, -1.0 * (max_foot_height - 0.05) / 0.1, 0.0)
        
        # Also penalize vertical base velocity (anti-bouncing, complements forward_motion reward)
        base_vertical_vel_abs = torch.abs(self.base_lin_vel[:, 2])
        stability_penalty_val = base_vertical_vel_abs
        # stability_reward = torch.exp(-stability_penalty_val / 0.03) # a value close to 1 if stable
        # Let's make it a direct penalty that scales with vertical velocity
        stability_penalty = -2.0 * stability_penalty_val # Higher multiplier = stronger penalty

        # Combine: strong incentive for feet on ground, penalty for floating, penalty for base bouncing
        return foot_on_ground_reward + floating_penalty + stability_penalty

    def _get_foot_positions(self):
        """
        Estimate the world positions of the four feet using base position and joint angles.
        Uses a simplified kinematic model based on approximate leg segment lengths.
        Assumes a specific joint order: FR, FL, RR, RL (hip, thigh, calf for each).
        Output shape: (num_envs, 4, 3) for FR, FL, RR, RL feet.
        """
        base_pos = self.base_pos # Shape: (num_envs, 3)
        base_quat = self.base_quat # Shape: (num_envs, 4)
        dof_pos = self.dof_pos # Shape: (num_envs, 12)

        # Approximate leg segment lengths (adjust if URDF/robot specs differ)
        # These are from hip origin to thigh joint, thigh joint to calf joint, calf joint to foot/end-effector.
        # These are illustrative; actual values would come from robot model.
        l_hip_offset = 0.08  # Approximate lateral offset of hip joint from centerline (Y-axis)
        l_thigh = 0.21      # Length of the thigh segment
        l_calf = 0.21       # Length of the calf segment

        foot_positions_local = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # Joint indices for each leg (hip_y, thigh, calf)
        # Assuming joint order: FR_hip_y, FR_thigh, FR_calf, FL_hip_y, FL_thigh, FL_calf, ...
        # Hip Y is often the abduction/adduction joint.
        # Note: The reference `train.py` uses hip_joint_indices = [0,3,6,9] for abduction/adduction.
        # Let's assume the first joint of each leg group is the hip abduction/adduction.
        # And the second is thigh pitch, third is calf pitch.

        leg_joint_indices = [
            [0, 1, 2],  # FR leg: FR_hip_y, FR_thigh, FR_calf
            [3, 4, 5],  # FL leg: FL_hip_y, FL_thigh, FL_calf
            [6, 7, 8],  # RR leg: RR_hip_y, RR_thigh, RR_calf
            [9, 10, 11] # RL leg: RL_hip_y, RL_thigh, RL_calf
        ]

        # Side sign for Y-axis offsets (FR, RR are positive Y in local frame, FL, RL are negative Y)
        # This depends on your URDF definition for local leg frames relative to base_link
        # Assuming a standard quadruped setup: R legs +y, L legs -y in local frame if hip is at (0,0,0) of leg chain
        # Let's assume hip joints are already offset laterally in the URDF from base_link origin.
        # The dof_pos[leg_joint_indices[leg_idx][0]] is the hip abduction/adduction angle.

        for leg_idx in range(4):
            hip_abduction_angle = dof_pos[:, leg_joint_indices[leg_idx][0]]
            thigh_pitch_angle = dof_pos[:, leg_joint_indices[leg_idx][1]]
            calf_pitch_angle = dof_pos[:, leg_joint_indices[leg_idx][2]]

            # Position of foot in local leg frame (origin at hip joint)
            # X: forward, Y: left, Z: up (standard robot frame conventions)
            # This simplified FK assumes hip joint is the origin of the leg chain.
            
            # Lateral position (Y) due to hip abduction/adduction
            # Assuming thigh and calf are in the leg's sagittal plane after abduction.
            # Local Y for the foot relative to the hip joint attachment point on the body.
            # If l_hip_offset is distance from centerline to hip joint along Y:
            # And hip_abduction_angle is rotation around X-axis of hip:
            # This part is tricky without knowing the exact URDF link structure.
            # For now, let's simplify: the hip abduction angle primarily affects the Y and Z of the thigh/calf in the hip frame.
            # Let's assume the `l_hip_offset` is handled by the URDF structure itself, and the angles start from there.

            # Calculate foot position in a coordinate system attached to the hip, oriented with the base
            # Local X (forward/backward in leg plane)
            # Thigh pitch positive = forward, Calf pitch positive = extends leg
            local_x = l_thigh * torch.sin(thigh_pitch_angle) + l_calf * torch.sin(thigh_pitch_angle + calf_pitch_angle)
            
            # Local Z (downward/upward in leg plane)
            # Thigh pitch positive = forward (so Z goes down), Calf pitch positive = extends (Z goes further down)
            local_z = - (l_thigh * torch.cos(thigh_pitch_angle) + l_calf * torch.cos(thigh_pitch_angle + calf_pitch_angle))

            # Local Y (sideways due to abduction/adduction)
            # This is the most complex without full FK. A simple approximation:
            # Assume the leg extends mainly in XZ plane of a frame rotated by hip_abduction_angle around body's X-axis.
            # This is not perfect. A true FK is needed for accuracy.
            # For a simple model, let's assume hip abduction moves the foot purely laterally *after* the XZ extension.
            # Or, that the l_hip_offset is a fixed body point, and abduction rotates the leg plane.

            # Let's define a base offset for each hip joint from the body center
            # These are rough estimates for a typical quadruped layout.
            # FR: (+x_hip_offset, +y_hip_offset, 0)
            # FL: (+x_hip_offset, -y_hip_offset, 0)
            # RR: (-x_hip_offset, +y_hip_offset, 0)
            # RL: (-x_hip_offset, -y_hip_offset, 0)
            x_hip_body_offset = 0.19 # Example: distance from base center to hip cluster (front/rear)
            y_hip_body_offset = 0.08 # Example: half distance between left/right hips

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

            # Foot position in a frame aligned with the hip attachment point, before body rotation
            # X (forward), Y (left), Z (up)
            # Assume hip abduction (around X axis of hip frame) rotates the leg plane
            # For now, a simplified 2D FK in the leg's sagittal plane, then place it laterally
            foot_pos_sagittal_plane = torch.stack([
                local_x,
                torch.zeros_like(local_x), # Y is zero in this sagittal plane
                local_z
            ], dim=-1)

            # Apply hip abduction rotation (around X-axis of hip frame)
            # This would rotate the Y and Z components of foot_pos_sagittal_plane
            # foot_y_after_abduction = foot_pos_sagittal_plane[:,1] * torch.cos(hip_abduction_angle) - foot_pos_sagittal_plane[:,2] * torch.sin(hip_abduction_angle)
            # foot_z_after_abduction = foot_pos_sagittal_plane[:,1] * torch.sin(hip_abduction_angle) + foot_pos_sagittal_plane[:,2] * torch.cos(hip_abduction_angle)
            # This simple 2D FK doesn't really work well with 3D abduction directly.
            # A common simplification: hip abduction mainly causes a lateral displacement of the foot *in the world XY plane* relative to hip.
            # For now, we'll stick to the 2D leg plane (local_x, local_z) and add the hip offset.
            # The `local_y` from abduction is harder to model simply.

            # Position of the foot relative to the hip joint in base frame coordinates
            # This is still a simplification. True FK is more involved.
            foot_positions_local[:, leg_idx, 0] = hip_origin_in_base_frame[:, 0] + local_x
            foot_positions_local[:, leg_idx, 1] = hip_origin_in_base_frame[:, 1] # Add abduction effect here if modeled
            foot_positions_local[:, leg_idx, 2] = hip_origin_in_base_frame[:, 2] + local_z
        
        # Rotate foot_positions_local by base_quat and add base_pos
        # Import from genesis if not already available (e.g. quat_apply)
        # For simplicity, if gs.math.quat_apply is not directly usable here, 
        # this part needs to be implemented carefully.
        # Placeholder for rotation:
        # foot_positions_world = gs.math.quat_apply_yaw(base_quat, foot_positions_local) + base_pos.unsqueeze(1)
        # Since we don't have gs.math.quat_apply easily, let's do a simplified yaw rotation for now
        # this is a MAJOR simplification and will only be correct if the robot has no roll/pitch.
        
        # Using a more complete quaternion rotation (if torch_utils or similar is available)
        # from isaacgym.torch_utils import quat_apply (this cannot be used here)
        # Manual quaternion rotation:
        foot_positions_world = torch.zeros_like(foot_positions_local)
        for i in range(self.num_envs):
            q = base_quat[i]
            # Normalize quaternion
            q = q / torch.norm(q)
            
            # Rodrigues' rotation formula components
            # vector part of quaternion
            u = q[0:3] # x, y, z components of quaternion assuming (x,y,z,w) or get from (w,x,y,z)
            s = q[3]   # scalar part (w)
            
            for j in range(4): # For each foot
                v = foot_positions_local[i, j, :]
                # v_rotated = v + 2 * torch.cross(u, torch.cross(u, v) + s * v)
                # More standard formula for qvq*: v' = v + 2s(u x v) + 2(u x (u x v))
                # Simplified: using matrix form of quaternion rotation for each vector v
                # R = [[1-2(qy^2+qz^2), 2(qxqy-qzw), 2(qxqz+qyw)],
                #      [2(qxqy+qzw), 1-2(qx^2+qz^2), 2(qyqz-qxw)],
                #      [2(qxqz-qyw), 2(qyqz+qxw), 1-2(qx^2+qy^2)]]
                # where q = (w,x,y,z) -> here (s, u[0], u[1], u[2])
                
                # Using simplified X-Y-Z euler from quat for now due to complexity of full quat_apply
                # This is NOT a full FK solution. For ground contact, Z is most critical.
                # We are primarily interested in the Z component for ground contact.
                # The local_z is Z relative to hip in leg plane. We need Z in world frame.
                # A very basic approximation: world_z_foot = base_z + local_z_rotated_by_base_pitch_roll
                # For now, a simple Z offset from base, assuming small pitch/roll:
                foot_positions_world[i,j,2] = base_pos[i,2] + foot_positions_local[i,j,2]
                # Copy X and Y as a rough estimate (will be inaccurate without full rotation)
                foot_positions_world[i,j,0] = base_pos[i,0] + foot_positions_local[i,j,0]
                foot_positions_world[i,j,1] = base_pos[i,1] + foot_positions_local[i,j,1]

        return foot_positions_world

    def reset(self):
        """Reset the environment and motion frames with proper ground contact."""
        # Store the original DOF state before resetting
        original_dof_pos = None
        if hasattr(self, 'dof_pos') and self.dof_pos is not None:
            original_dof_pos = self.dof_pos.clone()
        
        # Indices of environments to reset for this specific call within MotionImitationEnv logic (if needed)
        # This renames the variable to avoid conflict with the Go2Env.reset_idx *method*.
        current_call_reset_indices = torch.arange(self.num_envs, device=self.device)
        # If current_call_reset_indices was intended to be used to determine which envs Go2Env.reset_idx
        # should operate on, that logic would need to be passed differently, as Go2Env.reset_idx
        # internally calculates its own envs_idx based on self.reset_buf.
        # For now, this tensor is created but not directly used to call Go2Env.reset_idx, resolving the conflict.

        # Call parent reset method with careful initialization
        obs = super().reset() # This will call Go2Env.reset(), which uses its own reset_idx method.
        
        # Reset motion frame indices
        self.motion_frame_idx.zero_()
        
        # Apply reference motion's first frame for joint positions
        if self.reference_motion is not None and self.max_motion_frames > 0:
            first_frame = self.reference_motion[0].clone()
            if hasattr(self, 'dof_pos') and self.dof_pos is not None:
                self.dof_pos[:] = first_frame.unsqueeze(0).repeat(self.num_envs, 1)
                if hasattr(self, 'dof_targets'): self.dof_targets[:] = self.dof_pos[:]
                if hasattr(self, 'default_dof_targets'): self.default_dof_targets[:] = self.dof_pos[:]
                if hasattr(self, 'dof_vel'): self.dof_vel[:] = torch.zeros_like(self.dof_vel)
        
        if hasattr(self, 'root_states'):
            self.root_states[:, 2] = self.env_cfg["base_init_pos"][2]
            self.root_states[:, 7:13] = 0.0
        
        if hasattr(self, 'base_pos') and self.base_pos is not None:
            self.position_history = self.base_pos.clone()
        
        # Physics stabilization (remains the same)
        if hasattr(self, 'scene') and hasattr(self.scene, 'physics'):
            try:
                if hasattr(self.scene.physics, 'solver_iterations'): self.scene.physics.solver_iterations = 8
                if hasattr(self.scene.physics, 'enable_stabilization'): self.scene.physics.enable_stabilization = True
            except Exception as e:
                print(f"Warning: Could not set advanced physics parameters: {e}")
        
        if hasattr(self, 'scene'):
            try:
                for _ in range(10):
                    if hasattr(self.scene, 'step_no_render'): self.scene.step_no_render()
                    elif hasattr(self.scene, 'simulate'): self.scene.simulate(0.01)
            except Exception as e:
                print(f"Warning: Could not run stabilization steps: {e}")
        
        if hasattr(self, '_compute_observations'):
            try:
                self._compute_observations()
            except: 
                pass # Skip if method is not compatible
        
        return obs
    
    def step(self, actions):
        # Store previous base position for forward motion reward
        if hasattr(self, 'base_pos') and self.base_pos is not None: # Check if base_pos exists
            self.position_history = self.base_pos.clone()
        else: # Fallback if base_pos isn't available directly, try root_states
            if hasattr(self, 'root_states') and self.root_states is not None:
                 self.position_history = self.root_states[:, 0:3].clone()

        obs, rew_buf, reset_buf, extras = super().step(actions)
        
        # Enforce fixed base height if enabled in env_cfg (intended for specific debug scenarios)
        # if self.env_cfg.get("use_fixed_base_constraint", False):
        #     self._enforce_fixed_base_height() # This was from a previous fixed-base experiment
        
        self.motion_frame_idx += 1
        
        if torch.any(reset_buf):
            reset_idx = reset_buf.nonzero(as_tuple=False).flatten()
            self.motion_frame_idx[reset_idx] = 0
            if hasattr(self, 'position_history') and self.base_pos is not None: 
                self.position_history[reset_idx] = self.base_pos[reset_idx].clone()
            elif hasattr(self, 'root_states') and self.root_states is not None: 
                self.position_history[reset_idx] = self.root_states[reset_idx, 0:3].clone()

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
            # Alternatively, the scene might be closed by its __del__ or a similar mechanism in Genesis
            # For now, attempting to close viewer is a good first step.

    def _setup_physics_parameters(self):
        """Configure physics parameters to prevent bouncing and ensure stability."""
        # This method is currently minimal as physics are set in get_cfgs and passed via env_cfg
        # It can be expanded if specific runtime physics adjustments are needed beyond cfgs.
        if hasattr(self, 'scene') and self.scene is not None and hasattr(self.scene, 'physics'):
            # Example: self.scene.physics.gravity = (0, 0, self.env_cfg.get("gravity", -9.81))
            pass # Physics parameters are largely driven by env_cfg from get_cfgs
        return

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
        "base_height_target": 0.28, # Target height for walking
        "reward_scales": {
            "tracking_lin_vel": 0.4, 
            "tracking_ang_vel": 0.2, 
            "lin_vel_z": -2.0, # Strong penalty for vertical base velocity (anti-bounce)
            "action_rate": -0.015, 
            "similar_to_default": -0.03, 
            # Custom rewards are primary and will be scaled in main
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
    parser = argparse.ArgumentParser(description="Train Go2 robot to imitate retargeted motion files")
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
    parser.add_argument("--wandb-project", type=str, default="go2-motion-imitation",
                        help="W&B project name (default: go2-motion-imitation)")
    args = parser.parse_args()

    gs.init(logging_level="warning")
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    motion_basename = os.path.basename(args.file).split('.')[0]
    exp_name = f"go2-imitate-{motion_basename}"
    log_dir_basename = f"go2-imitate-{motion_basename}"
    log_dir = f"logs/{log_dir_basename}"
    if os.path.exists(log_dir) and not os.listdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    reference_motion = load_reference_motion(args.file)
    if reference_motion is None:
        print(f"Error: Could not load motion file {args.file}. Aborting training.")
        return
    
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(exp_name, args.iters)

    # --- Store motion_filename in env_cfg before saving cfgs.pkl ---
    env_cfg["motion_filename"] = args.file

    # --- Key Reward Scales for Grounded Walking --- 
    reward_cfg["reward_scales"]["joint_pose_matching"] = 3.0
    reward_cfg["reward_scales"]["ground_contact"] = 3.5
    reward_cfg["reward_scales"]["forward_motion"] = 2.0
    reward_cfg["reward_scales"]["chassis_height"] = 1.0
    reward_cfg["reward_scales"]["velocity_profile"] = 1.0
    reward_cfg["reward_scales"]["leg_symmetry"] = 0.75

    # Remove default base_height if custom chassis_height is used
    if "base_height" in reward_cfg["reward_scales"]:
         del reward_cfg["reward_scales"]["base_height"]
    
    print("Final Reward Scales:", reward_cfg["reward_scales"])

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    print(f"\nCreating {args.envs} environments...")
    env = MotionImitationEnv(
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
        "env_cfg": env_cfg, # Log env_cfg
        "policy": train_cfg["policy"],
        "algorithm": train_cfg["algorithm"],
    }
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    wandb_callback = WandbCallback(
        runner=runner,
        project_name=args.wandb_project,
        experiment_name=exp_name,
        config=config
    ) if use_wandb else None
    
    print(f"\n{'='*50}")
    print(f"Starting training for {args.iters} iterations")
    print(f"Motion file: {args.file}")
    print(f"Log directory: {log_dir}")
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