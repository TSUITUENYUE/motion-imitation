#!/usr/bin/env python3
"""
Script for retargeting a running motion to Go2 robot.
Simplified version focused only on the run motion.
"""
import os
import argparse
import torch
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz
from go2_env import Go2Env
from dog_walk_parser import DogWalkParser

# Define only the run motion parameters
RUN_MOTION = ["run", "data/dog_run00_joint_pos.txt", 430, 459]

class RunMotionRetargeter:
    """Class for retargeting running motion to Go2 robot."""
    
    def __init__(self, device=None):
        """Initialize the retargeter."""
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.joint_positions = None
        self.joint_rotations = None
        self.joint_velocities = None
        self.joint_names = None
        self.num_frames = 0
        
        # Mapping between source joint space and target joint space
        self.joint_mapping = self._create_joint_mapping()
        self.joint_idx_mapping = None
        self.joint_offsets = {}
        
        # Parameters specifically tuned for run motion
        self.motion_params = {
            "thigh_bias": 1.4,  # Higher value for more dynamic run
            "calf_bias": -1.6,  # More bent for running posture
            "rear_thigh_bias": 1.6,  # More lift for rear legs during running
            "thigh_amplitude": 2.0,  # Increased leg swing for running
            "calf_amplitude": 1.8,  # More pronounced calf movement for running
            "hip_amplitude": 0.7,  # Slightly higher hip movement for running
            "stride_peak_threshold": 0.65,  # Threshold for extra lift during stride
            "extra_lift_thigh_front": 2.8,  # Increased lift during stride peaks (front)
            "extra_lift_calf_front": -2.9,  # Increased bend during stride peaks (front)
            "extra_lift_thigh_rear": 3.0,  # Increased lift during stride peaks (rear)
            "extra_lift_calf_rear": -3.0,  # Increased bend during stride peaks (rear)
        }
        
        self.env = None
        self.motion_file_path = None
        
        # Environment configurations for Go2 environment - tuned for running
        self.env_cfg = {
            "num_actions": 12,
            "default_joint_angles": {
                # Hip joints - set for running stance
                "FL_hip_joint": 0.0,
                "FR_hip_joint": 0.0,
                "RL_hip_joint": 0.0,
                "RR_hip_joint": 0.0,
                # Thigh joints - more crouched for running
                "FL_thigh_joint": 0.7,
                "FR_thigh_joint": 0.7,
                "RL_thigh_joint": 0.9,
                "RR_thigh_joint": 0.9,
                # Calf joints - more bent for explosive running motion
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
    
    def _create_joint_mapping(self):
        """Create joint mapping from dog to Go2 joints."""
        mapping = {
            "FR_hip": "FR_hip_joint",
            "FR_knee": "FR_thigh_joint",
            "FR_ankle": "FR_calf_joint",
            "FL_hip": "FL_hip_joint",
            "FL_knee": "FL_thigh_joint",
            "FL_ankle": "FL_calf_joint",
            "BR_hip": "RR_hip_joint",
            "BR_knee": "RR_thigh_joint",
            "BR_ankle": "RR_calf_joint",
            "BL_hip": "RL_hip_joint",
            "BL_knee": "RL_thigh_joint",
            "BL_ankle": "RL_calf_joint",
        }
        return mapping
    
    def load_motion_data(self, motion_file=None, frame_start=None, frame_end=None):
        """Load and parse dog motion data with specific frame range."""
        # Use default run motion if none specified
        if motion_file is None:
            motion_name, motion_file, frame_start, frame_end = RUN_MOTION
            
        self.motion_file_path = motion_file
        parser = DogWalkParser(device=self.device)
        parser.parse_file(motion_file)
        data = parser.get_data()
        
        # Extract frame range if specified
        if frame_start is not None and frame_end is not None:
            # Ensure frame indices are within bounds
            frame_start = max(0, min(frame_start, data['joint_positions'].shape[0] - 1))
            frame_end = max(frame_start + 1, min(frame_end, data['joint_positions'].shape[0]))
            
            # Extract the specified frame range
            self.joint_positions = data['joint_positions'][frame_start:frame_end].clone()
            self.joint_rotations = data['joint_rotations'][frame_start:frame_end].clone()
            self.joint_velocities = data['joint_velocities'][frame_start:frame_end].clone()
            
            print(f"Extracted frames {frame_start} to {frame_end} from {motion_file}")
        else:
            # Use all frames
            self.joint_positions = data['joint_positions'].clone()
            self.joint_rotations = data['joint_rotations'].clone()
            self.joint_velocities = data['joint_velocities'].clone()
        
        self.num_frames = self.joint_positions.shape[0]
        self.joint_names = parser.get_joint_names()
        print(f"Loaded {self.num_frames} frames of dog motion")
        
        # Create mapping from dog joint names to Go2 joint indices
        self.joint_idx_mapping = {}
        for dog_idx, dog_name in enumerate(self.joint_names):
            if dog_name in self.joint_mapping:
                go2_joint_name = self.joint_mapping[dog_name]
                go2_idx = self.env_cfg["joint_names"].index(go2_joint_name)
                self.joint_idx_mapping[dog_name] = go2_idx
        
        # Set joint offsets to default values
        self.joint_offsets = {name: self.env_cfg["default_joint_angles"][name] 
                             for name in self.env_cfg["joint_names"]}
        
        return self
    
    def retarget_frame(self, frame_idx, custom_params=None):
        """Retarget a single frame of dog motion to Go2."""
        if self.joint_positions is None or self.joint_rotations is None:
            raise ValueError("No motion data loaded")
            
        if self.joint_idx_mapping is None:
            raise ValueError("Joint mapping not created")
        
        # Use custom parameters if provided, otherwise use defaults
        params = custom_params if custom_params is not None else self.motion_params
            
        # Initialize joint angles with default offsets
        go2_joint_angles = torch.zeros(12, device=self.device)
        for joint_name, offset in self.joint_offsets.items():
            go2_idx = self.env_cfg["joint_names"].index(joint_name)
            go2_joint_angles[go2_idx] = offset
        
        # Define joint indices for the Go2 robot
        fr_hip_idx = self.env_cfg["joint_names"].index("FR_hip_joint")
        fr_thigh_idx = self.env_cfg["joint_names"].index("FR_thigh_joint")
        fr_calf_idx = self.env_cfg["joint_names"].index("FR_calf_joint")
        
        fl_hip_idx = self.env_cfg["joint_names"].index("FL_hip_joint")
        fl_thigh_idx = self.env_cfg["joint_names"].index("FL_thigh_joint")
        fl_calf_idx = self.env_cfg["joint_names"].index("FL_calf_joint")
        
        rr_hip_idx = self.env_cfg["joint_names"].index("RR_hip_joint")
        rr_thigh_idx = self.env_cfg["joint_names"].index("RR_thigh_joint")
        rr_calf_idx = self.env_cfg["joint_names"].index("RR_calf_joint")
        
        rl_hip_idx = self.env_cfg["joint_names"].index("RL_hip_joint")
        rl_thigh_idx = self.env_cfg["joint_names"].index("RL_thigh_joint")
        rl_calf_idx = self.env_cfg["joint_names"].index("RL_calf_joint")
        
        # Print debug only for first frame
        print_debug = (frame_idx == 0)
        
        # Apply motion synthesis for running - faster phase for running
        leg_phase = frame_idx / self.num_frames * 2 * 3.14159 * 1.5  # Faster phase for running
        
        forward_offset = torch.zeros(3, device=self.device)
        
        # Create phase relationship for running (trot pattern)
        # In a trot, diagonal legs move together: FR+RL and FL+RR
        fr_phase = leg_phase
        fl_phase = leg_phase + 3.14159  # 180 degrees out of phase (trot)
        rr_phase = leg_phase + 3.14159  # 180 degrees out of phase with FL (trot)
        rl_phase = leg_phase            # In phase with FR (trot)
        
        # Use parameters for motion generation
        thigh_bias = params["thigh_bias"]
        calf_bias = params["calf_bias"]
        rear_thigh_bias = params["rear_thigh_bias"]
        thigh_amplitude = params["thigh_amplitude"]
        calf_amplitude = params["calf_amplitude"]
        hip_amplitude = params["hip_amplitude"]
        
        # Generate running motion with faster, more pronounced leg movement
        
        # Front right leg
        go2_joint_angles[fr_thigh_idx] = thigh_bias + thigh_amplitude * torch.sin(torch.tensor(fr_phase, device=self.device))
        go2_joint_angles[fr_calf_idx] = calf_bias - calf_amplitude * torch.sin(torch.tensor(fr_phase, device=self.device))
        
        # Front left leg
        go2_joint_angles[fl_thigh_idx] = thigh_bias + thigh_amplitude * torch.sin(torch.tensor(fl_phase, device=self.device))
        go2_joint_angles[fl_calf_idx] = calf_bias - calf_amplitude * torch.sin(torch.tensor(fl_phase, device=self.device))
        
        # Rear right leg
        go2_joint_angles[rr_thigh_idx] = rear_thigh_bias + thigh_amplitude * torch.sin(torch.tensor(rr_phase, device=self.device))
        go2_joint_angles[rr_calf_idx] = calf_bias - calf_amplitude * torch.sin(torch.tensor(rr_phase, device=self.device))
        
        # Rear left leg
        go2_joint_angles[rl_thigh_idx] = rear_thigh_bias + thigh_amplitude * torch.sin(torch.tensor(rl_phase, device=self.device)) 
        go2_joint_angles[rl_calf_idx] = calf_bias - calf_amplitude * torch.sin(torch.tensor(rl_phase, device=self.device))
        
        # Add hip rotation for more natural running motion (less lateral movement than walking)
        go2_joint_angles[fr_hip_idx] = hip_amplitude * 0.8 * torch.sin(torch.tensor(fr_phase - 0.2, device=self.device))
        go2_joint_angles[fl_hip_idx] = hip_amplitude * 0.8 * torch.sin(torch.tensor(fl_phase - 0.2, device=self.device))
        go2_joint_angles[rr_hip_idx] = hip_amplitude * 0.8 * torch.sin(torch.tensor(rr_phase - 0.2, device=self.device))
        go2_joint_angles[rl_hip_idx] = hip_amplitude * 0.8 * torch.sin(torch.tensor(rl_phase - 0.2, device=self.device))
        
        # Override the thigh values during stride peaks to get more pronounced knee action for running
        stride_peak_threshold = params["stride_peak_threshold"]
        
        # Get extra lift parameters - higher values for running
        extra_lift_thigh_front = params["extra_lift_thigh_front"]
        extra_lift_calf_front = params["extra_lift_calf_front"]
        extra_lift_thigh_rear = params["extra_lift_thigh_rear"]
        extra_lift_calf_rear = params["extra_lift_calf_rear"]
        
        # Front right extra lift during swing phase
        if torch.sin(torch.tensor(fr_phase, device=self.device)) > stride_peak_threshold:
            go2_joint_angles[fr_thigh_idx] = extra_lift_thigh_front
            go2_joint_angles[fr_calf_idx] = extra_lift_calf_front
            
        # Front left extra lift during swing phase
        if torch.sin(torch.tensor(fl_phase, device=self.device)) > stride_peak_threshold:
            go2_joint_angles[fl_thigh_idx] = extra_lift_thigh_front
            go2_joint_angles[fl_calf_idx] = extra_lift_calf_front
            
        # Rear right extra lift during swing phase
        if torch.sin(torch.tensor(rr_phase, device=self.device)) > stride_peak_threshold:
            go2_joint_angles[rr_thigh_idx] = extra_lift_thigh_rear
            go2_joint_angles[rr_calf_idx] = extra_lift_calf_rear
            
        # Rear left extra lift during swing phase
        if torch.sin(torch.tensor(rl_phase, device=self.device)) > stride_peak_threshold:
            go2_joint_angles[rl_thigh_idx] = extra_lift_thigh_rear
            go2_joint_angles[rl_calf_idx] = extra_lift_calf_rear
            
        # Apply limits to constrain the motion within reasonable bounds
        
        # Limit hip joints
        for idx in [fr_hip_idx, fl_hip_idx, rr_hip_idx, rl_hip_idx]:
            go2_joint_angles[idx] = torch.clamp(go2_joint_angles[idx], min=-0.7, max=0.7)
        
        # Limit thigh joints (different for front and rear)
        go2_joint_angles[fr_thigh_idx] = torch.clamp(go2_joint_angles[fr_thigh_idx], min=0.1, max=2.8)
        go2_joint_angles[fl_thigh_idx] = torch.clamp(go2_joint_angles[fl_thigh_idx], min=0.1, max=2.8)
        go2_joint_angles[rr_thigh_idx] = torch.clamp(go2_joint_angles[rr_thigh_idx], min=0.2, max=2.8)
        go2_joint_angles[rl_thigh_idx] = torch.clamp(go2_joint_angles[rl_thigh_idx], min=0.2, max=2.8)
        
        # Limit calf joints
        go2_joint_angles[fr_calf_idx] = torch.clamp(go2_joint_angles[fr_calf_idx], min=-3.0, max=-0.2)
        go2_joint_angles[fl_calf_idx] = torch.clamp(go2_joint_angles[fl_calf_idx], min=-3.0, max=-0.2)
        go2_joint_angles[rr_calf_idx] = torch.clamp(go2_joint_angles[rr_calf_idx], min=-3.0, max=-0.2)
        go2_joint_angles[rl_calf_idx] = torch.clamp(go2_joint_angles[rl_calf_idx], min=-3.0, max=-0.2)
        
        # Print debug info for first frame
        if print_debug:
            print(f"GO2 joint angles for running motion:")
            print(f"  FR_hip: {go2_joint_angles[fr_hip_idx]:.4f}")
            print(f"  FR_thigh: {go2_joint_angles[fr_thigh_idx]:.4f}")
            print(f"  FR_calf: {go2_joint_angles[fr_calf_idx]:.4f}")
            
        return go2_joint_angles, forward_offset
    
    def retarget_sequence(self, custom_params=None):
        """Retarget the entire motion sequence."""
        if self.num_frames == 0:
            raise ValueError("No motion data loaded")
        
        # Initialize tensor for all retargeted frames
        go2_joint_angles = torch.zeros((self.num_frames, 12), device=self.device)
        forward_offsets = torch.zeros((self.num_frames, 3), device=self.device)
        
        # Retarget each frame
        print(f"Retargeting {self.num_frames} frames of running motion...")
        for i in range(self.num_frames):
            # Process each frame separately
            joint_angles, forward_offset = self.retarget_frame(i, custom_params)
            go2_joint_angles[i] = joint_angles
            forward_offsets[i] = forward_offset
            
            # Progress indicator
            if i % 10 == 0 or i == self.num_frames - 1:
                print(f"  Processed {i+1}/{self.num_frames} frames")
        
        # Save retargeted motion to file
        output_data = {
            'joint_angles': go2_joint_angles,
            'forward_offsets': forward_offsets
        }
        
        os.makedirs("output", exist_ok=True)
        output_file = "output/run_retargeted.npy"
        torch.save(output_data, output_file)
        print(f"Saved retargeted running motion to {output_file}")
        
        return go2_joint_angles, forward_offsets
    
    def save_retargeted_motion(self, joint_angles, output_file):
        """Save retargeted motion to file."""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Save as torch file
        output_data = {
            'joint_angles': joint_angles
        }
        torch.save(output_data, output_file)
        print(f"Saved retargeted motion to {output_file}")
        return output_file


def main():
    """Main function to retarget dog motion to Go2 robot."""
    parser = argparse.ArgumentParser(description="Running motion retargeting for Go2 robot")
    parser.add_argument("--file", type=str, default=None, help="Custom motion file path")
    parser.add_argument("--start", type=int, help="Start frame for custom file")
    parser.add_argument("--end", type=int, help="End frame for custom file")
    parser.add_argument("--output", type=str, default="output/run_retargeted.npy", help="Output file")
    
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init()
    
    # Create retargeter
    retargeter = RunMotionRetargeter()
    
    # Load motion data (use default if not specified)
    if args.file:
        retargeter.load_motion_data(args.file, args.start, args.end)
    else:
        # Use default run motion
        motion_name, file_path, frame_start, frame_end = RUN_MOTION
        print(f"Using default running motion from {file_path}")
        retargeter.load_motion_data(file_path, frame_start, frame_end)
    
    # Retarget motion
    joint_angles, forward_offsets = retargeter.retarget_sequence()
    
    # Save to specified output file
    output_data = {
        'joint_angles': joint_angles,
        'forward_offsets': forward_offsets
    }
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save(output_data, args.output)
    print(f"Saved retargeted motion to {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
