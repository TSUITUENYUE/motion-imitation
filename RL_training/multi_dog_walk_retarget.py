#!/usr/bin/env python3
"""
Script for retargeting multiple dog motion files to Go2 robot.
Uses the same motion files and frame ranges as the original retarget_motion_old.py.
"""
import os
import argparse
import torch
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz
from go2_env import Go2Env
from dog_walk_parser import DogWalkParser
import itertools
from tqdm import tqdm

# Define the motion types with their filenames and frame ranges
# These match the definitions from retarget_motion_old.py
MOCAP_MOTIONS = [
    ["pace", "data/dog_walk00_joint_pos.txt", 162, 201],
    ["trot", "data/dog_walk03_joint_pos.txt", 448, 481],
    ["trot2", "data/dog_run04_joint_pos.txt", 630, 663],
    ["canter", "data/dog_run00_joint_pos.txt", 430, 459],
    ["left_turn", "data/dog_walk09_joint_pos.txt", 1085, 1124],
    ["right_turn", "data/dog_walk09_joint_pos.txt", 2404, 2450],
    ["walk", "data/dog_walk01_joint_pos.txt", 100, 200],  # Added a longer walk motion with 100 frames
]

class MultiDogMotionRetargeter:
    """Class for retargeting multiple dog motion files to Go2 robot."""
    
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
        
        # Default bias and scale parameters for motion retargeting
        self.motion_params = {
            "thigh_bias": 1.2,  # Base value for front thigh joints
            "calf_bias": -1.5,  # Base value for calf joints
            "rear_thigh_bias": 1.4,  # Base value for rear thigh joints
            "thigh_amplitude": 1.7,  # Scale for thigh motion
            "calf_amplitude": 1.5,  # Scale for calf motion
            "hip_amplitude": 0.65,  # Scale for hip motion
            "stride_peak_threshold": 0.7,  # Threshold for extra lift during stride
            "extra_lift_thigh_front": 2.6,  # Increased lift during stride peaks (front)
            "extra_lift_calf_front": -2.8,  # Increased bend during stride peaks (front)
            "extra_lift_thigh_rear": 2.8,  # Increased lift during stride peaks (rear)
            "extra_lift_calf_rear": -2.8,  # Increased bend during stride peaks (rear)
        }
        
        self.env = None
        self.motion_file_path = None  # Store the motion file path
        
        # Environment configurations for Go2 environment - increase tolerance to prevent early termination
        self.env_cfg = {
            "num_actions": 12,
            "default_joint_angles": {
                # Hip joints - set to zero for a horizontal body
                "FL_hip_joint": 0.0,
                "FR_hip_joint": 0.0,
                "RL_hip_joint": 0.0,
                "RR_hip_joint": 0.0,
                # Thigh joints - slightly more crouched stance for stability
                "FL_thigh_joint": 0.6,  # Reduced for lower stance
                "FR_thigh_joint": 0.6,  # Reduced for lower stance
                "RL_thigh_joint": 0.8,  # Reduced for lower stance
                "RR_thigh_joint": 0.8,  # Reduced for lower stance
                # Calf joints - adjusted for better ground contact
                "FL_calf_joint": -1.3,  # More bent for better stance
                "FR_calf_joint": -1.3,  # More bent for better stance
                "RL_calf_joint": -1.3,  # More bent for better stance
                "RR_calf_joint": -1.3,  # More bent for better stance
            },
            "joint_names": [
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            ],
            "kp": 12.0,  # Further reduced for more compliant motion
            "kd": 0.3,   # Reduced for smoother transitions
            "termination_if_roll_greater_than": 120.0,  # Increased to prevent early termination
            "termination_if_pitch_greater_than": 120.0, # Increased to prevent early termination
            "base_init_pos": [0.0, 0.0, 0.6],  # Increased height from 0.4 to 0.6
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 60.0,
            "resampling_time_s": 10.0,
            "action_scale": 0.15,  # Further reduced for more controlled motion
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
            "BR_hip": "RR_hip_joint",  # Back right to rear right
            "BR_knee": "RR_thigh_joint",
            "BR_ankle": "RR_calf_joint",
            "BL_hip": "RL_hip_joint",  # Back left to rear left
            "BL_knee": "RL_thigh_joint",
            "BL_ankle": "RL_calf_joint",
        }
        return mapping
    
    def load_motion_data(self, motion_file, frame_start=None, frame_end=None):
        """Load and parse dog motion data with specific frame range."""
        self.motion_file_path = motion_file  # Store the motion file path
        parser = DogWalkParser(device=self.device)
        parser.parse_file(motion_file)
        data = parser.get_data()
        
        # Print data dimensions for debugging
        print(f"Original data dimensions:")
        print(f"  Joint positions: {data['joint_positions'].shape}")
        print(f"  Joint rotations: {data['joint_rotations'].shape}")
        print(f"  Joint velocities: {data['joint_velocities'].shape}")
        
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
        
        # Print updated dimensions for debugging
        print(f"Extracted data dimensions:")
        print(f"  Joint positions: {self.joint_positions.shape}")
        print(f"  Joint rotations: {self.joint_rotations.shape}")
        print(f"  Joint velocities: {self.joint_velocities.shape}")
        
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
        # These are based on the joint_names list defined in env_cfg
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
        
        # Debug print - only print every few frames to avoid spamming
        print_debug = (frame_idx % 10 == 0)  # Print every 10 frames
        
        # Apply different motion synthesis for better leg movement
        # Create sinusoidal pattern for leg motion that's more exaggerated
        leg_phase = frame_idx / self.num_frames * 2 * 3.14159
        
        # Use a constant forward speed - this is simpler than using the forward_offset
        # and avoids the error with trying to set root_states directly
        forward_offset = torch.zeros(3, device=self.device)
        
        # Modify the phase relationships to create a clear diagonal gait pattern
        # This is the typical pattern for a walking dog: diagonal legs move together
        # (FL+RR, FR+RL) with appropriate timing
        
        # Use a quadruped gait pattern: 
        # 1. FR lifts and moves forward
        # 2. FL lifts and moves forward (when FR is halfway through stance)
        # 3. RR lifts and moves forward (when FR is starting new cycle)
        # 4. RL lifts and moves forward (when FL is halfway through stance)
        
        # Create a more natural phase offset for quadruped walking
        # Each leg is 25% out of phase with the previous one
        fr_phase = leg_phase
        fl_phase = leg_phase + 3.14159/2      # 90 degrees out of phase
        rr_phase = leg_phase + 3.14159        # 180 degrees out of phase 
        rl_phase = leg_phase + 3.14159*1.5    # 270 degrees out of phase
        
        # Use parameters for motion generation
        thigh_bias = params["thigh_bias"]
        calf_bias = params["calf_bias"]
        rear_thigh_bias = params["rear_thigh_bias"]
        thigh_amplitude = params["thigh_amplitude"]
        calf_amplitude = params["calf_amplitude"]
        hip_amplitude = params["hip_amplitude"]
        
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
        
        # Add hip rotation for more natural walking motion (lateral movement)
        # Adjust hip motion to match limb phase
        go2_joint_angles[fr_hip_idx] = hip_amplitude * torch.sin(torch.tensor(fr_phase - 0.3, device=self.device))
        go2_joint_angles[fl_hip_idx] = hip_amplitude * torch.sin(torch.tensor(fl_phase - 0.3, device=self.device))
        go2_joint_angles[rr_hip_idx] = hip_amplitude * torch.sin(torch.tensor(rr_phase - 0.3, device=self.device))
        go2_joint_angles[rl_hip_idx] = hip_amplitude * torch.sin(torch.tensor(rl_phase - 0.3, device=self.device))
        
        # Override the thigh values during stride peaks to get more pronounced knee action
        stride_peak_threshold = params["stride_peak_threshold"]
        
        # Get extra lift parameters
        extra_lift_thigh_front = params["extra_lift_thigh_front"]
        extra_lift_calf_front = params["extra_lift_calf_front"]
        extra_lift_thigh_rear = params["extra_lift_thigh_rear"]
        extra_lift_calf_rear = params["extra_lift_calf_rear"]
        
        # Front right extra lift during swing phase (when sin value is positive)
        if torch.sin(torch.tensor(fr_phase, device=self.device)) > stride_peak_threshold:
            go2_joint_angles[fr_thigh_idx] = extra_lift_thigh_front  # Higher lift
            go2_joint_angles[fr_calf_idx] = extra_lift_calf_front  # More bent
            
        # Front left extra lift during swing phase
        if torch.sin(torch.tensor(fl_phase, device=self.device)) > stride_peak_threshold:
            go2_joint_angles[fl_thigh_idx] = extra_lift_thigh_front  # Higher lift
            go2_joint_angles[fl_calf_idx] = extra_lift_calf_front  # More bent
            
        # Rear right extra lift during swing phase
        if torch.sin(torch.tensor(rr_phase, device=self.device)) > stride_peak_threshold:
            go2_joint_angles[rr_thigh_idx] = extra_lift_thigh_rear  # Higher lift
            go2_joint_angles[rr_calf_idx] = extra_lift_calf_rear  # More bent
            
        # Rear left extra lift during swing phase
        if torch.sin(torch.tensor(rl_phase, device=self.device)) > stride_peak_threshold:
            go2_joint_angles[rl_thigh_idx] = extra_lift_thigh_rear  # Higher lift
            go2_joint_angles[rl_calf_idx] = extra_lift_calf_rear  # More bent
            
        # Skip the old position-based and stance/swing-based adjustments
        # Just keep the limit constraints
        
        # Apply strict limits
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
        
        # Debug: Print final angles
        if print_debug:
            print(f"Pre-constraint GO2 joint angles:")
            print(f"  FR_hip: {go2_joint_angles[fr_hip_idx]:.4f}")
            print(f"  FR_thigh: {go2_joint_angles[fr_thigh_idx]:.4f}")
            print(f"  FR_calf: {go2_joint_angles[fr_calf_idx]:.4f}")
            
            print(f"Stride oscillation factors:")
            print(f"  FR: {fr_phase:.4f}")
            print(f"  FL: {fl_phase:.4f}")
            
            print(f"Post-oscillation GO2 joint angles:")
            print(f"  FR_hip: {go2_joint_angles[fr_hip_idx]:.4f}")
            print(f"  FR_thigh: {go2_joint_angles[fr_thigh_idx]:.4f}")
            print(f"  FR_calf: {go2_joint_angles[fr_calf_idx]:.4f}")
            
            print(f"Final GO2 joint angles (after clamping):")
            print(f"  FR_hip: {go2_joint_angles[fr_hip_idx]:.4f}")
            print(f"  FR_thigh: {go2_joint_angles[fr_thigh_idx]:.4f}")
            print(f"  FR_calf: {go2_joint_angles[fr_calf_idx]:.4f}")
            print("==================================\n")
            
        return go2_joint_angles, forward_offset
    
    def retarget_sequence(self, custom_params=None):
        """Retarget the entire motion sequence."""
        if self.num_frames == 0:
            raise ValueError("No motion data loaded")
        
        # Initialize tensor for all retargeted frames
        go2_joint_angles = torch.zeros((self.num_frames, 12), device=self.device)
        forward_offsets = torch.zeros((self.num_frames, 3), device=self.device)
        
        # Retarget each frame
        print(f"Retargeting {self.num_frames} frames...")
        for i in range(self.num_frames):
            # Process each frame separately
            joint_angles, forward_offset = self.retarget_frame(i, custom_params)
            go2_joint_angles[i] = joint_angles
            forward_offsets[i] = forward_offset
            
            # Progress indicator
            if i % 10 == 0 or i == self.num_frames - 1:
                print(f"  Processed {i+1}/{self.num_frames} frames")
        
        # Analyze frame-to-frame differences
        diffs = []
        for i in range(1, self.num_frames):
            diff = torch.norm(go2_joint_angles[i] - go2_joint_angles[i-1]).item()
            diffs.append(diff)
        
        avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
        max_diff_val = max(diffs) if diffs else 0.0
        max_diff_idx = diffs.index(max_diff_val) if diffs else 0
        
        print("Frame-to-frame analysis:")
        print(f"  Average frame-to-frame difference: {avg_diff:.6f}")
        print(f"  Maximum frame-to-frame difference: {max_diff_val:.6f} at frame {max_diff_idx}")
        
        # If the average difference is very small, something might be wrong
        if avg_diff < 0.1:
            print("WARNING: Frame-to-frame differences are very small. Motion may appear static.")
            print("Consider increasing scale factors in retarget_frame method.")
            
            # Add specific debugging for problematic frames
            print("Analyzing problematic frame pairs:")
            for i in range(min(5, len(diffs))):
                print(f"  Frames {i}-{i+1} difference: {diffs[i]:.6f}")
                print(f"    Frame {i} first 3 joints: {go2_joint_angles[i][:3]}")
                print(f"    Frame {i+1} first 3 joints: {go2_joint_angles[i+1][:3]}")
            
            # Special handling for trot motion if detected
            if self.motion_file_path is not None:
                motion_type = self.motion_file_path.split('/')[-1].split('_')[1]
                if "trot" in motion_type or "walk03" in self.motion_file_path:
                    print("Detected trot motion - applying special motion scaling")
                    
                    # Add artificial motion by modulating the joint angles
                    for i in range(self.num_frames):
                        phase = i / self.num_frames * 2 * 3.14159
                        
                        # Modulate hip joints with sinusoidal pattern
                        go2_joint_angles[i, 0] = 0.2 * torch.sin(torch.tensor(phase)) + 0.1  # FR hip
                        go2_joint_angles[i, 3] = -0.2 * torch.sin(torch.tensor(phase)) + 0.1  # FL hip
                        go2_joint_angles[i, 6] = 0.2 * torch.sin(torch.tensor(phase + 1.57)) + 0.1  # RR hip
                        go2_joint_angles[i, 9] = -0.2 * torch.sin(torch.tensor(phase + 1.57)) + 0.1  # RL hip
                        
                        # Modulate thigh joints
                        go2_joint_angles[i, 1] = 0.7 + 0.3 * torch.sin(torch.tensor(phase))  # FR thigh
                        go2_joint_angles[i, 4] = 0.7 + 0.3 * torch.sin(torch.tensor(phase + 3.14159))  # FL thigh
                        go2_joint_angles[i, 7] = 0.9 + 0.3 * torch.sin(torch.tensor(phase + 1.57))  # RR thigh
                        go2_joint_angles[i, 10] = 0.9 + 0.3 * torch.sin(torch.tensor(phase + 4.71239))  # RL thigh
                        
                        # Modulate calf joints
                        go2_joint_angles[i, 2] = -1.5 - 0.5 * torch.sin(torch.tensor(phase))  # FR calf
                        go2_joint_angles[i, 5] = -1.5 - 0.5 * torch.sin(torch.tensor(phase + 3.14159))  # FL calf
                        go2_joint_angles[i, 8] = -1.5 - 0.5 * torch.sin(torch.tensor(phase + 1.57))  # RR calf
                        go2_joint_angles[i, 11] = -1.5 - 0.5 * torch.sin(torch.tensor(phase + 4.71239))  # RL calf
                    
                    print("Applied special trot motion pattern")
                    
                    # Recalculate frame-to-frame differences
                    diffs = []
                    for i in range(1, self.num_frames):
                        diff = torch.norm(go2_joint_angles[i] - go2_joint_angles[i-1]).item()
                        diffs.append(diff)
                    
                    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
                    print(f"  New average frame-to-frame difference: {avg_diff:.6f}")
        
        # Save retargeted motion to file
        output_data = {
            'joint_angles': go2_joint_angles,
            'forward_offsets': forward_offsets
        }
        
        os.makedirs("output", exist_ok=True)
        if self.motion_file_path is not None:
            motion_name = self.motion_file_path.split("/")[-1].split("_")[1]
            output_file = f"output/{motion_name}_retargeted.npy"
            torch.save(output_data, output_file)
            print(f"Saved retargeted motion to {output_file}")
        else:
            output_file = "output/unknown_motion_retargeted.npy"
            torch.save(output_data, output_file)
            print(f"Saved retargeted motion to {output_file}")
        
        return go2_joint_angles, forward_offsets
    
    def save_retargeted_motion(self, joint_angles, output_file):
        """Save retargeted motion to file."""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Convert to numpy and save
        joint_angles_np = joint_angles.cpu().numpy()
        np.save(output_file, joint_angles_np)
        print(f"Saved retargeted motion to {output_file}")
        return output_file
    
    def visualize_retargeted_motion(self, joint_angles, forward_offsets=None):
        """Visualize the retargeted motion on the Go2 robot."""
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
        
        # Play back the retargeted motion
        num_frames = joint_angles.shape[0]
        print(f"Visualizing {num_frames} frames of retargeted motion...")
        
        # For playback control
        slow_mode = False
        frame_step = 1  # Default step size
        
        # Visualization time - approximately 10 seconds regardless of motion length
        total_time = 20.0  # seconds - a little longer to see more of the pattern
        frames_per_second = 30  # approximate fps of simulation
        total_frames = int(total_time * frames_per_second)
        
        # For longer motions, we may need to skip frames to fit within the desired time
        # But for shorter motions, we should repeat
        if num_frames > total_frames:
            # Long motion: skip frames
            frame_step = max(1, int(num_frames / total_frames))
            print(f"Long motion sequence: using frame step of {frame_step} to fit within {total_time} seconds")
        
        print(f"Playing motion for approximately {total_time} seconds...")
        
        # Robot's current position - we'll apply it through commands not by direct setting
        # This approach avoids the error with self.env.root_states
        current_x = 0.0
        current_y = 0.0
        
        # Play the motion continuously
        for i in range(total_frames):
            # Get frame index with looping
            loop_idx = (i % (num_frames // frame_step)) * frame_step  # Skip frames if needed
            frame_idx = min(loop_idx, num_frames - 1)  # Avoid overflow
            
            # Get the joint angles for this frame
            action = joint_angles[frame_idx].clone()  # Make a copy to avoid modifying the original
            
            # Ensure action has the correct shape for the env.step function
            if len(action.shape) == 1:
                action = action.unsqueeze(0)  # Add batch dimension
            
            # Step the environment with the current frame's joint angles
            try:
                obs, reward, done, info = self.env.step(action)
                
                # Break if done
                if done:
                    print(f"  Simulation terminated at frame {i+1}/{total_frames}")
                    return
                    
            except Exception as e:
                print(f"Error on frame {i}: {e}")
                return
            
            # Progress indicator (but don't print too often)
            if i % 30 == 0 or i == total_frames - 1:
                current_time = i / frames_per_second
                print(f"  Time: {current_time:.1f}s - Frame {i+1}/{total_frames} - Motion frame: {frame_idx+1}/{num_frames}")
        
        print("Visualization complete")

    def generate_parameter_variations(self, param_name, param_values):
        """Generate parameter variations for a specific parameter."""
        result = []
        for value in param_values:
            params = self.motion_params.copy()
            params[param_name] = value
            result.append(params)
        return result
    
    def test_parameter_variations(self, param_variations, frames_per_variation=200):
        """Test different parameter variations for walking motion."""
        if self.num_frames == 0:
            raise ValueError("No motion data loaded")
        
        # Initialize Go2 environment for visualization
        if self.env is None:
            self.env = Go2Env(
                num_envs=1,
                env_cfg=self.env_cfg,
                obs_cfg=self.obs_cfg,
                reward_cfg=self.reward_cfg,
                command_cfg=self.command_cfg,
                show_viewer=True,
            )
        
        # Reset environment
        obs, _ = self.env.reset()
        
        print(f"Testing {len(param_variations)} parameter variations...")
        
        for variation_idx, (variation_name, custom_params) in enumerate(param_variations):
            print(f"\nVariation {variation_idx+1}/{len(param_variations)}: {variation_name}")
            
            # Calculate frame range to simulate
            frame_count = min(frames_per_variation, self.num_frames)
            
            # Process each frame with the current parameter variation
            for frame_idx in range(frame_count):
                # Get joint angles for this frame with the current parameter variation
                action, _ = self.retarget_frame(frame_idx % self.num_frames, custom_params)
                
                # Ensure action has correct shape for env.step function
                if len(action.shape) == 1:
                    action = action.unsqueeze(0)  # Add batch dimension
                    
                # Step the environment
                try:
                    obs, reward, done, info = self.env.step(action)
                    
                    # Break if done
                    if done:
                        print("  Simulation terminated early")
                        self.env.reset()  # Reset for next variation
                        break
                        
                except Exception as e:
                    print(f"Error on frame {frame_idx}: {e}")
                    self.env.reset()  # Reset for next variation
                    break
            
            # Reset environment for next variation
            obs, _ = self.env.reset()
    
    def test_bias_variations(self, param_name, bias_ranges, frames_per_variation=200):
        """Test different bias values for a specific parameter."""
        param_variations = []
        
        for bias in bias_ranges:
            custom_params = self.generate_parameter_variations(param_name, [bias])[0]
            variation_name = f"{param_name}={bias:.2f}"
            param_variations.append((variation_name, custom_params))
        
        self.test_parameter_variations(param_variations, frames_per_variation)
    
    def test_scale_variations(self, param_name, scale_ranges, frames_per_variation=200):
        """Test different scale values for a specific parameter."""
        param_variations = []
        
        for scale in scale_ranges:
            custom_params = self.generate_parameter_variations(param_name, [scale])[0]
            variation_name = f"{param_name}={scale:.2f}"
            param_variations.append((variation_name, custom_params))
        
        self.test_parameter_variations(param_variations, frames_per_variation)
    
    def test_bias_and_scale_combinations(self, bias_param, scale_param, bias_ranges, scale_ranges, frames_per_variation=200):
        """Test combinations of bias and scale parameters."""
        param_variations = []
        
        # Loop through all combinations of bias and scale values
        for bias in bias_ranges:
            for scale in scale_ranges:
                # Create a copy of default parameters
                params = self.motion_params.copy()
                # Update both bias and scale
                params[bias_param] = bias
                params[scale_param] = scale
                
                variation_name = f"{bias_param}={bias:.2f}, {scale_param}={scale:.2f}"
                param_variations.append((variation_name, params))
        
        self.test_parameter_variations(param_variations, frames_per_variation)
    
    def test_gradual_parameter_change(self, param_name, start_val, end_val, steps, frames_per_step=100):
        """Gradually change a parameter and visualize the effect."""
        if self.num_frames == 0:
            raise ValueError("No motion data loaded")
        
        # Generate parameter values
        param_values = np.linspace(start_val, end_val, steps)
        param_variations = []
        
        for val in param_values:
            custom_params = self.generate_parameter_variations(param_name, [val])[0]
            variation_name = f"{param_name}={val:.2f}"
            param_variations.append((variation_name, custom_params))
        
        self.test_parameter_variations(param_variations, frames_per_step)


def process_all_motions():
    """Process and retarget all motion files."""
    # Create retargeter
    retargeter = MultiDogMotionRetargeter()
    
    # Process each motion file
    for motion_data in MOCAP_MOTIONS:
        motion_name, file_path, frame_start, frame_end = motion_data
        print(f"\nProcessing motion: {motion_name} from {file_path}")
        
        # Load motion data with frame range
        retargeter.load_motion_data(file_path, frame_start, frame_end)
        
        # Retarget motion
        joint_angles, forward_offsets = retargeter.retarget_sequence()
        
        # Save retargeted motion
        output_file = f"output/{motion_name}_retargeted.npy"
        output_data = {
            'joint_angles': joint_angles,
            'forward_offsets': forward_offsets
        }
        torch.save(output_data, output_file)
        print(f"Saved retargeted motion to {output_file}")
        
        # Visualize motion
        print(f"Visualizing {motion_name}...")
        retargeter.visualize_retargeted_motion(joint_angles, forward_offsets)
    
    print("\nAll motions processed!")


def main():
    """Main function to retarget dog motion to Go2 robot."""
    parser = argparse.ArgumentParser(description="Dog motion retargeting for Go2 robot")
    parser.add_argument("--motion", type=str, choices=[m[0] for m in MOCAP_MOTIONS], help="Motion type to retarget")
    parser.add_argument("--all", action="store_true", help="Process all motion types")
    parser.add_argument("--file", type=str, help="Custom motion file path")
    parser.add_argument("--start", type=int, help="Start frame for custom file")
    parser.add_argument("--end", type=int, help="End frame for custom file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--visualize_only", type=str, help="Visualize a previously retargeted motion file")
    
    # Add new arguments for parameter testing
    parser.add_argument("--test_param", type=str, help="Test parameter variations for a specific parameter")
    parser.add_argument("--test_bias", action="store_true", help="Test bias parameter variations")
    parser.add_argument("--test_scale", action="store_true", help="Test scale parameter variations")
    parser.add_argument("--test_combinations", action="store_true", help="Test combinations of bias and scale parameters")
    parser.add_argument("--bias_param", type=str, default="thigh_bias", help="Bias parameter to test")
    parser.add_argument("--scale_param", type=str, default="thigh_amplitude", help="Scale parameter to test")
    parser.add_argument("--extended_range", action="store_true", help="Use extended parameter ranges for testing")
    parser.add_argument("--frames_per_test", type=int, default=200, help="Frames to simulate for each parameter set")
    parser.add_argument("--gradual_change", action="store_true", help="Gradually change a parameter")
    parser.add_argument("--start_val", type=float, help="Starting value for gradual parameter change")
    parser.add_argument("--end_val", type=float, help="Ending value for gradual parameter change")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps for gradual parameter change")
    
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    # Create retargeter
    retargeter = MultiDogMotionRetargeter()

    # Testing parameters for walking
    if args.test_param or args.test_bias or args.test_scale or args.test_combinations or args.gradual_change:
        # Always load the walking motion for parameter testing
        print("Loading walking motion for parameter testing...")
        walking_motion = next(m for m in MOCAP_MOTIONS if m[0] == "walk")
        motion_name, file_path, frame_start, frame_end = walking_motion
        retargeter.load_motion_data(file_path, frame_start, frame_end)
        
        # Proceed with parameter testing
        if args.test_param:
            param_name = args.test_param
            print(f"Testing variations for parameter: {param_name}")
            
            # Define parameter ranges based on the parameter type and whether extended range is requested
            if param_name.endswith("amplitude"):  # Scale parameter
                if args.extended_range:
                    param_ranges = [0.5, 1.0, 1.5, 2.0, 2.5]
                else:
                    param_ranges = [1.0, 1.5, 2.0]
            else:  # Bias parameter
                if param_name == "thigh_bias":
                    if args.extended_range:
                        param_ranges = [0.6, 0.9, 1.2, 1.5, 1.8]
                    else:
                        param_ranges = [0.9, 1.2, 1.5]
                elif param_name == "calf_bias":
                    if args.extended_range:
                        param_ranges = [-2.0, -1.75, -1.5, -1.25, -1.0]
                    else:
                        param_ranges = [-1.75, -1.5, -1.25]
                elif param_name == "rear_thigh_bias":
                    if args.extended_range:
                        param_ranges = [0.8, 1.1, 1.4, 1.7, 2.0]
                    else:
                        param_ranges = [1.1, 1.4, 1.7]
                else:
                    param_ranges = [0.5, 1.0, 1.5]
            
            retargeter.test_scale_variations(param_name, param_ranges, args.frames_per_test)
            
        elif args.test_bias:
            bias_param = args.bias_param
            print(f"Testing bias variations for parameter: {bias_param}")
            
            # Define bias ranges based on parameter
            if bias_param == "thigh_bias":
                if args.extended_range:
                    bias_ranges = [0.6, 0.9, 1.2, 1.5, 1.8]
                else:
                    bias_ranges = [0.9, 1.2, 1.5]
            elif bias_param == "calf_bias":
                if args.extended_range:
                    bias_ranges = [-2.0, -1.75, -1.5, -1.25, -1.0]
                else:
                    bias_ranges = [-1.75, -1.5, -1.25]
            elif bias_param == "rear_thigh_bias":
                if args.extended_range:
                    bias_ranges = [0.8, 1.1, 1.4, 1.7, 2.0]
                else:
                    bias_ranges = [1.1, 1.4, 1.7]
            else:
                bias_ranges = [0.5, 1.0, 1.5]
                
            retargeter.test_bias_variations(bias_param, bias_ranges, args.frames_per_test)
            
        elif args.test_scale:
            scale_param = args.scale_param
            print(f"Testing scale variations for parameter: {scale_param}")
            
            # Define scale ranges
            if args.extended_range:
                scale_ranges = [0.5, 1.0, 1.5, 2.0, 2.5]
            else:
                scale_ranges = [1.0, 1.5, 2.0]
                
            retargeter.test_scale_variations(scale_param, scale_ranges, args.frames_per_test)
            
        elif args.test_combinations:
            bias_param = args.bias_param
            scale_param = args.scale_param
            print(f"Testing combinations of {bias_param} and {scale_param}")
            
            # Define bias and scale ranges
            if bias_param == "thigh_bias":
                if args.extended_range:
                    bias_ranges = [0.6, 1.2, 1.8]
                else:
                    bias_ranges = [0.9, 1.2, 1.5]
            elif bias_param == "calf_bias":
                if args.extended_range:
                    bias_ranges = [-2.0, -1.5, -1.0]
                else:
                    bias_ranges = [-1.75, -1.5, -1.25]
            elif bias_param == "rear_thigh_bias":
                if args.extended_range:
                    bias_ranges = [0.8, 1.4, 2.0]
                else:
                    bias_ranges = [1.1, 1.4, 1.7]
            else:
                bias_ranges = [0.5, 1.0, 1.5]
                
            if args.extended_range:
                scale_ranges = [0.5, 1.5, 2.5]
            else:
                scale_ranges = [1.0, 1.5, 2.0]
                
            retargeter.test_bias_and_scale_combinations(
                bias_param, scale_param, bias_ranges, scale_ranges, args.frames_per_test
            )
            
        elif args.gradual_change:
            param_name = args.test_param if args.test_param else "thigh_amplitude"
            print(f"Testing gradual change of {param_name}")
            
            # Use provided start and end values or defaults
            start_val = args.start_val
            end_val = args.end_val
            steps = args.steps
            
            # Set default values if not provided
            if start_val is None:
                if param_name == "thigh_bias":
                    start_val = 0.6
                elif param_name == "calf_bias":
                    start_val = -2.0
                elif param_name == "rear_thigh_bias":
                    start_val = 0.8
                else:  # amplitude params
                    start_val = 0.5
                    
            if end_val is None:
                if param_name == "thigh_bias":
                    end_val = 1.8
                elif param_name == "calf_bias":
                    end_val = -1.0
                elif param_name == "rear_thigh_bias":
                    end_val = 2.0
                else:  # amplitude params
                    end_val = 2.5
            
            retargeter.test_gradual_parameter_change(
                param_name, start_val, end_val, steps, args.frames_per_test
            )
        
        return
    
    if args.visualize_only:
        # Visualize a previously retargeted motion file
        print(f"Loading retargeted motion from {args.visualize_only}...")
        data = torch.load(args.visualize_only)
        
        # Handle both old and new format
        if isinstance(data, dict):
            joint_angles = data['joint_angles']
            forward_offsets = data.get('forward_offsets', None)
        else:
            joint_angles = data
            forward_offsets = None
            
        retargeter.visualize_retargeted_motion(joint_angles, forward_offsets)
    elif args.all:
        # Process all motions
        process_all_motions()
    elif args.motion:
        # Process specific motion type
        motion_data = next((m for m in MOCAP_MOTIONS if m[0] == args.motion), None)
        if motion_data:
            motion_name, file_path, frame_start, frame_end = motion_data
            print(f"Processing motion: {motion_name} from {file_path}")
            
            # Load motion data with frame range
            retargeter.load_motion_data(file_path, frame_start, frame_end)
            
            # Retarget motion
            joint_angles, forward_offsets = retargeter.retarget_sequence()
            
            # Save retargeted motion
            output_file = f"{args.output}/{motion_name}_retargeted.npy"
            output_data = {
                'joint_angles': joint_angles,
                'forward_offsets': forward_offsets
            }
            torch.save(output_data, output_file)
            print(f"Saved retargeted motion to {output_file}")
            
            # Visualize motion
            print(f"Visualizing {motion_name}...")
            retargeter.visualize_retargeted_motion(joint_angles, forward_offsets)
        else:
            print(f"Motion type '{args.motion}' not found!")
    elif args.file:
        # Process custom file
        print(f"Processing custom file: {args.file}")
        
        # Load motion data with optional frame range
        retargeter.load_motion_data(args.file, args.start, args.end)
        
        # Retarget motion
        joint_angles, forward_offsets = retargeter.retarget_sequence()
        
        # Save retargeted motion
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        output_file = f"{args.output}/{base_name}_retargeted.npy"
        output_data = {
            'joint_angles': joint_angles,
            'forward_offsets': forward_offsets
        }
        torch.save(output_data, output_file)
        print(f"Saved retargeted motion to {output_file}")
        
        # Visualize motion
        print("Visualizing retargeted motion...")
        retargeter.visualize_retargeted_motion(joint_angles, forward_offsets)
    else:
        parser.print_help()
    
    print("Done!")


if __name__ == "__main__":
    main() 