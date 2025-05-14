#!/usr/bin/env python3
"""
Convert motion files from JSON format (.txt) to NumPy (.npy) format for RL training.
This script extracts the joint angles from the retargeted motion files and saves them
in a format compatible with the training script.
"""
import os
import argparse
import json
import numpy as np
import torch

# Joint order in visualization script (from visualize_retarget.py)
VISUALIZATION_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

# Joint order in training script (from train.py)
TRAINING_JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
]

def remap_joint_angles(angles, source_order, target_order):
    """
    Remap joint angles from source order to target order.
    
    Args:
        angles: List or array of joint angles
        source_order: List of joint names in the source order
        target_order: List of joint names in the target order
        
    Returns:
        Remapped joint angles
    """
    # Create mapping from source indices to target indices
    index_map = {}
    for i, joint in enumerate(source_order):
        if joint in target_order:
            index_map[i] = target_order.index(joint)
    
    # Remap angles
    remapped = [0] * len(target_order)
    for src_idx, tgt_idx in index_map.items():
        remapped[tgt_idx] = angles[src_idx]
    
    return remapped

def convert_motion_file(input_file, output_file=None, remap_joints=True):
    """
    Convert a motion file from JSON format to NumPy format.
    
    Args:
        input_file: Path to the input JSON motion file (.txt)
        output_file: Path to save the output NumPy file (.npy)
                     If None, will use same name with .npy extension
        remap_joints: Whether to remap joint angles for training compatibility
    
    Returns:
        Path to the output file
    """
    # Determine output file path if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.npy"
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Load JSON motion data
    with open(input_file, 'r') as f:
        motion_data = json.load(f)
    
    # Extract frames
    frames = motion_data["Frames"]
    
    # Extract joint angles from each frame
    # Each frame has format: [root_x, root_y, root_z, quat_x, quat_y, quat_z, quat_w, joint1, joint2, ...]
    # We need to extract only the joint angles (elements 7+)
    joint_angles = []
    for frame in frames:
        # Extract only the joint angles (skipping root position and orientation)
        angles = frame[7:]
        
        # Remap joint angles if needed
        if remap_joints:
            angles = remap_joint_angles(angles, VISUALIZATION_JOINT_ORDER, TRAINING_JOINT_ORDER)
            
        joint_angles.append(angles)
    
    # Convert to numpy array
    joint_angles_np = np.array(joint_angles, dtype=np.float32)
    
    # Save as .npy file
    np.save(output_file, joint_angles_np)
    
    print(f"Converted {len(frames)} frames with {len(joint_angles[0])} joint angles per frame")
    print(f"Output saved to {output_file}")
    
    return output_file

def convert_motion_file_torch(input_file, output_file=None, remap_joints=True):
    """
    Convert a motion file from JSON format to PyTorch tensor format.
    
    Args:
        input_file: Path to the input JSON motion file (.txt)
        output_file: Path to save the output PyTorch file (.pt)
                     If None, will use same name with .pt extension
        remap_joints: Whether to remap joint angles for training compatibility
    
    Returns:
        Path to the output file
    """
    # Determine output file path if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.pt"
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Load JSON motion data
    with open(input_file, 'r') as f:
        motion_data = json.load(f)
    
    # Extract frames
    frames = motion_data["Frames"]
    
    # Extract joint angles from each frame
    # Each frame has format: [root_x, root_y, root_z, quat_x, quat_y, quat_z, quat_w, joint1, joint2, ...]
    # We need to extract only the joint angles (elements 7+)
    joint_angles = []
    for frame in frames:
        # Extract only the joint angles (skipping root position and orientation)
        angles = frame[7:]
        
        # Remap joint angles if needed
        if remap_joints:
            angles = remap_joint_angles(angles, VISUALIZATION_JOINT_ORDER, TRAINING_JOINT_ORDER)
            
        joint_angles.append(angles)
    
    # Convert to torch tensor
    joint_angles_tensor = torch.tensor(joint_angles, dtype=torch.float32)
    
    # Save as .pt file
    torch.save(joint_angles_tensor, output_file)
    
    print(f"Converted {len(frames)} frames with {len(joint_angles[0])} joint angles per frame")
    print(f"Output saved to {output_file}")
    
    return output_file

def print_joint_orders():
    """Print the joint orders used in visualization and training for comparison."""
    print("\nJoint order mapping:")
    print("--------------------")
    print("Visualization script joint order (source):")
    for i, joint in enumerate(VISUALIZATION_JOINT_ORDER):
        print(f"{i}: {joint}")
    
    print("\nTraining script joint order (target):")
    for i, joint in enumerate(TRAINING_JOINT_ORDER):
        print(f"{i}: {joint}")
    
    print("\nRemapping:")
    for src_idx, src_joint in enumerate(VISUALIZATION_JOINT_ORDER):
        tgt_idx = TRAINING_JOINT_ORDER.index(src_joint)
        print(f"{src_idx}: {src_joint} -> {tgt_idx}: {src_joint}")

def main():
    parser = argparse.ArgumentParser(description="Convert motion files from JSON to NumPy/PyTorch for RL training")
    parser.add_argument("--file", type=str, required=True, help="Path to the input motion file (.txt)")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output file (.npy or .pt)")
    parser.add_argument("--format", type=str, choices=["numpy", "torch"], default="numpy", 
                         help="Output format: numpy (.npy) or torch (.pt)")
    parser.add_argument("--no-remap", action="store_true", help="Skip joint remapping")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.file):
        print(f"Error: Input file {args.file} not found")
        return
    
    # Check output path
    output_file = args.output
    
    # Print joint order information
    print_joint_orders()
    
    # Convert based on format
    if args.format == "numpy":
        if output_file is None:
            output_file = os.path.splitext(args.file)[0] + ".npy"
        convert_motion_file(args.file, output_file, remap_joints=not args.no_remap)
    else:  # torch
        if output_file is None:
            output_file = os.path.splitext(args.file)[0] + ".pt"
        convert_motion_file_torch(args.file, output_file, remap_joints=not args.no_remap)
    
    # Provide command example
    print("\nTo use this motion file for training, run:")
    print(f"python train.py --file {output_file} --envs 64 --iters 500")

if __name__ == "__main__":
    main() 