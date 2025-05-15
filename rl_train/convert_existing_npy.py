#!/usr/bin/env python3
"""
Convert existing .npy motion files to the new format with position and velocity data.
This script adds dummy position/velocity data if needed to maintain compatibility with 
the updated training script.
"""
import os
import argparse
import numpy as np
import glob
import torch

def convert_legacy_npy(input_file, output_file=None, overwrite=False):
    """
    Convert a legacy NPY motion file to the new format with velocity data.
    For short cyclic motions, we focus on relative frame-to-frame velocity.
    
    Args:
        input_file: Path to the input NPY motion file
        output_file: Path to save the output NPY file. If None, will use the same path.
        overwrite: Whether to overwrite the input file (if output_file is None)
    
    Returns:
        Path to the output file
    """
    if output_file is None:
        if overwrite:
            output_file = input_file
        else:
            # Add "_new" suffix
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_new.npy"
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Load the legacy NPY file
    try:
        data = np.load(input_file, allow_pickle=True)
        
        # Check if already in new format
        if isinstance(data, np.ndarray) and data.dtype == np.dtype('O') and isinstance(data.item(), dict):
            print(f"File {input_file} appears to already be in the new format. Skipping.")
            return None
        
        # Otherwise, assume it's just joint angles
        joint_angles = data
        num_frames = joint_angles.shape[0]
        
        print(f"Found {num_frames} frames of joint angles")
        
        # Calculate frame-to-frame joint angle changes as a measure of velocity
        # This is more appropriate for short cyclic motions than absolute position
        joint_velocities = np.zeros_like(joint_angles)
        if num_frames > 1:
            # Calculate velocities as joint angle differences between frames
            # First frame velocity is set to same as second frame for consistency
            joint_velocities[1:] = joint_angles[1:] - joint_angles[:-1]
            joint_velocities[0] = joint_velocities[1]  # Copy second frame to first
            
            # For the last frame, calculate velocity that would continue the cycle
            # by connecting back to the first frame (for cyclic motions)
            # Improves velocity consistency at loop boundaries
            loop_velocity = joint_angles[0] - joint_angles[-1]
            joint_velocities[-1] = loop_velocity
        
        # Create new format dictionary - we don't track positions at all
        new_data = {
            'joint_angles': joint_angles,
            'joint_velocities': joint_velocities
        }
        
        # Save in new format
        np.save(output_file, new_data)
        print(f"Saved converted file to {output_file}")
        print(f"Note: This format focuses on frame-to-frame joint velocities for short cyclic motions")
        return output_file
        
    except Exception as e:
        print(f"Error converting file {input_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert legacy NPY motion files to new format with position and velocity data")
    parser.add_argument("--file", type=str, help="Single NPY file to convert")
    parser.add_argument("--dir", type=str, help="Directory containing NPY files to convert")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original files")
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        convert_legacy_npy(args.file, overwrite=args.overwrite)
    
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory {args.dir} not found")
            return
        
        npy_files = glob.glob(os.path.join(args.dir, "*.npy"))
        if not npy_files:
            print(f"No .npy files found in {args.dir}")
            return
        
        print(f"Found {len(npy_files)} .npy files to convert")
        
        success_count = 0
        for npy_file in npy_files:
            result = convert_legacy_npy(npy_file, overwrite=args.overwrite)
            if result:
                success_count += 1
        
        print(f"Successfully converted {success_count} out of {len(npy_files)} files")
    
    else:
        print("Error: Please specify either --file or --dir")
        parser.print_help()

if __name__ == "__main__":
    main() 