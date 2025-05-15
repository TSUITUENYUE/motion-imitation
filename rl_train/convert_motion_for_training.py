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
import glob # Added for finding files

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
                     If None, will use same name with .npy extension in the same directory.
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
    try:
        with open(input_file, 'r') as f:
            motion_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_file}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return None
        
    # Extract frames
    if "Frames" not in motion_data:
        print(f"Error: 'Frames' key not found in {input_file}")
        return None
        
    frames = motion_data["Frames"]
    
    # Extract joint angles and positions from each frame
    # Each frame has format: [root_x, root_y, root_z, quat_w, quat_x, quat_y, quat_z, joint1, joint2, ...]
    joint_angles_list = []
    
    for i, frame in enumerate(frames):
        if len(frame) < 7: # 3 for pos, 4 for quat
            print(f"Warning: Frame {i} in {input_file} has fewer than 7 elements, skipping this frame.")
            continue
            
        # Extract joint angles (elements 7+)
        angles = frame[7:]
        
        if len(angles) != len(VISUALIZATION_JOINT_ORDER):
            print(f"Warning: Frame {i} in {input_file} has {len(angles)} angle values, expected {len(VISUALIZATION_JOINT_ORDER)}. Skipping remapping if applicable.")
            joint_angles_list.append(angles) # Add as is, or handle error differently
            continue

        # Remap joint angles if needed
        if remap_joints:
            try:
                angles = remap_joint_angles(angles, VISUALIZATION_JOINT_ORDER, TRAINING_JOINT_ORDER)
            except IndexError as e:
                print(f"Error remapping joints for frame {i} in {input_file}: {e}. Raw angles: {frame[7:]}")
                continue # Skip this frame or handle error
            
        joint_angles_list.append(angles)
    
    if not joint_angles_list:
        print(f"No valid joint angles extracted from {input_file}. Output file not created.")
        return None

    # Convert to numpy arrays
    try:
        joint_angles_np = np.array(joint_angles_list, dtype=np.float32)
    except ValueError as e:
        print(f"Error converting data to NumPy array for {input_file}: {e}")
        return None

    # Calculate joint velocities for cyclic motion
    joint_velocities_np = np.zeros_like(joint_angles_np)
    if joint_angles_np.shape[0] > 1:
        # Calculate velocities as joint angle differences between frames
        joint_velocities_np[1:] = joint_angles_np[1:] - joint_angles_np[:-1]
        
        # For cyclic motion: set first frame velocity assuming loop from last to first
        joint_velocities_np[0] = joint_angles_np[0] - joint_angles_np[-1]
    
    # Save as .npy file with both joint angles and joint velocities
    try:
        output_data = {
            'joint_angles': joint_angles_np,
            'joint_velocities': joint_velocities_np
        }
        np.save(output_file, output_data)
    except Exception as e:
        print(f"Error saving NumPy file {output_file}: {e}")
        return None
        
    print(f"Converted {joint_angles_np.shape[0]} frames with {joint_angles_np.shape[1] if joint_angles_np.ndim > 1 else 'N/A'} joint angles per frame")
    print(f"Included joint velocities for frame-to-frame transitions (better for cyclic motion)")
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
    # Ensure all source joints are in target for this printout to be safe
    # This is just for display, actual remapping handles missing joints
    for src_idx, src_joint in enumerate(VISUALIZATION_JOINT_ORDER):
        if src_joint in TRAINING_JOINT_ORDER:
            tgt_idx = TRAINING_JOINT_ORDER.index(src_joint)
            print(f"{src_idx}: {src_joint} -> {tgt_idx}: {src_joint}")
        else:
            print(f"{src_idx}: {src_joint} -> Not in target order")

def main():
    parser = argparse.ArgumentParser(description="Convert motion files from JSON (.txt) to NumPy (.npy) for RL training. \
                                                 Processes .txt files in a 'data' subdirectory relative to the script location.")
    parser.add_argument("--no-remap", action="store_true", help="Skip joint remapping")
    args = parser.parse_args()
    
    # Determine the script's directory and the data subdirectory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(script_dir, 'data')

    # Check if data directory exists
    if not os.path.isdir(data_dir_path):
        print(f"Error: Data directory {data_dir_path} not found or is not a directory")
        print("Please ensure a 'data' subdirectory exists in the same directory as the script, containing your .txt motion files.")
        return
    
    # Print joint order information
    if not args.no_remap:
        print_joint_orders()
    
    input_txt_files = glob.glob(os.path.join(data_dir_path, '*.txt'))
    
    if not input_txt_files:
        print(f"No .txt files found in {data_dir_path}")
        return
        
    print(f"\nFound {len(input_txt_files)} .txt files to process in {data_dir_path}")
    
    successful_conversions = 0
    failed_conversions = 0
    
    for txt_file_path in input_txt_files:
        # Output .npy file will be in the same directory with the same base name
        output_npy_path = os.path.splitext(txt_file_path)[0] + ".npy"
        
        result_path = convert_motion_file(txt_file_path, output_npy_path, remap_joints=not args.no_remap)
        if result_path:
            successful_conversions += 1
        else:
            failed_conversions +=1
            
    print(f"\nConversion summary:")
    print(f"Successfully converted: {successful_conversions} files.")
    print(f"Failed to convert: {failed_conversions} files.")
    
    if successful_conversions > 0:
        print("\nTo use these motion files for training, you can typically point your training script to one of the generated .npy files, for example:")
        print(f"python train.py --motion_file {os.path.join(data_dir_path, 'your_motion_file.npy')} ...")
    else:
        print("\nNo files were successfully converted.")

if __name__ == "__main__":
    main() 