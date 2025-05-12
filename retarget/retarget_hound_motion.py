#!/usr/bin/env python3
"""
Script to retarget the hound_joint_pos.txt file to a format compatible with the Go2 robot.
This script adapts the 91-dimensional hound joint data to the 12-DOF Go2 robot format.
"""
import os
import torch
import genesis as gs
import numpy as np
from RL_training.multi_dog_walk_retarget import MultiDogMotionRetargeter

def main():
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    # Input and output file paths
    input_file = "data/hound_joint_pos.txt"
    output_file = "data/hound_joint_pos_retargeted.pt"
    
    print(f"Retargeting motion from {input_file} to Go2 robot format")
    
    # Create the retargeter
    retargeter = MultiDogMotionRetargeter()
    
    # Load the motion data from the hound file
    # The MultiDogMotionRetargeter expects a .txt file in a specific format
    # and will process it through the DogWalkParser
    retargeter.load_motion_data(input_file)
    
    # Retarget the motion sequence
    print("Retargeting motion sequence...")
    joint_angles, forward_offsets = retargeter.retarget_sequence()
    
    # Check the shape of the retargeted motion
    print(f"Retargeted motion shape: {joint_angles.shape}")
    
    # Save the retargeted motion
    output_data = {
        'joint_angles': joint_angles,
        'forward_offsets': forward_offsets
    }
    
    torch.save(output_data, output_file)
    print(f"Saved retargeted motion to {output_file}")
    
    # Optionally visualize the motion
    print("Visualizing retargeted motion...")
    retargeter.visualize_retargeted_motion(joint_angles, forward_offsets)
    
    # If you want to use this for training, you would need to load it as:
    # python genesis_motion_imitation.py --exp_name hound_motion_imitation --motion_file data/hound_joint_pos_retargeted.pt --max_iterations 300 --num_envs 256
    
    print("Retargeting complete!")

if __name__ == "__main__":
    main() 