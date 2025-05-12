import torch
import numpy as np
import math
import os
import argparse
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from go2_env import Go2Env
from txt_motion_parser import TxtMotionParser

class MocapData:
    """Class to load and process mocap data from TXT files."""
    def __init__(self, txt_file, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.txt_file = txt_file
        self.joint_positions = None
        self.joint_rotations = None
        self.frame_time = None
        self.num_frames = None
        self.joints = []
        self.joint_map = {}
        self.load_txt()
    
    def load_txt(self):
        """Load TXT file and extract joint positions and rotations."""
        print(f"Loading mocap data from {self.txt_file}...")
        
        # Use our TXT parser
        parser = TxtMotionParser(device=self.device)
        parser.parse_file(self.txt_file)
        data = parser.get_data()
        
        # Extract data
        self.joint_positions = data['joint_positions']
        self.joint_rotations = data['joint_rotations']
        self.frame_time = data['frame_time']
        self.num_frames = data['num_frames']
        
        # Create simplified joint structure
        self._create_joints()
        
        print(f"Loaded {self.num_frames} frames at {1/self.frame_time}fps")
    
    def _create_joints(self):
        """Create simplified joint structure for compatibility with retargeting code."""
        # Define simplified joint names based on a dog skeleton
        joint_names = [
            "Root",
            "FR_Shoulder", "FR_Knee", "FR_Foot",
            "FL_Shoulder", "FL_Knee", "FL_Foot",
            "BR_Hip", "BR_Knee", "BR_Foot",
            "BL_Hip", "BL_Knee", "BL_Foot"
        ]
        
        # Create Joint objects for compatibility
        class SimpleJoint:
            def __init__(self, name, parent=None):
                self.name = name
                self.parent = parent
                self.children = []
                
            def add_child(self, child):
                self.children.append(child)
                
            def __str__(self):
                return f"Joint: {self.name}"
        
        # Create joints and establish hierarchy
        parent_indices = [
            -1,  # 0: Root has no parent
            0,   # 1: Front right shoulder -> Root
            1,   # 2: Front right knee -> Front right shoulder
            2,   # 3: Front right foot -> Front right knee
            0,   # 4: Front left shoulder -> Root
            4,   # 5: Front left knee -> Front left shoulder
            5,   # 6: Front left foot -> Front left knee
            0,   # 7: Back right hip -> Root
            7,   # 8: Back right knee -> Back right hip
            8,   # 9: Back right foot -> Back right knee
            0,   # 10: Back left hip -> Root
            10,  # 11: Back left knee -> Back left hip
            11,  # 12: Back left foot -> Back left knee
        ]
        
        # Create joint objects
        num_joints = min(len(joint_names), self.joint_positions.shape[1])
        joint_objects = [SimpleJoint(joint_names[i]) for i in range(num_joints)]
        
        # Set up parent-child relationships
        for i in range(num_joints):
            parent_idx = parent_indices[i]
            if parent_idx >= 0 and parent_idx < num_joints:
                joint_objects[i].parent = joint_objects[parent_idx]
                joint_objects[parent_idx].add_child(joint_objects[i])
        
        # Store joints and create joint map
        self.joints = joint_objects
        self.joint_map = {joint.name: i for i, joint in enumerate(joint_objects)}


class Go2Retargeter:
    """Main class for retargeting motion capture data to Go2 robot."""
    def __init__(self, env_cfg=None, obs_cfg=None, reward_cfg=None, command_cfg=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize default configs if not provided
        if env_cfg is None:
            self.env_cfg = {
                "num_actions": 12,
                "default_joint_angles": {
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
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ],
                "kp": 20.0,
                "kd": 0.5,
                "termination_if_roll_greater_than": 10,
                "termination_if_pitch_greater_than": 10,
                "base_init_pos": [0.0, 0.0, 0.42],
                "base_init_quat": [1.0, 0.0, 0.0, 0.0],
                "episode_length_s": 20.0,
                "resampling_time_s": 4.0,
                "action_scale": 0.25,
                "simulate_action_latency": True,
                "clip_actions": 100.0,
            }
        else:
            self.env_cfg = env_cfg
            
        if obs_cfg is None:
            self.obs_cfg = {
                "num_obs": 45,
                "obs_scales": {
                    "lin_vel": 2.0,
                    "ang_vel": 0.25,
                    "dof_pos": 1.0,
                    "dof_vel": 0.05,
                },
            }
        else:
            self.obs_cfg = obs_cfg
            
        if reward_cfg is None:
            self.reward_cfg = {
                "tracking_sigma": 0.25,
                "base_height_target": 0.3,
                "feet_height_target": 0.075,
                "reward_scales": {},
            }
        else:
            self.reward_cfg = reward_cfg
            
        if command_cfg is None:
            self.command_cfg = {
                "num_commands": 3,
                "lin_vel_x_range": [0.0, 0.0],
                "lin_vel_y_range": [0.0, 0.0],
                "ang_vel_range": [0.0, 0.0],
            }
        else:
            self.command_cfg = command_cfg
        
        # Initialize the environment for visualization
        self.env = Go2Env(
            num_envs=1,
            env_cfg=self.env_cfg,
            obs_cfg=self.obs_cfg,
            reward_cfg=self.reward_cfg,
            command_cfg=self.command_cfg,
            show_viewer=True,
        )
        
        # Neural network for motion retargeting
        self.retarget_network = self._create_retarget_network()
        
        # For saving retargeted motions
        self.motion_parser = None
        
        # For joint mapping
        self.joint_mapping = None
        self.joint_index_mapping = {}
    
    def _create_retarget_network(self):
        """Create a neural network for motion retargeting."""
        # Simple feedforward network to map from mocap pose to robot pose
        network = torch.nn.Sequential(
            torch.nn.Linear(12 * 7, 128),  # 12 joints with pos(3) + rot(4)
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 12),  # 12 joint angles for Go2
            torch.nn.Tanh()  # Output normalized joint angles
        ).to(self.device)
        
        return network
    
    def create_joint_mapping(self, mocap_data):
        """Create a mapping between mocap joints and robot joints."""
        # Define go2 robot joint names 
        go2_joint_names = self.env_cfg["joint_names"]
        
        # Print mocap joint names for debugging
        print("Mocap joints:")
        for i, joint in enumerate(mocap_data.joints):
            print(f"  {i}: {joint.name}")
        
        # Print Go2 joint names for debugging
        print("Go2 joints:")
        for i, name in enumerate(go2_joint_names):
            print(f"  {i}: {name}")
        
        # Create a mapping from mocap joints to Go2 robot joints
        mapping = {
            # Map front right leg joints
            "FR_Shoulder": "FR_hip_joint",
            "FR_Knee": "FR_thigh_joint",
            "FR_Foot": "FR_calf_joint",
            
            # Map front left leg joints
            "FL_Shoulder": "FL_hip_joint",
            "FL_Knee": "FL_thigh_joint", 
            "FL_Foot": "FL_calf_joint",
            
            # Map back right leg joints
            "BR_Hip": "RR_hip_joint",
            "BR_Knee": "RR_thigh_joint",
            "BR_Foot": "RR_calf_joint",
            
            # Map back left leg joints
            "BL_Hip": "RL_hip_joint",
            "BL_Knee": "RL_thigh_joint",
            "BL_Foot": "RL_calf_joint",
        }
        
        # Store the joint mapping
        self.joint_mapping = mapping
        
        # Create a mapping from mocap joint indices to robot joint indices
        self.joint_index_mapping = {}
        
        for mocap_idx, joint in enumerate(mocap_data.joints):
            if joint.name in mapping:
                robot_joint_name = mapping[joint.name]
                if robot_joint_name in go2_joint_names:
                    robot_idx = go2_joint_names.index(robot_joint_name)
                    self.joint_index_mapping[mocap_idx] = robot_idx
        
        print(f"Created mapping between {len(self.joint_index_mapping)} joints")
        return mapping
    
    def retarget_frame(self, mocap_frame):
        """Retarget a single frame of mocap data to the Go2 robot."""
        # Extract joint positions and rotations from the mocap frame
        positions = mocap_frame["positions"]  # shape: [num_joints, 3]
        rotations = mocap_frame["rotations"]  # shape: [num_joints, 4]
        
        # Flatten the mocap data
        mocap_features = torch.cat([positions.flatten(), rotations.flatten()], dim=0)
        
        # Run through the retargeting network
        with torch.no_grad():
            robot_joint_angles = self.retarget_network(mocap_features)
        
        return robot_joint_angles
    
    def manually_retarget_frame(self, mocap_data, frame_idx):
        """Manually retarget a frame based on joint orientations."""
        # This method implements direct joint angle copying between mapped joints
        # It's a simpler approach than using a neural network
        
        # Initialize robot joint angles with default values
        robot_joint_angles = torch.zeros(12, device=self.device)
        
        # For each mapped joint
        for mocap_idx, robot_idx in self.joint_index_mapping.items():
            # Get the quaternion rotation of the mocap joint
            quat = mocap_data.joint_rotations[frame_idx, mocap_idx]
            
            # Convert quaternion to Euler angles
            euler = self._quaternion_to_euler(quat)
            
            # Extract the most relevant angle for each joint type
            # This is a simplified approach - in a real implementation, you would
            # do a proper transformation based on the joint types and orientations
            
            joint_name = self.env_cfg["joint_names"][robot_idx]
            
            if "hip" in joint_name:
                # Hip joints control lateral movement
                angle = euler[1]  # Y angle
            elif "thigh" in joint_name:
                # Thigh joints control forward movement
                angle = euler[0]  # X angle
            elif "calf" in joint_name:
                # Calf joints control forward movement
                angle = euler[0]  # X angle
            else:
                angle = 0.0
            
            # Scale the angle to the robot's range
            # This is a placeholder - you would need to scale properly for each joint
            scaled_angle = angle * 0.5
            
            # Store the angle
            robot_joint_angles[robot_idx] = scaled_angle
        
        # Return the retargeted joint angles
        return robot_joint_angles
    
    def retarget_sequence(self, mocap_data):
        """Retarget a sequence of mocap frames to the Go2 robot."""
        # Create joint mapping if not already done
        if self.joint_mapping is None:
            self.create_joint_mapping(mocap_data)
        
        # Store motion parser for later use in saving
        self.motion_parser = TxtMotionParser(device=self.device)
        
        num_frames = mocap_data.num_frames
        robot_joint_angles = torch.zeros((num_frames, 12), device=self.device)
        
        # Initially, use manual retargeting instead of the neural network
        print("Retargeting motion sequence...")
        for i in range(num_frames):
            # Use manual retargeting for now
            robot_joint_angles[i] = self.manually_retarget_frame(mocap_data, i)
            
            # Progress indicator
            if i % 100 == 0:
                print(f"  Processed {i}/{num_frames} frames")
        
        return robot_joint_angles
    
    def _quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles (XYZ order)."""
        # Quaternion components
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw])
    
    def visualize_retargeted_motion(self, robot_joint_angles):
        """Visualize the retargeted motion on the Go2 robot."""
        # Reset the environment
        obs, _ = self.env.reset()
        
        # Play back the retargeted motion
        num_frames = robot_joint_angles.shape[0]
        print(f"Visualizing {num_frames} frames of retargeted motion...")
        
        for i in range(num_frames):
            # Scale the joint angles to the robot's action space
            action = robot_joint_angles[i]
            
            # Ensure action has the correct shape for the env.step function
            # The env expects a 2D tensor [num_envs, num_actions]
            if len(action.shape) == 1:
                action = action.unsqueeze(0)  # Add batch dimension
            
            # Step the environment
            try:
                obs, reward, done, info = self.env.step(action)
                
                # Break if done
                if done:
                    print("  Simulation terminated early")
                    break
                    
            except Exception as e:
                print(f"Error on frame {i}: {e}")
                break
            
            # Progress indicator
            if i % 100 == 0:
                print(f"  Frame {i}/{num_frames}")
        
        print("Visualization complete")
    
    def train_retargeting_network(self, mocap_data, num_epochs=100, learning_rate=1e-4):
        """Train the retargeting network using mocap data."""
        print("Training retargeting network...")
        
        # Create joint mapping if not already done
        if self.joint_mapping is None:
            self.create_joint_mapping(mocap_data)
        
        # Define optimizer
        optimizer = torch.optim.Adam(self.retarget_network.parameters(), lr=learning_rate)
        
        # Generate target poses using manual retargeting
        target_poses = torch.zeros((mocap_data.num_frames, 12), device=self.device)
        for i in range(mocap_data.num_frames):
            target_poses[i] = self.manually_retarget_frame(mocap_data, i)
        
        # Main training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            # Process each frame
            for i in range(mocap_data.num_frames):
                # Create frame dictionary
                frame = {
                    "positions": mocap_data.joint_positions[i],
                    "rotations": mocap_data.joint_rotations[i]
                }
                
                # Flatten the mocap data
                mocap_features = torch.cat([
                    frame["positions"].flatten(),
                    frame["rotations"].flatten()
                ], dim=0)
                
                # Forward pass through network
                robot_joint_angles = self.retarget_network(mocap_features)
                
                # Define loss: compare with manually retargeted poses
                loss = torch.mean(torch.square(robot_joint_angles - target_poses[i]))
                
                # Add regularization to penalize extreme joint angles
                reg_loss = 0.1 * torch.mean(torch.square(robot_joint_angles))
                loss += reg_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/mocap_data.num_frames:.4f}")
        
        print("Training complete!")
    
    def save_retargeted_motion(self, robot_joint_angles, output_file):
        """Save the retargeted motion to a file."""
        # Convert joint angles to numpy
        joint_angles_np = robot_joint_angles.cpu().numpy()
        
        # Save as numpy file
        np.save(output_file + ".npy", joint_angles_np)
        print(f"Saved retargeted motion to {output_file}.npy")
        
        return output_file + ".npy"


def main():
    """Main function to demonstrate motion retargeting."""
    parser = argparse.ArgumentParser(description="Motion retargeting for Go2 robot")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to motion capture file (TXT format)")
    parser.add_argument("--output_file", type=str, default="retargeted_motion", help="Output filename (without extension)")
    parser.add_argument("--train", action="store_true", help="Train the retargeting network")
    parser.add_argument("--visualize", action="store_true", help="Visualize the retargeted motion")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init()
    
    # Load mocap data
    mocap_data = MocapData(args.motion_file)
    
    # Create retargeter
    retargeter = Go2Retargeter()
    
    # Train if requested
    if args.train:
        retargeter.train_retargeting_network(
            mocap_data, 
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
    
    # Retarget motion
    robot_joint_angles = retargeter.retarget_sequence(mocap_data)
    
    # Save retargeted motion
    output_file = retargeter.save_retargeted_motion(robot_joint_angles, args.output_file)
    
    # Visualize if requested
    if args.visualize:
        retargeter.visualize_retargeted_motion(robot_joint_angles)
    
    print("Motion retargeting complete!")


if __name__ == "__main__":
    main() 