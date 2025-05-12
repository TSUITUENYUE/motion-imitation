import torch
import numpy as np
import genesis as gs

class DogWalkParser:
    """Parser specifically for dog_walk01_pose.txt file format."""
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Motion data attributes
        self.raw_data = None
        self.num_frames = 0
        self.frame_time = 0.01  # Default 100 fps
        self.params_per_frame = 0
        
        # Processed motion data
        self.joint_positions = None
        self.joint_rotations = None
        self.joint_velocities = None
        
        # Mapping from frame data to joint structure
        self.joint_map = self._create_default_joint_map()
    
    def _create_default_joint_map(self):
        """
        Create a default mapping from the data columns to joint structure.
        Based on analysis of the dog_walk01_pose.txt file.
        """
        # The file has rows representing frames and 87 columns of values:
        # Looking at the file patterns, we can map as follows:
        return {
            # Root/body position and orientation
            "root_pos": [0, 1, 2],  # columns for root position (x, y, z)
            "root_rot": [3, 4, 5, 6],  # columns for root orientation (quat)
            
            # Front right leg (3 joints)
            "FR_hip": { 
                "pos": [7, 8, 9],
                "rot": [10, 11, 12, 13]
            },
            "FR_knee": {
                "pos": [14, 15, 16],
                "rot": [17, 18, 19, 20]
            },
            "FR_ankle": {
                "pos": [21, 22, 23],
                "rot": [24, 25, 26, 27]
            },
            
            # Front left leg (3 joints)
            "FL_hip": {
                "pos": [28, 29, 30],
                "rot": [31, 32, 33, 34]
            },
            "FL_knee": {
                "pos": [35, 36, 37],
                "rot": [38, 39, 40, 41]
            },
            "FL_ankle": {
                "pos": [42, 43, 44],
                "rot": [45, 46, 47, 48]
            },
            
            # Back right leg (3 joints)
            "BR_hip": {
                "pos": [49, 50, 51],
                "rot": [52, 53, 54, 55]
            },
            "BR_knee": {
                "pos": [56, 57, 58], 
                "rot": [59, 60, 61, 62]
            },
            "BR_ankle": {
                "pos": [63, 64, 65],
                "rot": [66, 67, 68, 69]
            },
            
            # Back left leg (3 joints)
            "BL_hip": {
                "pos": [70, 71, 72],
                "rot": [73, 74, 75, 76]
            },
            "BL_knee": {
                "pos": [77, 78, 79],
                "rot": [80, 81, 82, 83]
            },
            "BL_ankle": {
                "pos": [84, 85, 86],
                "rot": [87, 88, 89, 90]
            }
        }
    
    def parse_file(self, file_path):
        """Parse the dog_walk01_pose.txt file and extract motion data."""
        print(f"Parsing dog walk data from {file_path}...")
        
        # Read all lines and parse data
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Process each line (frame) of data
        data_rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Split by commas, remove any whitespace, and convert to float
            values = [float(val.strip()) for val in line.split(',')]
            data_rows.append(values)
        
        # Convert to numpy array
        self.raw_data = np.array(data_rows)
        self.num_frames = len(data_rows)
        self.params_per_frame = self.raw_data.shape[1] if self.num_frames > 0 else 0
        
        print(f"Parsed {self.num_frames} frames with {self.params_per_frame} parameters per frame")
        
        # Process the parsed data into structured joint data
        self._process_motion_data()
        
        return self
    
    def _process_motion_data(self):
        """Process raw motion data into joint positions and rotations."""
        if self.raw_data is None or self.num_frames == 0:
            return
        
        # Count the number of joints from our mapping
        num_joints = 1 + sum(1 for k in self.joint_map.keys() if not k.startswith("root") and isinstance(self.joint_map[k], dict))
        
        # Initialize tensors for structured joint data
        self.joint_positions = torch.zeros((self.num_frames, num_joints, 3), device=self.device)
        self.joint_rotations = torch.zeros((self.num_frames, num_joints, 4), device=self.device)
        
        # Set default quaternion (identity rotation) for all joints
        self.joint_rotations[:, :, 0] = 1.0  # w component = 1.0 for identity rotation
        
        # Convert numpy data to tensors for processing
        raw_data_tensor = torch.tensor(self.raw_data, device=self.device)
        
        # Process data for each frame
        for frame_idx in range(self.num_frames):
            # Get data for this frame
            frame_data = raw_data_tensor[frame_idx]
            
            # Process root position (joint index 0)
            root_pos_indices = self.joint_map["root_pos"]
            if max(root_pos_indices) < len(frame_data):
                self.joint_positions[frame_idx, 0] = frame_data[root_pos_indices]
            
            # Process root rotation
            root_rot_indices = self.joint_map["root_rot"]
            if max(root_rot_indices) < len(frame_data):
                self.joint_rotations[frame_idx, 0] = frame_data[root_rot_indices]
            
            # Process joint data
            joint_idx = 1  # Start after root
            for key, data in self.joint_map.items():
                if key.startswith("root"):
                    continue  # Skip root, already processed
                    
                # Position
                pos_indices = data["pos"]
                if max(pos_indices) < len(frame_data):
                    self.joint_positions[frame_idx, joint_idx] = frame_data[pos_indices]
                
                # Rotation
                rot_indices = data["rot"]
                if max(rot_indices) < len(frame_data):
                    self.joint_rotations[frame_idx, joint_idx] = frame_data[rot_indices]
                
                joint_idx += 1
        
        # Compute joint velocities
        self._compute_velocities()
        
        print(f"Processed motion data into {num_joints} joints across {self.num_frames} frames")
    
    def _compute_velocities(self):
        """Compute joint velocities from positions."""
        if self.joint_positions is None:
            return
            
        # Initialize velocities tensor
        num_joints = self.joint_positions.shape[1]
        self.joint_velocities = torch.zeros((self.num_frames, num_joints, 3), device=self.device)
        
        # Compute velocities as difference between consecutive frames
        if self.num_frames > 1:
            # For all frames except the first one
            self.joint_velocities[1:] = (self.joint_positions[1:] - self.joint_positions[:-1]) / self.frame_time
            
            # For the first frame, use same velocity as second frame
            if self.num_frames > 1:
                self.joint_velocities[0] = self.joint_velocities[1]
    
    def get_joint_names(self):
        """Get a list of joint names based on the joint map."""
        joint_names = ["root"]
        for key in self.joint_map.keys():
            if not key.startswith("root") and isinstance(self.joint_map[key], dict):
                joint_names.append(key)
        return joint_names
    
    def get_data(self):
        """Return the parsed motion data for use in retargeting."""
        return {
            'joint_positions': self.joint_positions,
            'joint_rotations': self.joint_rotations,
            'joint_velocities': self.joint_velocities,
            'joint_names': self.get_joint_names(),
            'num_frames': self.num_frames,
            'frame_time': self.frame_time,
            'raw_data': self.raw_data
        }
    
    def save_as_npy(self, filename):
        """Save motion data to a NPY file."""
        if self.raw_data is None:
            print("No motion data to save!")
            return
            
        # Save the raw motion data
        np.save(filename, self.raw_data)
        print(f"Saved motion data to {filename}")


# Test the parser if this file is run directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dog_walk_parser.py <txt_file>")
        sys.exit(1)
    
    parser = DogWalkParser()
    parser.parse_file(sys.argv[1])
    data = parser.get_data()
    
    # Print stats about the parsed data
    print(f"Parsed motion sequence with {data['num_frames']} frames")
    print(f"Joints: {', '.join(data['joint_names'])}")
    print(f"Joint positions shape: {data['joint_positions'].shape}")
    print(f"Joint rotations shape: {data['joint_rotations'].shape}")
    print(f"Joint velocities shape: {data['joint_velocities'].shape}") 