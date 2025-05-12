import torch
import numpy as np
import genesis as gs

class TxtMotionParser:
    """Parser for txt motion capture files as found in dog_walk01_pose.txt format."""
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.motion_data = None
        self.num_frames = 0
        self.frame_time = 0.01  # Default 100 fps, can be overridden
        
        # Representation data
        self.joint_positions = None
        self.joint_rotations = None
        
        # Number of parameters per frame 
        self.params_per_frame = 0
        
    def parse_file(self, file_path):
        """Parse a txt motion file and extract motion data."""
        print(f"Parsing TXT motion file: {file_path}")
        
        # Read all lines and parse data
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Filter out empty lines and process data
        data_rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Split by commas and convert to float
            values = [float(val.strip()) for val in line.split(',')]
            data_rows.append(values)
        
        # Convert to numpy array
        self.motion_data = np.array(data_rows)
        self.num_frames = len(data_rows)
        self.params_per_frame = self.motion_data.shape[1] if self.num_frames > 0 else 0
        
        print(f"Parsed {self.num_frames} frames with {self.params_per_frame} parameters per frame")
        
        # Process the data into joint rotations and positions
        self._process_motion_data()
        
        return self
    
    def _process_motion_data(self):
        """
        Process the raw motion data into joint rotations and positions.
        
        The TXT format appears to contain a flat representation of parameters,
        which we'll interpret as:
        - First 3 values: root position (x, y, z)
        - Remaining values: quaternion rotations (w, x, y, z) for each joint
        """
        if self.motion_data is None or self.num_frames == 0:
            return
            
        # Determine how many joints we have based on parameters
        # Assuming first 3 values are position, and rest are quaternions (4 values per joint)
        num_quaternions = (self.params_per_frame - 3) // 4
        num_joints = num_quaternions + 1  # +1 for the root joint
        
        # Initialize tensors for joint data
        self.joint_positions = torch.zeros((self.num_frames, num_joints, 3), device=self.device)
        self.joint_rotations = torch.zeros((self.num_frames, num_joints, 4), device=self.device)
        
        # Set default quaternion (identity rotation) for all joints
        self.joint_rotations[:, :, 0] = 1.0  # w component
        
        # Process each frame
        for frame_idx in range(self.num_frames):
            frame_data = self.motion_data[frame_idx]
            
            # Extract root position (first 3 values)
            root_pos = frame_data[0:3]
            self.joint_positions[frame_idx, 0] = torch.tensor(root_pos, device=self.device)
            
            # Extract quaternions for each joint
            for joint_idx in range(num_joints):
                # For the root joint (idx 0) and other joints
                quat_start_idx = 3 + (joint_idx * 4)
                
                # Check if we have enough data for this joint
                if quat_start_idx + 4 <= len(frame_data):
                    # Extract quaternion (w, x, y, z)
                    quat = frame_data[quat_start_idx:quat_start_idx+4]
                    self.joint_rotations[frame_idx, joint_idx] = torch.tensor(quat, device=self.device)
        
        # Calculate positions for non-root joints based on a simplified skeletal structure
        # This is an approximation since we don't have explicit skeleton data
        self._calculate_joint_positions()
        
    def _calculate_joint_positions(self):
        """
        Calculate positions for non-root joints based on a simple dog skeleton.
        This is an approximation and would need to be adjusted based on the actual joint hierarchy.
        """
        # Define a simple dog skeleton with joint offsets (approximate values)
        # These values would need to be tuned based on the actual dog model
        joint_offsets = [
            [0.0, 0.0, 0.0],     # 0: Root (already positioned)
            [0.2, 0.1, 0.0],     # 1: Front right shoulder
            [0.0, 0.0, -0.15],   # 2: Front right knee
            [0.0, 0.0, -0.15],   # 3: Front right foot
            [0.2, -0.1, 0.0],    # 4: Front left shoulder
            [0.0, 0.0, -0.15],   # 5: Front left knee
            [0.0, 0.0, -0.15],   # 6: Front left foot
            [-0.2, 0.1, 0.0],    # 7: Back right hip
            [0.0, 0.0, -0.15],   # 8: Back right knee
            [0.0, 0.0, -0.15],   # 9: Back right foot
            [-0.2, -0.1, 0.0],   # 10: Back left hip
            [0.0, 0.0, -0.15],   # 11: Back left knee
            [0.0, 0.0, -0.15],   # 12: Back left foot
        ]
        
        # Define parent-child relationships
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
        
        # Ensure we don't exceed the number of joints we have data for
        num_joints = min(len(joint_offsets), self.joint_rotations.shape[1])
        
        # For each frame
        for frame_idx in range(self.num_frames):
            # For each joint (except root which is already positioned)
            for joint_idx in range(1, num_joints):
                parent_idx = parent_indices[joint_idx]
                
                # Skip if parent index is invalid
                if parent_idx < 0 or parent_idx >= num_joints:
                    continue
                
                # Get parent position and rotation
                parent_pos = self.joint_positions[frame_idx, parent_idx]
                parent_rot = self.joint_rotations[frame_idx, parent_idx]
                
                # Get offset from parent to this joint
                offset = torch.tensor(joint_offsets[joint_idx], device=self.device)
                
                # Rotate offset by parent's rotation
                rotated_offset = self._rotate_vector_by_quaternion(offset, parent_rot)
                
                # Calculate global position
                global_pos = parent_pos + rotated_offset
                self.joint_positions[frame_idx, joint_idx] = global_pos
    
    def _rotate_vector_by_quaternion(self, v, q):
        """Rotate vector v by quaternion q."""
        # Convert v to a quaternion with w=0 (pure vector)
        qv = torch.tensor([0, v[0], v[1], v[2]], device=self.device)
        
        # Compute q * qv * q^-1
        # First compute q^-1
        q_conj = torch.tensor([q[0], -q[1], -q[2], -q[3]], device=self.device)
        q_norm = torch.sum(q * q)
        q_inv = q_conj / (q_norm + 1e-8)  # Avoid division by zero
        
        # Compute the rotation
        qv_rotated = self._multiply_quaternions(q, qv)
        qv_rotated = self._multiply_quaternions(qv_rotated, q_inv)
        
        # Extract the vector part
        return qv_rotated[1:]
    
    def _multiply_quaternions(self, q1, q2):
        """Multiply two quaternions."""
        # q = [w, x, y, z]
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.tensor([w, x, y, z], device=self.device)
    
    def get_data(self):
        """Return the parsed motion data for use in retargeting."""
        return {
            'joint_positions': self.joint_positions,
            'joint_rotations': self.joint_rotations,
            'num_frames': self.num_frames,
            'frame_time': self.frame_time,
            'raw_data': self.motion_data
        }

    def save_as_npy(self, filename):
        """Save motion data to a NPY file."""
        if self.motion_data is None:
            print("No motion data to save!")
            return
            
        # Save the raw motion data
        np.save(filename, self.motion_data)
        print(f"Saved motion data to {filename}")


# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python txt_motion_parser.py <txt_file>")
        sys.exit(1)
    
    parser = TxtMotionParser()
    parser.parse_file(sys.argv[1])
    data = parser.get_data()
    
    print(f"Parsed {data['num_frames']} frames")
    print(f"Joint positions shape: {data['joint_positions'].shape}")
    print(f"Joint rotations shape: {data['joint_rotations'].shape}") 