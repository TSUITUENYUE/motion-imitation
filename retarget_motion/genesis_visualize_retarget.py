"""Visualize retargeted motion files for Go2 robot in Genesis.

This script loads and visualizes the retargeted motion files that were
generated by the retarget_motion.py script using the Go2Env from go2_env.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
import sys
import time
import glob
import argparse
import numpy as np
import json
import torch

# Fix the path to include the parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Import the Go2 config and Go2Env
import retarget_config_go2 as config
from go2_env import Go2Env

# Check if Genesis is available and initialize it
try:
    import genesis as gs
    from genesis.utils.geom import transform_quat_by_quat # Import for orientation offset
    HAS_GENESIS = True
    gs.init()  # Initialize Genesis here
    print("Genesis initialized successfully.")
    # Ensure gs.device is set, as Go2Env expects it.
    if not hasattr(gs, 'device'):
        gs.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Genesis 'gs.device' was not set by gs.init(). Defaulting to '{gs.device}'.")
    else:
        print(f"Genesis 'gs.device' is already set to '{gs.device}'.")
except ImportError:
    print("Warning: Genesis module or genesis.utils.geom.transform_quat_by_quat not found. Visualization will fall back to mock.")
    HAS_GENESIS = False
except Exception as e:
    print(f"Error during Genesis initialization or device setup: {e}")
    HAS_GENESIS = False

FRAME_DURATION = 0.01667
DEFAULT_ENV_CFG = {
    "num_actions": 12,
    "episode_length_s": 20.0,
    "base_init_pos": [0, 0, 0.32],
    "base_init_quat": [1, 0, 0, 0], # Assuming W,X,Y,Z for identity
    "joint_names": [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
    ],
    "default_joint_angles": {
        "FL_hip_joint": 0.0, "FL_thigh_joint": 0.5, "FL_calf_joint": -1.0, 
        "FR_hip_joint": 0.0, "FR_thigh_joint": 0.5, "FR_calf_joint": -1.0, 
        "RL_hip_joint": 0.0, "RL_thigh_joint": 0.5, "RL_calf_joint": -1.0, 
        "RR_hip_joint": 0.0, "RR_thigh_joint": 0.5, "RR_calf_joint": -1.0
    },
    "kp": 20.0,
    "kd": 0.5,
    "action_scale": 0.25,
    "clip_actions": 1.0,
    "resampling_time_s": 5.0,
    "termination_if_pitch_greater_than": 1.0,
    "termination_if_roll_greater_than": 1.0
}

DEFAULT_OBS_CFG = {
    "num_obs": 45,
    "obs_scales": {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05
    }
}

DEFAULT_REWARD_CFG = {
    "reward_scales": {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "action_rate": -0.01,
        "similar_to_default": -0.01,
        "base_height": -0.5
    },
    "tracking_sigma": 0.25,
    "base_height_target": 0.32
}

DEFAULT_COMMAND_CFG = {
    "num_commands": 3,
    "lin_vel_x_range": [-1.0, 1.0],
    "lin_vel_y_range": [-1.0, 1.0],
    "ang_vel_range": [-1.0, 1.0]
}

# Motion names mapped to files
MOTION_NAMES = {
    "pace": "pace.txt",
    "trot": "trot.txt",
    "trot2": "trot2.txt", 
    "canter": "canter.txt",
    "left_turn": "left turn0.txt",
    "right_turn": "right turn0.txt"
}

def load_motion(motion_file):
    """Load a retargeted motion file."""
    print(f"Loading motion file: {motion_file}")
    
    with open(motion_file, 'r') as f:
        content = f.read()
    
    # Parse the JSON-like format
    try:
        # Try direct JSON parsing first
        data = json.loads(content)
        frames = np.array(data["Frames"])
        frame_duration = data.get("FrameDuration", FRAME_DURATION)
        loop_mode = data.get("LoopMode", "Wrap")
    except json.JSONDecodeError:
        print("Warning: Could not parse file as JSON, using manual parsing.")
        # Fall back to manual parsing
        frames_start = content.find('"Frames":\n[')
        frames_end = content.rfind(']')
        
        # Extract frame duration
        duration_match = content.find('"FrameDuration":')
        if duration_match != -1:
            duration_start = duration_match + len('"FrameDuration":')
            duration_end = content.find(',', duration_start)
            frame_duration = float(content[duration_start:duration_end].strip())
        else:
            frame_duration = FRAME_DURATION
            
        # Extract loop mode
        loop_match = content.find('"LoopMode":')
        if loop_match != -1:
            loop_start = loop_match + len('"LoopMode":')
            loop_end = content.find(',', loop_start)
            loop_mode = content[loop_start:loop_end].strip(' "\'')
        else:
            loop_mode = "Wrap"
        
        frames_text = content[frames_start+11:frames_end]
        
        # Parse the frames data
        frames = []
        for line in frames_text.split('\n'):
            line = line.strip()
            if line.startswith('[') and ']' in line:
                # Remove brackets and split by comma
                values_text = line[1:line.find(']')]
                frame_values = [float(x.strip()) for x in values_text.split(',')]
                frames.append(frame_values)
    
    return np.array(frames), frame_duration, loop_mode

class GenesisVisualizer:
    """Class to handle Genesis visualization using Go2Env."""
    
    def __init__(self):
        """Initialize the Genesis visualizer."""
        if not HAS_GENESIS:
            raise ImportError("Genesis is not available or failed to initialize.")
        
        # Initialize the environment with defaults
        try:
            print("Initializing Go2Env for visualization...")
            self.env = Go2Env(
                num_envs=1,
                env_cfg=DEFAULT_ENV_CFG,
                obs_cfg=DEFAULT_OBS_CFG,
                reward_cfg=DEFAULT_REWARD_CFG,
                command_cfg=DEFAULT_COMMAND_CFG,
                show_viewer=True
            )
            print("Go2Env initialized successfully.")
        except Exception as e:
            print(f"Error initializing Go2Env: {e}")
            print("Ensure that your Genesis environment is correctly configured and all dependencies for Go2Env are met.")
            raise # Re-raise the exception to stop execution if Go2Env fails
        
        # Initialize keyboard state
        self.keys_pressed = set()
        self.last_key_update = time.time()
        
    def process_keyboard(self):
        """Process keyboard input from Genesis."""
        if not hasattr(self.env.scene, 'viewer') or self.env.scene.viewer is None:
            # print("Viewer not available for keyboard input.") # Too verbose
            return {}
        
        # Only check keyboard every 0.1 seconds to avoid spamming
        current_time = time.time()
        if current_time - self.last_key_update < 0.1:
            return {}
            
        self.last_key_update = current_time
        
        # Get keyboard events from Genesis
        keys = {}
        
        try:
            # Check for common keys using viewer methods if available
            if self.env.scene.viewer.is_key_pressed(gs.KEY_ESCAPE):
                keys['esc'] = True
            if self.env.scene.viewer.is_key_pressed(gs.KEY_SPACE):
                keys['space'] = True
            if self.env.scene.viewer.is_key_pressed(gs.KEY_N):
                keys['n'] = True
            if self.env.scene.viewer.is_key_pressed(gs.KEY_Q):
                keys['q'] = True
                
        except Exception as e:
            # Fallback or different method if is_key_pressed is not standard
            # print(f"Could not get keyboard state via is_key_pressed: {e}")
            # This part might need adjustment based on the exact Genesis keyboard API
            pass 
            
        return keys
        
    def play_motion(self, frames, frame_duration, loop=True, speed=1.0):
        """Play the motion on the robot."""
        num_frames = len(frames)
        print(f"Playing motion with {num_frames} frames")
        print(f"Frame duration: {frame_duration:.4f} seconds, Target DT: {frame_duration / speed:.4f}")
        print(f"Loop mode: {'enabled' if loop else 'disabled'}")
        print("Controls: ESC/Q = Quit, SPACE = Pause/Play, N = Next motion (when paused or finished)")
        
        frame_index = 0
        last_update_time = time.time()
        playing = True
        
        # Convert the entire frames array to torch tensor
        frames_tensor = torch.tensor(frames, dtype=torch.float32, device=gs.device)
        
        # Main visualization loop
        while True: 
            # Process potential keyboard input
            keys = self.process_keyboard()
            
            # Check for exit conditions
            if keys.get('esc', False) or keys.get('q', False):
                print("Exit key pressed.")
                return False  # Indicate to exit completely
                
            if keys.get('n', False) and (not playing or frame_index >= num_frames):
                print("'Next' key pressed.")
                return True  # Indicate to move to next motion
                
            if keys.get('space', False):
                playing = not playing
                print("Playback:", "Playing" if playing else "Paused")
                # Reset last_update_time when resuming to avoid a jump
                if playing:
                    last_update_time = time.time() 
            
            # Update the frame if playing
            current_time = time.time()
            elapsed = current_time - last_update_time
            target_dt_for_frame = frame_duration / speed
            
            if playing and elapsed >= target_dt_for_frame:
                if frame_index < num_frames:
                    # Get the current frame data
                    pose_data = frames_tensor[frame_index]
                    
                    # Apply the pose
                    self.apply_pose(pose_data)
                    
                    # Update time and index
                    frame_index += 1
                    # last_update_time = current_time # More accurate to add target_dt
                    last_update_time += target_dt_for_frame 
                    
                    # Handle loop or end of animation
                    if frame_index >= num_frames:
                        if loop:
                            frame_index = 0
                            print("Looping animation")
                        else:
                            playing = False
                            print("Animation complete. Press N for next or ESC/Q to quit.")
            
            # Step the scene
            if hasattr(self.env, 'scene') and hasattr(self.env.scene, 'step'):
                self.env.scene.step()
            else:
                print("Scene or step method not found, cannot update visualization.")
                return False # Critical error
            
            # Slight delay to be CPU-friendly, but ensure it doesn't disrupt playback timing too much
            # The main timing is handled by checking 'elapsed' against 'target_dt_for_frame'
            time.sleep(0.001) 
    
    def apply_pose(self, pose_data):
        """Apply the pose data to the robot, assuming W,X,Y,Z quaternion format."""
        # Extract data from pose
        position = pose_data[0:3].unsqueeze(0) 
        # pose_data[3] is w, pose_data[4:7] is [x, y, z] from the motion file (WXYZ source)
        w_motion_component = pose_data[3]
        xyz_motion_components = pose_data[4:7]

        # Convert motion data quaternion to XYZW format [x, y, z, w] as a base
        base_orientation_xyzw_flat = torch.cat((xyz_motion_components, w_motion_component.unsqueeze(0)), dim=0)
        base_orientation_xyzw = base_orientation_xyzw_flat.unsqueeze(0) # Shape [1, 4]

        # Convert this base XYZW orientation to WXYZ for transform_quat_by_quat
        # base_orientation_xyzw is [x,y,z,w], we need [w,x,y,z]
        w_val = base_orientation_xyzw[..., 3]
        x_val = base_orientation_xyzw[..., 0]
        y_val = base_orientation_xyzw[..., 1]
        z_val = base_orientation_xyzw[..., 2]
        base_orientation_wxyz = torch.stack((w_val, x_val, y_val, z_val), dim=-1)

        # Define the 180-degree Y-axis rotation quaternion in WXYZ format: [w,x,y,z] = [0,0,1,0]
        # This offset will be applied first in multiplication: rot_offset * base_orientation
        rot_y_180_offset_wxyz = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=gs.device).unsqueeze(0)

        # Apply rotation: result_wxyz = rot_y_180_offset_wxyz * base_orientation_wxyz
        rotated_orientation_wxyz = transform_quat_by_quat(rot_y_180_offset_wxyz, base_orientation_wxyz)

        # Convert the rotated result from WXYZ back to XYZW for set_quat
        # rotated_orientation_wxyz is [w,x,y,z], we need [x,y,z,w]
        final_x = rotated_orientation_wxyz[..., 1]
        final_y = rotated_orientation_wxyz[..., 2]
        final_z = rotated_orientation_wxyz[..., 3]
        final_w = rotated_orientation_wxyz[..., 0]
        final_orientation_for_set_quat_xyzw = torch.stack((final_x, final_y, final_z, final_w), dim=-1)
        
        joint_angles = pose_data[7:].unsqueeze(0) 
                
        # Set joint positions directly
        self.env.robot.set_dofs_position(
            position=joint_angles,
            dofs_idx_local=self.env.motors_dof_idx,
            zero_velocity=True
        )
        
        # Set the base position and FINAL orientation
        self.env.robot.set_pos(position)
        self.env.robot.set_quat(final_orientation_for_set_quat_xyzw)

class MockVisualizer:
    """Mock visualizer for when Genesis is not available."""
    
    def __init__(self):
        """Initialize the mock visualizer."""
        print("Using mock visualizer (Genesis not available or Go2Env failed to initialize)")
        
    def play_motion(self, frames, frame_duration, loop=True, speed=1.0):
        """Simulate playing the motion."""
        num_frames = len(frames)
        print(f"Simulating playback of {num_frames} frames")
        print(f"Duration: {frame_duration * num_frames / speed:.2f} seconds")
        print(f"Loop mode: {'enabled' if loop else 'disabled'}")
        
        # Simulate playback
        if loop:
            print("Looping animation - press Ctrl+C to stop")
            try:
                loop_count = 0
                while True:
                    time.sleep(frame_duration * num_frames / speed)
                    loop_count += 1
                    print(f"Completed loop {loop_count}")
            except KeyboardInterrupt:
                print("Animation stopped")
        else:
            print(f"Animation will play for {frame_duration * num_frames / speed:.2f} seconds")
            time.sleep(frame_duration * num_frames / speed)
            print("Animation complete")
            
        return True  # Continue to next motion
    
    def apply_pose(self, pose_data):
        """Mock applying pose data."""
        pass

def find_motion_files(motion_dir, motion_names=None):
    """Find motion files based on directory and optional motion names."""
    all_motion_files = {}
    
    # Make sure we have the correct absolute path for retarget_result
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.abspath(os.path.join(parentdir, motion_dir))
    
    # Debug: show where we're looking
    print(f"Looking for motion files in: {motion_dir}")
    
    # Try to find motion files in the provided directory
    if motion_names:
        # Handle specific motion requests
        motion_files = []
        for motion_name in motion_names:
            # Check if it's a key in the MOTION_NAMES dictionary
            if motion_name in MOTION_NAMES:
                filename = MOTION_NAMES[motion_name]
                filepath = os.path.join(motion_dir, filename)
                print(f"Checking for motion file: {filepath}")
                if os.path.exists(filepath):
                    motion_files.append(filepath)
                    all_motion_files[motion_name] = filepath
                else:
                    print(f"Motion '{motion_name}' file not found: {filepath}")
            else:
                print(f"Unknown motion name: {motion_name}")
    else:
        # Get all txt files in the directory
        motion_files = sorted(glob.glob(os.path.join(motion_dir, '*.txt')))
        
        # If no files found, try the retarget_result directory
        if not motion_files:
            alt_dir = os.path.join(parentdir, "retarget_result")
            print(f"No files found in {motion_dir}, trying: {alt_dir}")
            motion_files = sorted(glob.glob(os.path.join(alt_dir, '*.txt')))
            motion_dir = alt_dir
    
        # Map filenames to motion names
        for filepath in motion_files:
            filename = os.path.basename(filepath)
            # Get motion name by removing .txt extension
            motion_name = os.path.splitext(filename)[0]
            all_motion_files[motion_name] = filepath
    
    if not motion_files:
        print(f"No motion files found in {motion_dir}")
        return []
        
    print(f"Found {len(motion_files)} motion files:")
    for i, f in enumerate(motion_files):
        print(f"  {i+1}. {os.path.basename(f)}")
    
    return motion_files

def visualize_motions(motion_files, loop=True, playback_speed=1.0):
    """Visualize the retargeted motions."""
    # Select the appropriate visualizer
    visualizer = None # Initialize to None
    if HAS_GENESIS:
        try:
            visualizer = GenesisVisualizer()
        except Exception as e:
            print(f"Failed to initialize Genesis visualizer: {e}")
            print("Falling back to mock visualizer.")
            # visualizer will remain None, so MockVisualizer will be used below
    
    if visualizer is None: # If HAS_GENESIS was false or GenesisVisualizer failed
        visualizer = MockVisualizer()
    
    # Play each motion file
    for motion_file in motion_files:
        print(f"\nVisualizing: {os.path.basename(motion_file)}")
        
        # Load the motion data
        frames, frame_duration, loop_mode = load_motion(motion_file)
        
        # Respect the file's loop mode unless overridden
        file_should_loop = (loop_mode.lower() == "wrap")
        should_loop = loop or file_should_loop
        
        # Play the motion
        continue_to_next = visualizer.play_motion(
            frames, 
            frame_duration, 
            loop=should_loop, 
            speed=playback_speed
        )
        
        if not continue_to_next:
            break

def main():
    parser = argparse.ArgumentParser(description='Visualize retargeted motion files in Genesis')
    parser.add_argument('--motion_dir', type=str, default='retarget_result', 
                        help='Directory containing retargeted motion files')
    parser.add_argument('--motion_file', type=str, default=None,
                        help='Specific motion file to visualize (optional)')
    parser.add_argument('--motion', type=str, nargs='+', choices=list(MOTION_NAMES.keys()),
                        help='Specific motion(s) to visualize by name (e.g., trot, pace)')
    parser.add_argument('--loop', action='store_true', 
                        help='Loop the animation (overrides file setting)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier')
    
    args = parser.parse_args()
    
    # Give motion argument priority over motion_file
    if args.motion:
        motion_files = find_motion_files(args.motion_dir, args.motion)
    elif args.motion_file:
        # Handle individual file specification
        if os.path.isabs(args.motion_file):
            motion_files = [args.motion_file]
        else:
            # Try various locations
            possible_paths = [
                args.motion_file,
                os.path.join(args.motion_dir, args.motion_file),
                os.path.join(parentdir, "retarget_result", args.motion_file)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    motion_files = [path]
                    break
            else:
                print(f"Motion file {args.motion_file} not found")
                return
    else:
        # Get all motion files
        motion_files = find_motion_files(args.motion_dir)
    
    if not motion_files:
        return
    
    visualize_motions(motion_files, loop=args.loop, playback_speed=args.speed)

if __name__ == "__main__":
    main() 