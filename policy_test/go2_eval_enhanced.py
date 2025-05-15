import argparse
import os
import pickle
import torch
import glob
import re
from rsl_rl.modules import ActorCritic
import genesis as gs
import time

# Import from parent directory
import sys
sys.path.append("..")
from go2_env import Go2Env # Import Go2Env unconditionally

# Try to import EnhancedMotionImitationEnv from the enhanced training script
MOTION_IMITATION_ENV_IMPORTED = False
try:
    # Import from train_experiment.py instead of train.py
    from rl_train.train_experiment import EnhancedMotionImitationEnv, load_reference_motion
    MOTION_IMITATION_ENV_IMPORTED = True
    print("Successfully imported EnhancedMotionImitationEnv from rl_train.train_experiment")
except ImportError as e:
    print(f"Warning: Could not import EnhancedMotionImitationEnv from rl_train.train_experiment: {e}")
    print("Falling back to default Go2Env. This might cause issues if the model was trained with EnhancedMotionImitationEnv.")

def get_cfgs():
    """Define standard configurations for Go2 robot."""
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.75,
            "FR_thigh_joint": 0.75,
            "RL_thigh_joint": 0.75,
            "RR_thigh_joint": 0.75,
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
        "kp": 50.0, # Moderate stiffness 
        "kd": 0.8,  # Moderate damping
        "termination_if_roll_greater_than": 25, 
        "termination_if_pitch_greater_than": 25,
        "base_init_pos": [0.0, 0.0, 0.28],  # Start low to encourage ground contact
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0, 
        "action_scale": 0.25, # Reduced for finer control
        "simulate_action_latency": False,
        "clip_actions": 0.75, # Tighter clipping
        "ground_friction": 1.0,
        "ground_restitution": 0.0, # No bounce from ground
        "joint_friction": 0.03,
        "foot_friction": 1.0,
        "enable_stabilizer": True,
        # Enhanced parameters
        "use_time_dependent_rewards": True,
        "imitation_decay_rate": 5.0,
        "robustness_rise_rate": 8.0,
    }
    
    obs_cfg = {
        "num_obs": 45, 
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "tracking_sigma": 0.25, 
        "base_height_target": 0.28, # Target height for walking
        "reward_scales": {
            "tracking_lin_vel": 0.4, 
            "tracking_ang_vel": 0.2, 
            "lin_vel_z": -2.0, # Strong penalty for vertical base velocity (anti-bounce)
            "action_rate": -0.015, 
            "similar_to_default": -0.03, 
            "joint_pose_matching": 3.0,
            "ground_contact": 3.5,
            "forward_motion": 2.0,
            "chassis_height": 1.0,
            "velocity_profile": 1.0,
            "leg_symmetry": 0.75,
            "end_effector_matching": 2.0,
            "gait_continuation": 1.5,
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5], # Target moderate forward speed
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0], 
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def create_policy(num_obs, num_actions, policy_cfg_dict):
    """Create a policy network based on a policy configuration dictionary."""
    print(f"Creating actor-critic network with obs: {num_obs}, actions: {num_actions}")
    print(f"Policy Cfg: {policy_cfg_dict}")
    
    policy = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=policy_cfg_dict.get("actor_hidden_dims", [512, 256, 128]),
        critic_hidden_dims=policy_cfg_dict.get("critic_hidden_dims", [512, 256, 128]),
        activation=policy_cfg_dict.get("activation", "elu"),
        init_noise_std=policy_cfg_dict.get("init_noise_std", 1.0)
    )
    return policy

def find_latest_checkpoint(motion_type=None):
    """Find the checkpoint with the highest iteration number in the logs directory."""
    # Get the root directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use the absolute path to the motion_imitation directory instead of relative paths
    # This ensures the logs directory is found correctly regardless of where the script is run from
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    logs_dir = os.path.join(root_dir, "rl_train", "logs")
    
    # If motion_type is specified, look in that specific directory
    if motion_type:
        # Check in both enhanced and original log directories
        log_dirs_to_check = [
            os.path.join(logs_dir, f"go2-enhanced-{motion_type}"),
            os.path.join(logs_dir, f"go2-imitate-{motion_type}")
        ]
        
        found_dir = None
        for dir_path in log_dirs_to_check:
            if os.path.exists(dir_path):
                found_dir = dir_path
                print(f"Found log directory: {found_dir}")
                break
                
        if found_dir:
            logs_dir = found_dir
        else:
            print(f"Log directory for motion type '{motion_type}' not found in any expected location:")
            for dir_path in log_dirs_to_check:
                print(f"  - Checked: {dir_path}")
            return None
    
    # If we get here without a specific directory (no motion_type), find all suitable log directories
    if not motion_type:
        # Find all checkpoint directories
        log_dirs = []
        try:
            for item in os.listdir(logs_dir):
                full_path = os.path.join(logs_dir, item)
                if os.path.isdir(full_path) and (item.startswith("go2-enhanced-") or item.startswith("go2-imitate-")):
                    log_dirs.append(full_path)
        except FileNotFoundError:
            print(f"Logs directory not found: {logs_dir}")
            return None
            
        if not log_dirs:
            print(f"No log directories found in {logs_dir}")
            return None
        
        # Use the most recent directory
        log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        logs_dir = log_dirs[0]
        print(f"Using most recent log directory: {os.path.basename(logs_dir)}")
    
    # Find all model files in the directory
    model_files = glob.glob(os.path.join(logs_dir, "model_*.pt"))
    
    if not model_files:
        print(f"No model files found in {logs_dir}")
        return None
    
    # Extract iteration numbers from filenames
    iter_numbers = []
    for file_path in model_files:
        # Extract number from filename (model_X.pt)
        match = re.search(r'model_(\d+)\.pt', os.path.basename(file_path))
        if match:
            iter_numbers.append((int(match.group(1)), file_path))
    
    # Sort by iteration number (highest first)
    iter_numbers.sort(reverse=True)
    
    if not iter_numbers:
        print(f"No valid model files found in {logs_dir}")
        return None
    
    # Return the file with the highest iteration number
    highest_iter, latest_file = iter_numbers[0]
    print(f"Found latest checkpoint: {os.path.basename(latest_file)} (iteration {highest_iter})")
    
    return latest_file

# Add function to handle unpacking joint velocity data
def load_reference_motion_wrapper(motion_file):
    """Wrapper to handle both old and new reference motion formats."""
    try:
        # Try loading with the updated function that returns joint_angles, joint_velocities
        result = load_reference_motion(motion_file)
        
        # Check if result is a tuple (new format) or direct tensor (old format)
        if isinstance(result, tuple):
            joint_angles, joint_velocities = result
            # Important: Return both joint angles and velocities for proper motion
            return joint_angles, joint_velocities
        else:
            # Legacy format - just return the joint angles
            return result, None
    except Exception as e:
        print(f"Error loading motion file {motion_file}: {e}")
        try:
            # Try direct numpy loading if the function fails
            import numpy as np
            data = np.load(motion_file, allow_pickle=True)
            
            # Check if it's a dictionary format (new) or direct array (old)
            if isinstance(data, np.ndarray) and data.dtype == np.dtype('O') and isinstance(data.item(), dict):
                # New format with dictionary
                data_dict = data.item()
                joint_angles = torch.tensor(data_dict['joint_angles'], device=gs.device, dtype=torch.float32)
                
                # Also load joint velocities if available
                if 'joint_velocities' in data_dict:
                    joint_velocities = torch.tensor(data_dict['joint_velocities'], device=gs.device, dtype=torch.float32)
                    return joint_angles, joint_velocities
                else:
                    return joint_angles, None
            else:
                # Old format with direct joint angles
                joint_angles = torch.tensor(data, device=gs.device, dtype=torch.float32)
                return joint_angles, None
        except Exception as nested_e:
            print(f"Secondary error when trying manual loading: {nested_e}")
            return None, None

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Go2 policy with enhanced motion imitation",
        formatter_class=argparse.RawTextHelpFormatter  # Allow newlines in help text
    )
    
    # Group checkpoint selection options
    checkpoint_group = parser.add_argument_group('Model Checkpoint Options (choose one)')
    
    checkpoint_group.add_argument("-m", "--motion", type=str, default=None,
                    help="Motion type (e.g., 'canter', 'trot', 'pace') for auto-finding latest checkpoint")
    
    checkpoint_group.add_argument("-e", "--experiment", type=str, default=None,
                    help="Explicit experiment name to use (e.g., 'go2-enhanced-canter_new')")
    
    checkpoint_group.add_argument("--ckpt", type=int, default=None,
                    help="Specific checkpoint number to load (e.g., 999 for model_999.pt)\n"
                         "Must be used with --experiment or --run_dir")
    
    checkpoint_group.add_argument("--run_dir", type=str, default=None,
                    help="Full path to specific training run directory\n" 
                         "(e.g., logs/go2-enhanced-canter_new)")
    
    checkpoint_group.add_argument("--model", type=str, default=None,
                    help="Full path to specific model file\n"
                         "(e.g., logs/go2-enhanced-canter_new/model_999.pt)")
    
    # Other options
    parser.add_argument("-v", "--viz", action="store_true", default=True,
                    help="Enable visualization (default: True)")
    
    parser.add_argument("--duration", type=int, default=60,
                    help="Duration in seconds to run the evaluation (default: 60)")
    
    parser.add_argument("--cfg", type=str, default=None,
                    help="Path to a specific cfg.pkl file to use")
    
    parser.add_argument("--compat", action="store_true", default=True,
                    help="Enable compatibility mode for models trained with original environment (default: True)")
                    
    parser.add_argument("--show-rewards", action="store_true", default=False,
                    help="Show detailed reward breakdown during evaluation (default: False)")
    
    parser.add_argument("--reward-interval", type=int, default=30,
                    help="Display reward breakdown every N frames (default: 30)")
    
    args = parser.parse_args()
    
    # Print argument usage example if no arguments provided
    if len(sys.argv) == 1:
        print("\nUsage examples:")
        print("  # Load the latest model for canter motion:")
        print("  python ../policy_test/go2_eval_enhanced.py -m canter")
        print("\n  # Load a specific checkpoint from an experiment with reward visualization:")
        print("  python ../policy_test/go2_eval_enhanced.py -e go2-enhanced-canter_new --ckpt 499 --show-rewards")
        print("\n  # Load from a specific run directory:")
        print("  python ../policy_test/go2_eval_enhanced.py --run_dir logs/go2-enhanced-canter_new")
        print("\n  # Load a specific model file and show rewards less frequently:")
        print("  python ../policy_test/go2_eval_enhanced.py --model logs/go2-enhanced-canter_new/model_499.pt --show-rewards --reward-interval 60")
        print("\n  # Run for a specific duration:")
        print("  python ../policy_test/go2_eval_enhanced.py -m canter --duration 120")
        print("\n")

    # Initialize Genesis
    gs.init()

    # Resolve the model path using the various options
    model_path = args.model
    run_dir_path = args.run_dir
    cfg_path = args.cfg  # Custom path to cfg file

    # If experiment name is given, construct run_dir path
    if args.experiment:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exp_logs_dir = os.path.join(script_dir, "..", "rl_train", "logs", args.experiment)
        
        if os.path.exists(exp_logs_dir):
            run_dir_path = exp_logs_dir
            print(f"Using experiment log directory: {run_dir_path}")
            
            # If checkpoint also specified, construct model path
            if args.ckpt is not None:
                model_path = os.path.join(run_dir_path, f"model_{args.ckpt}.pt")
                if not os.path.exists(model_path):
                    print(f"Warning: Specified checkpoint {model_path} not found.")
                    model_path = None  # Reset to trigger auto-find
        else:
            print(f"Warning: Experiment log directory {exp_logs_dir} not found")
    
    # If motion type is given, find the latest checkpoint
    if args.motion and not model_path:
        found_path = find_latest_checkpoint(args.motion)
        if found_path:
            model_path = found_path
            # Infer run_dir_path from model_path if not already set
            if not run_dir_path:
                run_dir_path = os.path.dirname(model_path)
    
    # If model path is given but run_dir_path is not, infer run_dir_path from model_path
    if model_path and not run_dir_path:
        run_dir_path = os.path.dirname(os.path.abspath(model_path))
        print(f"Inferred run_dir from model path: {run_dir_path}")
    
    # If run_dir_path is given but model_path is not, find latest model in run_dir
    if run_dir_path and not model_path:
        if not os.path.isdir(run_dir_path):
            print(f"Warning: run_dir '{run_dir_path}' is not a valid directory")
            return
            
        model_files = glob.glob(os.path.join(run_dir_path, "model_*.pt"))
        if not model_files:
            print(f"No model_*.pt files found in {run_dir_path}")
            return
        iter_numbers = []
        for file_path_iter in model_files: # Renamed to avoid conflict with outer scope
            match = re.search(r'model_(\d+)\.pt', os.path.basename(file_path_iter))
            if match:
                iter_numbers.append((int(match.group(1)), file_path_iter))
        if not iter_numbers:
            print(f"No valid model files (e.g. model_100.pt) found in {run_dir_path}")
            return
        iter_numbers.sort(reverse=True)
        model_path = iter_numbers[0][1]
        print(f"Found latest model in run_dir: {os.path.basename(model_path)}")
    elif not model_path and not run_dir_path:
        print("Error: You must specify --model (path or name) and/or --run_dir.")
        return

    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model path is invalid or model does not exist: {model_path}")
        return
    
    # Determine cfg_path based on the final run_dir_path
    if not cfg_path and run_dir_path and os.path.isdir(run_dir_path):
        cfg_path_candidate = os.path.join(run_dir_path, "cfgs.pkl")
        if os.path.exists(cfg_path_candidate):
            cfg_path = cfg_path_candidate
        else:
            print(f"Warning: cfgs.pkl not found in the determined run_dir: {run_dir_path}")
    else:
        print(f"Warning: run_dir '{run_dir_path}' is not a valid directory. Cannot load cfgs.pkl.")

    print(f"Loading model from: {model_path}")

    # --- Load configurations from cfgs.pkl --- 
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_loaded = None, None, None, None, None
    loaded_policy_cfg_dict = {} # For ActorCritic parameters

    if cfg_path and os.path.exists(cfg_path):
        print(f"Loading configurations from: {cfg_path}")
        with open(cfg_path, 'rb') as f:
            loaded_configs = pickle.load(f)
            # The structure of cfgs.pkl is a list: [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg]
            if len(loaded_configs) == 5:
                env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_loaded = loaded_configs
                print("Successfully loaded all 5 configurations from cfgs.pkl")
                if train_cfg_loaded and "policy" in train_cfg_loaded:
                    loaded_policy_cfg_dict = train_cfg_loaded["policy"]
            else:
                print(f"Warning: cfgs.pkl at {cfg_path} does not contain the expected 5 configurations. It has {len(loaded_configs)} items.")
                # Fallback to default if any essential cfg is missing
                if env_cfg is None: env_cfg, _, _, _ = get_cfgs() # type: ignore
                if obs_cfg is None: _, obs_cfg, _, _ = get_cfgs() # type: ignore
                if command_cfg is None: _, _, _, command_cfg = get_cfgs() # type: ignore
                # reward_cfg is less critical for eval physics but good to have for consistency if EnhancedMotionImitationEnv uses it
                if reward_cfg is None: _, _, reward_cfg, _ = get_cfgs() # type: ignore
    else:
        print("Warning: cfgs.pkl not found. Using default configurations from go2_eval_enhanced.py. This may lead to unexpected behavior if the model was trained with different settings.")
        # Load default configurations if cfgs.pkl is not found
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        # Use default policy arch if no train_cfg loaded
        loaded_policy_cfg_dict = {"actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128], "activation": "elu", "init_noise_std": 1.0}

    # Check for compatibility issues with older models
    is_original_model = False
    if run_dir_path:
        is_original_model = os.path.basename(run_dir_path).startswith("go2-imitate-") and not os.path.basename(run_dir_path).startswith("go2-enhanced-")
    
    # Add detection for paper-rewards models
    is_paper_rewards_model = False
    if run_dir_path:
        is_paper_rewards_model = os.path.basename(run_dir_path).startswith("go2-paper-rewards-")
    
    # Add missing reward scales for backward compatibility
    if args.compat:
        print("Compatibility mode enabled. Handling reward scales for model compatibility.")
        
        # Default reward scales needed by the enhanced environment
        default_reward_scales = {
            "joint_pose_matching": 3.0,
            "velocity_profile": 1.0,
            "leg_symmetry": 0.75,
            "end_effector_matching": 2.0,
            "forward_motion": 2.0,
            "chassis_height": 1.0,
            "ground_contact": 3.5,
            "gait_continuation": 1.5,
        }
        
        # For paper-rewards models, we don't need certain rewards
        if is_paper_rewards_model:
            print("Detected paper-rewards model, removing unnecessary reward scales")
            if "chassis_height" in default_reward_scales:
                del default_reward_scales["chassis_height"]
            if "leg_symmetry" in default_reward_scales:
                del default_reward_scales["leg_symmetry"]
            if "gait_continuation" in default_reward_scales:
                del default_reward_scales["gait_continuation"]
        
        if not reward_cfg:
            # Create reward_cfg if it doesn't exist
            _, _, reward_cfg, _ = get_cfgs()
        elif "reward_scales" not in reward_cfg:
            # Create reward_scales if it doesn't exist
            reward_cfg["reward_scales"] = {}
        
        # Add any missing reward scales
        for key, value in default_reward_scales.items():
            if key not in reward_cfg["reward_scales"]:
                print(f"Adding missing reward scale: {key} = {value}")
                reward_cfg["reward_scales"][key] = value
        
        # Set compatibility flags if needed
        if "use_time_dependent_rewards" not in env_cfg:
            env_cfg["use_time_dependent_rewards"] = False
            print("Setting use_time_dependent_rewards to False for compatibility")
    
    # --- Environment Creation --- 
    # Determine which environment class to use
    EnvClassToUse = Go2Env # Default
    reference_motion_for_eval = None
    motion_filename_for_eval = None

    # Determine if EnhancedMotionImitationEnv should be used
    use_motion_imitation_env = False
    if MOTION_IMITATION_ENV_IMPORTED:
        # Priority 1: motion_filename explicitly stored in loaded env_cfg
        if env_cfg and env_cfg.get("motion_filename"):
            motion_filename_for_eval = env_cfg["motion_filename"]
            if os.path.exists(motion_filename_for_eval):
                print(f"Using motion_filename from env_cfg: {motion_filename_for_eval}")
                use_motion_imitation_env = True
            else:
                print(f"Warning: motion_filename '{motion_filename_for_eval}' from env_cfg not found. Will try to infer.")
                motion_filename_for_eval = None # Reset to trigger inference

        # Priority 2: Infer from experiment name if not found in env_cfg or if path was invalid
        if not use_motion_imitation_env and train_cfg_loaded and train_cfg_loaded.get("runner", {}).get("experiment_name", ""):
            exp_name = train_cfg_loaded.get("runner", {}).get("experiment_name", "")
            # Try to extract the base motion name from the experiment name
            match = re.search(r"go2-(enhanced|imitate)-([a-zA-Z0-9_]+)(?:-[a-zA-Z0-9_]+)*", exp_name)
            if match:
                inferred_motion_name = match.group(2) # This should be the core motion name like "canter"
                print(f"Inferred base motion name from exp_name '{exp_name}': '{inferred_motion_name}'")
                if model_path:
                    # Construct path to standard data directory relative to this script's location
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    # Assumes structure: ./policy_test/go2_eval_enhanced.py, ../rl_train/data/motion.npy
                    data_dir_candidate = os.path.normpath(os.path.join(script_dir, "..", "rl_train", "data"))
                    motion_file_cand = os.path.join(data_dir_candidate, inferred_motion_name + ".npy")
                    if os.path.exists(motion_file_cand):
                        motion_filename_for_eval = motion_file_cand
                        print(f"Found inferred motion file for eval: {motion_filename_for_eval}")
                        use_motion_imitation_env = True
                    else:
                        print(f"Could not find inferred motion file '{inferred_motion_name}.npy' at: {data_dir_candidate}")
                else:
                    print("Cannot infer motion file path from exp_name without model_path being resolved to find relative data dir.")
            else:
                print(f"Could not parse motion name from experiment name: {exp_name}")
        
        # Fallback: if env_cfg indicates it IS a motion_imitation_env but we couldn't get filename
        if not use_motion_imitation_env and env_cfg.get('_is_motion_imitation_env', False):
             print("env_cfg suggests EnhancedMotionImitationEnv, but motion_filename could not be determined. Proceeding with EnhancedMotionImitationEnv without reference motion.")
             use_motion_imitation_env = True # Will use EnhancedMotionImitationEnv but reference_motion might be None

    if use_motion_imitation_env:
        print("Using EnhancedMotionImitationEnv for evaluation.")
        EnvClassToUse = EnhancedMotionImitationEnv
        if motion_filename_for_eval:
            if MOTION_IMITATION_ENV_IMPORTED: # Ensure class was actually imported
                reference_motion_for_eval, reference_velocities_for_eval = load_reference_motion_wrapper(motion_filename_for_eval)
                if reference_motion_for_eval is None:
                    print(f"Warning: load_reference_motion returned None for '{motion_filename_for_eval}'.")
            else:
                 print("EnhancedMotionImitationEnv class was not imported, cannot load reference motion.")
        else:
            print("Warning: No motion_filename available for EnhancedMotionImitationEnv. Motion imitation rewards will be zero or behave unexpectedly.")
    else:
        print("Using Go2Env for evaluation.")
        EnvClassToUse = Go2Env

    print(f"Using Environment Class: {EnvClassToUse.__name__}")
    print("Using env_cfg for evaluation:", env_cfg) # Log the actual env_cfg being used

    # Create environment instance
    if EnvClassToUse == EnhancedMotionImitationEnv:
        # Define a custom class that overrides __init__ to patch reward functions
        class CompatibleEnhancedEnv(EnhancedMotionImitationEnv):
            def __init__(self, *args, **kwargs):
                # Silently create stubs for all possible reward methods to avoid any errors
                reward_methods = [
                    "_reward_chassis_height", "_reward_leg_symmetry", "_reward_gait_continuation", 
                    "_reward_forward_motion", "_reward_velocity_matching", "_reward_joint_velocity_matching", 
                    "_reward_ground_contact", "_reward_velocity_profile", "_reward_forward_progression"
                ]
                
                # Add silent stubs for all possible reward methods
                for method_name in reward_methods:
                    if not hasattr(CompatibleEnhancedEnv, method_name):
                        # Define a silent stub method
                        def make_stub_method(name):
                            def stub_method(self):
                                return torch.zeros(self.num_envs, device=self.device)
                            return stub_method
                        
                        # Bind silently - no prints
                        setattr(CompatibleEnhancedEnv, method_name, make_stub_method(method_name))
                
                # Call parent constructor with all args
                try:
                    super().__init__(*args, **kwargs)
                except Exception as e:
                    # If error, try to add the missing reward method and try again
                    if "_reward_" in str(e):
                        # Extract the missing reward method name
                        import re
                        match = re.search(r"'([^']*_reward_[^']*)'", str(e))
                        method_name = match.group(1) if match else "_reward_unknown"
                        
                        # Add the stub silently
                        def stub_method(self):
                            return torch.zeros(self.num_envs, device=self.device)
                        setattr(CompatibleEnhancedEnv, method_name, stub_method)
                        
                        # Try again
                        super().__init__(*args, **kwargs)
                    else:
                        # Re-raise other errors
                        raise
                        
                # Override reset and step to improve motion - no behavior change, just making sure motion flows
                self._original_reset = self.reset
                self._original_step = self.step
                
                # Silence any future reward stub creation outputs from environment
                if hasattr(self, 'reward_functions'):
                    # Find all reward methods that would have verbose prints during execution
                    for reward_name in list(self.reward_cfg.get("reward_scales", {}).keys()):
                        method_name = f"_reward_{reward_name}"
                        if not hasattr(self, method_name):
                            # Define a silent stub
                            def make_stub_method(name):
                                def stub_method(self):
                                    return torch.zeros(self.num_envs, device=self.device)
                                return stub_method
                            
                            # Replace silently - no print
                            setattr(self.__class__, method_name, make_stub_method(method_name))
                            self.reward_functions[reward_name] = getattr(self, method_name)
                
                def reset_wrapper(self, *args, **kwargs):
                    obs, extras = self._original_reset(*args, **kwargs)
                    # Ensure the robot is in a good state to start moving
                    if hasattr(self, 'dof_pos') and self.dof_pos is not None and hasattr(self, 'reference_motion') and self.reference_motion is not None:
                        # Set to first frame position
                        self.dof_pos[:] = self.reference_motion[0].unsqueeze(0)
                        if hasattr(self, 'dof_targets'):
                            self.dof_targets[:] = self.dof_pos
                    return obs, extras
                
                def step_wrapper(self, *args, **kwargs):
                    # Just call parent step
                    return self._original_step(*args, **kwargs)
                
                # Replace methods
                self.reset = lambda *args, **kwargs: reset_wrapper(self, *args, **kwargs)
                self.step = lambda *args, **kwargs: step_wrapper(self, *args, **kwargs)
        
        # Use the compatible class instead
        try:
            # Important: Pass both joint angles and velocities
            env = CompatibleEnhancedEnv(
                num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, 
                command_cfg=command_cfg, show_viewer=True, 
                reference_motion=reference_motion_for_eval,
                reference_velocities=reference_velocities_for_eval,
                motion_filename=motion_filename_for_eval if motion_filename_for_eval else ''
            )
            print("Successfully created CompatibleEnhancedEnv with motion imitation.")
        except Exception as e:
            print(f"Failed to create CompatibleEnhancedEnv: {e}")
            print("Falling back to standard Go2Env")
            env = Go2Env(
                num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, 
                reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True
            )
    else: # Fallback to Go2Env
        env = Go2Env(
            num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, 
            reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True
        )

    print(f"Loading policy with num_obs={env.num_obs}, num_actions={env.num_actions}")
    policy = create_policy(env.num_obs, env.num_actions, loaded_policy_cfg_dict)
    
    try:
        loaded_dict = torch.load(model_path, map_location=gs.device) # Ensure model loads to correct device
        if "model_state_dict" in loaded_dict:
            policy.load_state_dict(loaded_dict["model_state_dict"])
        elif "optimizer_state_dict" in loaded_dict: # Older checkpoint from rsl_rl might have this structure
             policy.load_state_dict(loaded_dict["model_state_dict"])
        else: # Raw policy state_dict
            policy.load_state_dict(loaded_dict)
        policy.to(gs.device)
        policy.eval()
        print("Policy loaded successfully and set to eval mode.")
        
    except Exception as e:
        print(f"Error loading policy state_dict: {e}")
        import traceback
        traceback.print_exc()
        env.close() # Close env if policy load fails
        return

    obs, _ = env.reset()
    print(f"Running evaluation for {args.duration} seconds...")
    start_time = time.time()
    num_steps = 0
    
    try:
        with torch.no_grad():
            while True:
                actions = policy.act_inference(obs)
                
                try:
                    obs, _, dones, _ = env.step(actions)
                    num_steps += 1
                    
                    # Show reward breakdown if enabled
                    if args.show_rewards and num_steps % args.reward_interval == 0:
                        if hasattr(env, 'visualize_rewards') and callable(env.visualize_rewards):
                            env.visualize_rewards()
                    
                    if torch.any(dones):
                        print(f"Episode finished after {num_steps} steps.")
                        obs, _ = env.reset()
                        num_steps = 0 # Reset step count for new episode
                except KeyError as e:
                    # Catch reward scale compatibility issues
                    print(f"KeyError in environment step: {e}")
                    print("Try running with --compat flag to handle reward scale compatibility issues.")
                    break
                except Exception as e:
                    print(f"Error during environment step: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
                elapsed = time.time() - start_time
                if elapsed > args.duration:
                    print(f"Evaluation duration of {args.duration}s reached.")
                    break
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        print("Evaluation complete.")
        env.close() # Important to close the environment and viewer

if __name__ == "__main__":
    main()

"""
# evaluation
python policy_test/go2_eval_enhanced.py -m canter --duration 60
""" 