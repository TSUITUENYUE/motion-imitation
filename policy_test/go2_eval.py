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

# If train.py and its MotionImitationEnv are in rl_train, we need to import that version
# For now, let's assume MotionImitationEnv might be needed if cfgs.pkl expects it.
# We should ideally use the *exact* same environment class used during training.

# Try to import MotionImitationEnv from the expected training script location
# This is crucial if cfgs.pkl was saved with an instance of MotionImitationEnv
MOTION_IMITATION_ENV_IMPORTED = False
try:
    # Assuming train.py is in ../rl_train relative to this script (policy_test/go2_eval.py)
    from rl_train.train import MotionImitationEnv, load_reference_motion # Make sure these are importable
    # Also need get_cfgs from train.py if we are truly replicating the train env for some reason
    # However, the goal is to load cfgs.pkl, which should contain all necessary env_cfg details.
    MOTION_IMITATION_ENV_IMPORTED = True
    print("Successfully imported MotionImitationEnv from rl_train.train")
except ImportError as e:
    print(f"Warning: Could not import MotionImitationEnv from rl_train.train: {e}")
    print("Falling back to default Go2Env. This might cause issues if the model was trained with MotionImitationEnv specific rewards/logic not covered by cfgs.pkl.")
    # Go2Env is already imported, so no need for a fallback import here

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
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
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
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {  # Standard reward scales for evaluation
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [1.0, 1.0],  # Higher target velocity to match training
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
        # Add other params if your policy class takes them e.g. from train_cfg["policy"]
    )
    return policy

def find_latest_checkpoint(motion_type=None):
    """Find the checkpoint with the highest iteration number in the logs directory."""
    # Get the root directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level and into the logs directory
    logs_dir = os.path.join(script_dir, "..", "rl_train", "logs")
    
    # If motion_type is specified, look in that specific directory
    if motion_type:
        logs_dir = os.path.join(logs_dir, f"go2-imitate-{motion_type}")
        if not os.path.exists(logs_dir):
            print(f"Log directory for motion type '{motion_type}' not found")
            return None
    
    # Find all checkpoint directories
    log_dirs = []
    for item in os.listdir(logs_dir):
        full_path = os.path.join(logs_dir, item)
        if os.path.isdir(full_path) and item.startswith("go2-imitate-"):
            log_dirs.append(full_path)
    
    if not log_dirs:
        print(f"No log directories found in {logs_dir}")
        return None
    
    # If no specific motion type, use the most recent directory
    if not motion_type:
        # Sort by modification time (most recent first)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model file name (e.g., model_499.pt or path like logs/exp/model_499.pt). If not provided, latest in --run_dir is used.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to the training run directory (e.g., logs/go2-imitate-canter-grounded). If not given, inferred from --model path if possible.")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration of evaluation in seconds")
    args = parser.parse_args()

    gs.init()
    
    model_path = None
    cfg_path = None
    run_dir_path = args.run_dir

    if args.model:
        if os.path.isabs(args.model):
            model_path = args.model
            # If run_dir not given, try to infer it from absolute model_path
            if not run_dir_path:
                run_dir_path = os.path.dirname(model_path)
                print(f"Inferred run_dir from absolute model path: {run_dir_path}")
        else: # args.model is a relative path
            if run_dir_path: # Relative model name, relative to run_dir
                model_path = os.path.join(run_dir_path, args.model)
            else: # Relative model path (e.g., logs/exp/model.pt), relative to CWD
                model_path = os.path.abspath(args.model) # Resolve relative to CWD
                run_dir_path = os.path.dirname(model_path) # Infer run_dir
                print(f"Interpreting --model '{args.model}' as relative to CWD: {model_path}")
                print(f"Inferred run_dir from relative model path: {run_dir_path}")

    # If model_path is still not set (e.g. only run_dir was given, or model arg was just a name)
    if not model_path and run_dir_path:
        print(f"Searching for latest model in specified run_dir: {run_dir_path}")
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
    if run_dir_path and os.path.isdir(run_dir_path): # Ensure run_dir_path is valid before using
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
                # reward_cfg is less critical for eval physics but good to have for consistency if MotionImitationEnv uses it
                if reward_cfg is None: _, _, reward_cfg, _ = get_cfgs() # type: ignore
    else:
        print("Warning: cfgs.pkl not found. Using default configurations from go2_eval.py. This may lead to unexpected behavior if the model was trained with different settings.")
        # Load default configurations if cfgs.pkl is not found
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        # Use default policy arch if no train_cfg loaded
        loaded_policy_cfg_dict = {"actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128], "activation": "elu", "init_noise_std": 1.0}

    # --- Environment Creation --- 
    # Determine which environment class to use
    EnvClassToUse = Go2Env # Default
    reference_motion_for_eval = None
    motion_filename_for_eval = None

    # Determine if MotionImitationEnv should be used
    use_motion_imitation_env = False
    if MOTION_IMITATION_ENV_IMPORTED:
        # Priority 1: motion_filename explicitly stored in loaded env_cfg (from train.py)
        if env_cfg and env_cfg.get("motion_filename"):
            motion_filename_for_eval = env_cfg["motion_filename"]
            if os.path.exists(motion_filename_for_eval):
                print(f"Using motion_filename from env_cfg: {motion_filename_for_eval}")
                use_motion_imitation_env = True
            else:
                print(f"Warning: motion_filename '{motion_filename_for_eval}' from env_cfg not found. Will try to infer.")
                motion_filename_for_eval = None # Reset to trigger inference

        # Priority 2: Infer from experiment name if not found in env_cfg or if path was invalid
        if not use_motion_imitation_env and train_cfg_loaded and train_cfg_loaded.get("runner", {}).get("experiment_name", "").startswith("go2-imitate"):
            exp_name = train_cfg_loaded.get("runner", {}).get("experiment_name", "")
            # Try to extract the base motion name (e.g., "canter" from "go2-imitate-canter-grounded" or "go2-imitate-canter")
            match = re.search(r"go2-imitate-([a-zA-Z0-9_]+)(?:-[a-zA-Z0-9_]+)*", exp_name)
            if match:
                inferred_motion_name = match.group(1) # This should be the core motion name like "canter"
                print(f"Inferred base motion name from exp_name '{exp_name}': '{inferred_motion_name}'")
                if model_path:
                    # Construct path to standard data directory relative to this script's location
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    # Assumes structure: ./policy_test/go2_eval.py, ../rl_train/data/motion.npy
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
             print("env_cfg suggests MotionImitationEnv, but motion_filename could not be determined. Proceeding with MotionImitationEnv without reference motion.")
             use_motion_imitation_env = True # Will use MotionImitationEnv but reference_motion might be None

    if use_motion_imitation_env:
        print("Using MotionImitationEnv for evaluation.")
        EnvClassToUse = MotionImitationEnv
        if motion_filename_for_eval:
            if MOTION_IMITATION_ENV_IMPORTED: # Ensure class was actually imported
                reference_motion_for_eval = load_reference_motion(motion_filename_for_eval)
                if reference_motion_for_eval is None:
                    print(f"Warning: load_reference_motion returned None for '{motion_filename_for_eval}'.")
            else:
                 print("MotionImitationEnv class was not imported, cannot load reference motion.")
        else:
            print("Warning: No motion_filename available for MotionImitationEnv. Motion imitation rewards will be zero or behave unexpectedly.")
    else:
        print("Using Go2Env for evaluation.")
        EnvClassToUse = Go2Env

    print(f"Using Environment Class: {EnvClassToUse.__name__}")
    print("Using env_cfg for evaluation:", env_cfg) # Log the actual env_cfg being used

    # Create environment instance
    if EnvClassToUse == MotionImitationEnv:
        env = MotionImitationEnv(
            num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, 
            command_cfg=command_cfg, show_viewer=True, 
            reference_motion=reference_motion_for_eval, # Pass loaded reference motion
            motion_filename=motion_filename_for_eval if motion_filename_for_eval else '' # Pass motion filename
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
                obs, _, dones, _ = env.step(actions)
                num_steps += 1
                if torch.any(dones):
                    print(f"Episode finished after {num_steps} steps.")
                    obs, _ = env.reset()
                    num_steps = 0 # Reset step count for new episode
                
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
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""