import argparse
import os
import pickle
import torch
from rsl_rl.modules import ActorCritic
import genesis as gs
import time

# Import from parent directory
import sys
sys.path.append("..")
from go2_env import Go2Env

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
        "reward_scales": {},  # Empty for evaluation
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def create_policy(num_obs, num_actions):
    """Create a policy network with standard architecture."""
    # Standard architecture from motion imitation
    hidden_dims = [512, 256, 128]
    activation = "elu"
    
    # Print info
    print(f"Creating actor-critic network with obs: {num_obs}, actions: {num_actions}")
    print(f"Hidden layers: {hidden_dims}, activation: {activation}")
    
    # Create network
    policy = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation=activation
    )
    
    return policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model_999.pt",
                        help="Model file name (should be in the same folder as the script)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration of evaluation in seconds")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    
    print(f"Loading model from: {model_path}")
    
    # Get default configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Create environment with visualization
    env = Go2Env(
        num_envs=1,  # Single environment for visualization
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Load the model directly
    print(f"Loading policy...")
    try:
        # Create policy network
        policy = create_policy(env.num_obs, env.num_actions)
        
        # Load the state dict
        loaded_dict = torch.load(model_path)
        policy.load_state_dict(loaded_dict["model_state_dict"])
        policy.to(gs.device)
        
        print("Policy loaded successfully")
        
        # Set to evaluation mode
        policy.eval()
        
    except Exception as e:
        print(f"Error loading policy: {e}")
        print("Exception details:", str(e.__class__.__name__))
        import traceback
        traceback.print_exc()
        return

    # Run evaluation loop
    obs, _ = env.reset()
    
    print(f"Running evaluation for {args.duration} seconds...")
    start_time = time.time()
    
    try:
        with torch.no_grad():
            while True:
                # Get policy action
                actions = policy.act_inference(obs)
                
                # Step environment
                obs, _, dones, _ = env.step(actions)
                
                # Reset if episode is done
                if torch.any(dones):
                    obs, _ = env.reset()
                
                # Check if duration elapsed
                elapsed = time.time() - start_time
                if elapsed > args.duration:
                    break
    
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user")
    
    print("Evaluation complete")

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""