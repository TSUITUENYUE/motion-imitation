#!/usr/bin/env python3
"""
Script to visualize a trained Go2 policy without requiring the reference motion file.
This script demonstrates that the trained policy can run independently.
"""
import os
import torch
import genesis as gs
import argparse
import time

from go2_env import Go2Env
from genesis_motion_imitation import get_cfgs, get_train_cfg
from rsl_rl.modules import ActorCritic

def main():
    parser = argparse.ArgumentParser(description="Visualize trained policy (no reference motion needed)")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model file (e.g., logs/dog_walk_npy/model_999.pt)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration of visualization in seconds")
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Remove any motion-specific rewards since we don't have a reference motion
    if "joint_pose_matching" in reward_cfg["reward_scales"]:
        del reward_cfg["reward_scales"]["joint_pose_matching"]
    
    # Create environment with visualization enabled
    print("Creating environment...")
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True
    )
    
    # Load policy
    print(f"Loading policy from {args.model_path}")
    try:
        # Get train config to create an actor_critic network with the same architecture
        train_cfg = get_train_cfg("policy_visualization", 1)
        
        # Create a new actor_critic network
        actor_hidden_dims = train_cfg["policy"]["actor_hidden_dims"]
        critic_hidden_dims = train_cfg["policy"]["critic_hidden_dims"]
        activation = train_cfg["policy"]["activation"]
        
        num_obs = env.num_obs
        num_actions = env.num_actions
        
        print(f"Creating actor-critic network with obs: {num_obs}, actions: {num_actions}")
        print(f"Hidden layers: {actor_hidden_dims}, activation: {activation}")
        
        actor_critic = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation
        )
        
        # Load the state dict
        loaded_dict = torch.load(args.model_path)
        actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        actor_critic.to(gs.device)
        
        print("Policy loaded successfully")
        
        # Set to evaluation mode
        actor_critic.eval()
        
    except Exception as e:
        print(f"Error loading policy: {e}")
        print("Exception details:", str(e.__class__.__name__))
        import traceback
        traceback.print_exc()
        return
    
    # Reset environment to start visualization
    obs, _ = env.reset()
    
    print(f"Running policy visualization for {args.duration} seconds...")
    start_time = time.time()
    
    # Main visualization loop
    try:
        while (time.time() - start_time) < args.duration:
            # Get action from policy
            with torch.no_grad():
                actions = actor_critic.act_inference(obs)
            
            # Step the environment
            obs, _, reset_buf, _ = env.step(actions)
            
            # Reset if needed
            if reset_buf.any():
                obs, _ = env.reset()
            
    except KeyboardInterrupt:
        print("Visualization stopped by user")
    
    print("Visualization complete")

if __name__ == "__main__":
    main() 