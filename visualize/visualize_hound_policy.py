#!/usr/bin/env python3
"""
Script to visualize the trained hound motion imitation policy.
"""
import os
import torch
import genesis as gs
import argparse
import time

from go2_env import Go2Env
from RL_training.genesis_motion_imitation import load_reference_motion, get_cfgs, MotionImitationEnv, get_train_cfg
from rsl_rl.modules import ActorCritic

def main():
    parser = argparse.ArgumentParser(description="Visualize trained motion imitation policy")
    parser.add_argument("--model_path", type=str, default="logs/hound_motion_imitation/model_299.pt", 
                        help="Path to the trained model file")
    parser.add_argument("--motion_file", type=str, default="data/hound_joint_pos_retargeted.pt",
                        help="Path to the reference motion file")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration of visualization in seconds")
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    # Load reference motion
    reference_motion = load_reference_motion(args.motion_file)
    if reference_motion is None:
        print(f"Failed to load reference motion from {args.motion_file}")
        return
    
    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Configure reward for joint matching if using reference motion
    reward_cfg["reward_scales"]["joint_pose_matching"] = 1.0
    
    # Create environment with visualization enabled
    print("Creating environment...")
    env = MotionImitationEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        reference_motion=reference_motion,
        show_viewer=True
    )
    
    # Load policy
    print(f"Loading policy from {args.model_path}")
    try:
        # Get train config to create an actor_critic network with the same architecture
        train_cfg = get_train_cfg("hound_motion_imitation", 300)
        
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
        print(f"Actor network: {actor_critic.actor}")
        print(f"Critic network: {actor_critic.critic}")
        
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
    
    print(f"Running visualization for {args.duration} seconds...")
    start_time = time.time()
    
    # Main visualization loop
    try:
        while (time.time() - start_time) < args.duration:
            # Get action from policy
            with torch.no_grad():
                actions = actor_critic.act_inference(obs)
            
            # Step the environment
            obs, _, _, _ = env.step(actions)
            
            # Optional: Sleep to slow down visualization
            # time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Visualization stopped by user")
    
    print("Visualization complete")

if __name__ == "__main__":
    main() 