#!/usr/bin/env python3
"""
Genesis to MuJoCo Sim2Sim Transfer

This script implements sim2sim transfer from Genesis to MuJoCo for the Go2 robot.
It loads a trained policy from Genesis and creates a standalone model for MuJoCo simulation.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import time
import argparse
import pickle

class Go2PolicyNetwork(nn.Module):
    """
    Neural network for the Go2 robot policy.
    Takes observations as input and predicts actions.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], activation='elu'):
        super(Go2PolicyNetwork, self).__init__()
        
        # Create layers list
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers with activation
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if activation.lower() == 'elu':
                layers.append(nn.ELU())
            elif activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            prev_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)


def load_genesis_policy(model_path, input_dim=45, output_dim=12, hidden_dims=[512, 256, 128], activation='elu'):
    """
    Load a trained Genesis policy and convert it to a standalone neural network.
    
    Args:
        model_path: Path to the Genesis model file (.pt)
        input_dim: Dimension of the input observations
        output_dim: Dimension of the output actions
        hidden_dims: List of hidden layer dimensions
        activation: Activation function to use
        
    Returns:
        policy_network: Trained policy network
    """
    print(f"Loading Genesis policy from {model_path}")
    
    # Create policy network
    policy_network = Go2PolicyNetwork(input_dim, output_dim, hidden_dims, activation)
    
    # Load trained model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract actor network state dict
    if 'model_state_dict' in state_dict:
        model_state_dict = state_dict['model_state_dict']
        
        # Extract only the actor network weights (not the critic)
        actor_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('actor.'):
                # Convert key from 'actor.0.weight' to '0.weight'
                new_key = key[len('actor.'):]
                actor_state_dict[f'model.{new_key}'] = value
                
        # Load state dict into the network
        policy_network.load_state_dict(actor_state_dict)
        print("Successfully loaded actor weights from Genesis model")
    else:
        raise ValueError("Model file does not contain 'model_state_dict' key")
    
    # Set to evaluation mode
    policy_network.eval()
    
    return policy_network


def save_mujoco_model(model, output_path):
    """
    Save the model for MuJoCo inference.
    
    Args:
        model: Trained policy network
        output_path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), output_path)
    print(f"Saved MuJoCo-compatible model to {output_path}")


def export_onnx_model(model, output_path, input_dim=45):
    """
    Export the model to ONNX format for efficient inference.
    
    Args:
        model: Trained policy network
        output_path: Path to save the ONNX model
        input_dim: Dimension of the input observations
    """
    # Create dummy input
    dummy_input = torch.randn(1, input_dim)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    print(f"Exported ONNX model to {output_path}")


def observation_normalizer(observation, observation_mean=None, observation_std=None):
    """
    Normalize observations based on training statistics.
    
    Args:
        observation: Raw observation
        observation_mean: Mean of observations (optional)
        observation_std: Standard deviation of observations (optional)
        
    Returns:
        normalized_observation: Normalized observation
    """
    if observation_mean is not None and observation_std is not None:
        return (observation - observation_mean) / observation_std
    return observation


def test_inference(model, input_dim=45, noise_level=0.1):
    """
    Test model inference with random inputs.
    
    Args:
        model: Policy network
        input_dim: Dimension of input observations
        noise_level: Standard deviation of random noise for test
    """
    print("Testing model inference with random inputs...")
    
    # Generate random input
    test_input = torch.randn(1, input_dim) * noise_level
    
    # Time the inference
    start_time = time.time()
    with torch.no_grad():
        output = model(test_input)
    end_time = time.time()
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Example output: {output[0][:3].numpy()} (first 3 values)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Genesis to MuJoCo sim2sim transfer")
    parser.add_argument("--model", type=str, default="logs/go2-motion-imitation/model_999.pt",
                        help="Path to the Genesis trained model")
    parser.add_argument("--output", type=str, default="mujoco_models/go2_policy.pt",
                        help="Output path for the MuJoCo-compatible model")
    parser.add_argument("--export_onnx", action="store_true",
                        help="Export the model to ONNX format")
    parser.add_argument("--input_dim", type=int, default=45,
                        help="Dimension of input observations")
    parser.add_argument("--output_dim", type=int, default=12,
                        help="Dimension of output actions")
    args = parser.parse_args()
    
    # Load Genesis policy
    policy_network = load_genesis_policy(
        args.model,
        input_dim=args.input_dim,
        output_dim=args.output_dim
    )
    
    # Save model for MuJoCo
    save_mujoco_model(policy_network, args.output)
    
    # Export to ONNX if requested
    if args.export_onnx:
        onnx_path = args.output.replace('.pt', '.onnx')
        export_onnx_model(policy_network, onnx_path, args.input_dim)
    
    # Test the model
    test_inference(policy_network, args.input_dim)
    
    print("Genesis to MuJoCo transfer complete!")


if __name__ == "__main__":
    main() 