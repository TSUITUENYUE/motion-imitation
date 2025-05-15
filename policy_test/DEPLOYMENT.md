# Go2 Robot Deployment Guide

This repository contains a Go2 quadruped robot implementation that runs in both Genesis and MuJoCo physics simulators using a trained neural network policy.

## Quick Start

### MuJoCo Implementation (Simplified Version)
```bash
# Run with default settings (command: 0.5)
python go2_mujoco_simple.py

# Run with custom forward velocity command
python go2_mujoco_simple.py --command 0.7
```

Command-line options:
- `--command` sets the forward velocity (default: 0.5)
- `--model` specifies a custom path to model weights (default: model_999.pt)

### Genesis Implementation
```bash
# Run the Genesis evaluation
python go2_eval.py -e go2-walking
```

## Required Files

The deployment requires these essential files:
- `go2_mujoco_simple.py` - Simplified MuJoCo implementation
- `go2_eval.py` - Genesis evaluation script
- `scene.xml` - MuJoCo scene configuration
- `go2.xml` - Go2 robot model
- `model_999.pt` - Trained policy weights

## System Requirements

- Python 3.8+
- PyTorch
- MuJoCo 
- Genesis
- numpy

## Implementation Details

Both implementations use an identical neural network architecture:
- Input: 45 features (robot state and commands)
  - 3 angular velocities
  - 3 gravity vector components
  - 3 command values
  - 12 joint positions
  - 12 joint velocities
  - 12 previous actions
- Output: 12 joint position targets
- Network: 45 → 512 → 256 → 128 → 12 (ELU activations)

## Control System

The simplified MuJoCo implementation uses:
- Proportional-Derivative (PD) control with gains Kp=30.0, Kd=1.0
- Action scaling of 0.25
- Fixed timestep of 0.02s (50Hz) to match Genesis
- Action latency simulation to match real robot behavior
- Configurable velocity commands

## Troubleshooting

1. **Robot Movement**: If the robot doesn't move as expected, try increasing the command value with `--command 0.7` or higher.
2. **Simulation Crashes**: The implementation has automatic recovery from instability.
3. **Model Loading**: If there are issues loading the model, make sure the model_999.pt file is in the correct location.

## Differences from Full Version

The simplified version has removed:
- Pattern mode (predefined walking gaits)
- Debug information display
- Additional visualization options

This version focuses only on running the trained neural network policy.
