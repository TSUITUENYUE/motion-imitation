# Go2 Motion Imitation

This project allows you to retarget dog motion data to a Go2 robot, visualize the retargeted motions, and train policies to replicate these motions.

## Setup

Before using these scripts, ensure you have:

1. Genesis installed with PyTorch support
2. RSL-RL-lib version 2.2.4 installed:
   ```
   pip uninstall rsl_rl -y  # Remove old version if present
   pip install rsl-rl-lib==2.2.4
   ```
3. Weights & Biases for experiment tracking (enabled by default):
   ```
   pip install wandb
   ```

## File Structure

The system consists of three main scripts:

1. `go2_run_retarget.py` - Generates retargeted running motion
2. `visualize.py` - Visualizes retargeted motion files
3. `train.py` - Trains a policy to imitate the retargeted motion

## Workflow

### 1. Retarget Dog Motion

First, generate a retargeted motion file:

```bash
# Generate a run motion with default parameters
python go2_run_retarget.py

# Or specify custom parameters
python go2_run_retarget.py --output output/custom_run.npy
```

This creates an `.npy` file in the `output` directory containing the retargeted motion.

### 2. Visualize the Motion

Before training, you can visualize the retargeted motion:

```bash
# Visualize the default run motion
python visualize.py --file output/run_retargeted.npy

# Adjust playback speed and duration
python visualize.py --file output/run_retargeted.npy --speed 1.5 --time 20.0

# Compare multiple motions
python visualize.py --compare output/run_retargeted.npy output/custom_run.npy --labels "Default Run" "Custom Run"
```

### 3. Train a Policy

Train a reinforcement learning policy to imitate the motion:

```bash
# Train using default run motion
python train.py

# Train with a different motion file 
python train.py --file output/custom_run.npy

# Adjust environment count and training iterations
python train.py --file output/run_retargeted.npy --envs 512 --iters 2000

# Enable visualization during training
python train.py --file output/run_retargeted.npy --viz

# Disable W&B logging if needed
python train.py --file output/run_retargeted.npy --no-wandb
```

Training results are saved in the `logs` directory.

## Available Parameters

### go2_run_retarget.py
- `--file`: Custom motion file path
- `--start`: Start frame for custom file
- `--end`: End frame for custom file
- `--output`: Output file path (default: output/run_retargeted.npy)

### visualize.py
- `--file`: Motion file to visualize
- `--time`: Playback time in seconds (default: 15.0)
- `--speed`: Playback speed factor (default: 1.0)
- `--no-loop`: Disable looping of motion
- `--compare`: List of motion files to compare
- `--labels`: Labels for comparison

### train.py
- `--file`: Path to retargeted motion file (default: output/run_retargeted.npy)
- `--envs`: Number of parallel training environments (default: 256)
- `--iters`: Maximum number of training iterations (default: 1000)
- `--viz`: Enable visualization during training
- `--no-wandb`: Disable Weights & Biases logging (enabled by default)
- `--wandb-project`: W&B project name (default: "go2-motion-imitation")

## Reward Functions

The training uses the following reward components:

1. **Joint Pose Matching** (weight: 1.0)
   - Main reward for imitating the reference motion
   - Calculated as exp(-MSE/1.0) where MSE is the mean squared error between joints

2. **Tracking Linear Velocity** (weight: 1.0)
   - Encourages the robot to maintain the commanded linear velocity
   - Calculated as exp(-error/0.25) for x-y velocity

3. **Tracking Angular Velocity** (weight: 0.2)
   - Encourages the robot to maintain the commanded angular velocity
   - Calculated as exp(-error/0.25) for yaw velocity

4. **Linear Velocity Z** (weight: -1.0)
   - Penalizes vertical motion (bouncing)
   - Calculated as square of z-velocity

5. **Base Height** (weight: -50.0)
   - Penalizes deviation from target height (0.3m)
   - Calculated as square of height difference

6. **Action Rate** (weight: -0.005)
   - Penalizes rapid changes in actions
   - Smooths motion by discouraging jerky movements

7. **Default Pose Similarity** (weight: -0.1)
   - Penalizes deviation from default pose
   - Encourages return to neutral pose when appropriate

All rewards are combined with their respective weights to form the final reward.

## Weights & Biases Integration

The training script uses Weights & Biases (W&B) by default for experiment tracking and visualization. It will:

1. Log all rewards and losses
2. Track training progress in real-time
3. Store hyperparameters and configurations
4. Allow comparison between different runs

You can view your experiments at https://wandb.ai after logging in.

To use W&B:
```bash
# First, log in to your W&B account
wandb login

# Then run training (W&B is enabled by default)
python train.py --file output/run_retargeted.npy

# Specify a custom project name
python train.py --file output/run_retargeted.npy --wandb-project "my-custom-project"

# Disable W&B logging if needed
python train.py --file output/run_retargeted.npy --no-wandb
```

## Tips

1. Start with the default run motion, then experiment with custom parameters.
2. Use visualization to check if the retargeted motion looks natural.
3. For faster training, increase the environment count (`--envs`) if your hardware supports it.
4. The experiment name is automatically generated from the motion file name.
5. Track your training progress with W&B to identify the best hyperparameters. 