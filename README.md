# Walkthrough: Training Go2 Walking Motion Imitation

This walkthrough provides step-by-step instructions for training a Go2 robot to imitate walking motion using the Genesis simulator.

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/TSUITUENYUE/motion-imitation.git
cd motion_imitation

# Create and activate the conda environment
conda env create -f environment.yml
conda activate genesis

# Verify installation
python -c "import genesis; print(f'Genesis version: {genesis.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: Using pip

```bash
# Create a virtual environment (optional but recommended)
python -m venv motion_env
source motion_env/bin/activate  # On Windows: motion_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Note on CUDA

The PyTorch packages in the requirements use CUDA 12.8. If you have a different CUDA version, you'll need to install PyTorch compatible with your CUDA version. See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for details.

## Quick Start (5 minutes)

If you want to quickly see results with a pre-trained model:

```bash
# Visualize pre-trained walking policy
python visualize_trained_policy.py --model_path logs/go2-motion-imitation/model_999.pt --duration 30
```

## Complete Walkthrough

### Step 1: Set Up Your Environment

```bash
# Make sure you're in the right environment
conda activate genesis  # or your environment name
```

### Step 2: Download Walking Motion Data

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Download sample walking motion data (if not already available)
# For this example, we'll use the hound_joint_pos.txt file
```

### Step 3: Convert Motion Data to Tensor Format

```bash
# Convert motion data to PyTorch tensor format
python export_pose/export_pose.py "breed-name"
```

This creates `data/hound_joint_pos.pt`

### Step 4: Retarget Motion to Go2 Robot

```bash
# Generate retargeted motion with the run retargeter
python rl_train/go2_run_retarget.py

# Or specify custom parameters
python rl_train/go2_run_retarget.py --output output/custom_run.npy
```

This creates `output/run_retargeted.npy`

### Step 5: Visualize the Motion (Optional)

Before training, you can visualize the retargeted motion:

```bash
# Visualize the retargeted motion
python rl_train/visualize.py --file output/run_retargeted.npy

# Adjust playback speed and duration
python rl_train/visualize.py --file output/run_retargeted.npy --speed 1.5 --time 20.0
```

### Step 6: Train Motion Imitation Policy

```bash
# Train with the simplified training script
python rl_train/train.py

# Or specify different motion file, environments, and iterations
python rl_train/train.py --file output/custom_run.npy --envs 512 --iters 2000

# Enable visualization during training
python rl_train/train.py --file output/run_retargeted.npy --viz
```

The trained models will be saved to the `logs` directory, with the experiment name automatically generated from the motion file.

### Step 7: Monitor Training Progress

You can monitor training progress using TensorBoard or Weights & Biases:

```bash
# Using TensorBoard
tensorboard --logdir logs/

# Using Weights & Biases (enabled by default)
# Visit: https://wandb.ai/
# To disable W&B logging, use:
python rl_train/train.py --no-wandb
```

## Reward Functions

The training uses the following reward components:

1. **Joint Pose Matching** (weight: 1.0)
   - Main reward for imitating the reference motion
   - Calculated as exp(-MSE/1.0) where MSE is the mean squared error between joints

2. **Tracking Linear Velocity** (weight: 1.0)
   - Encourages the robot to maintain the commanded linear velocity

3. **Tracking Angular Velocity** (weight: 0.2)
   - Encourages the robot to maintain the commanded angular velocity

4. **Linear Velocity Z** (weight: -1.0)
   - Penalizes vertical motion (bouncing)

5. **Base Height** (weight: -50.0)
   - Penalizes deviation from target height

6. **Action Rate** (weight: -0.005)
   - Penalizes rapid changes in actions
   - Smooths motion by discouraging jerky movements

7. **Default Pose Similarity** (weight: -0.1)
   - Penalizes deviation from default pose
   - Encourages return to neutral pose when appropriate

## Additional Options and Tips

### Training with Different Parameters

- Increase `--envs` for faster training (requires more GPU memory)
- Increase `--iters` for potentially better results
- Use `--viz` to visualize during training (slows down training)

### Using Your Own Motion Data

1. Convert your motion data to CSV format (frame × joint angles)
2. Use the convert script to create a tensor format
3. Retarget to Go2 joint space
4. Train as normal

### Common Issues

- **Out of memory errors**: Reduce `--envs` to use less GPU memory
- **Unstable training**: Add more training iterations
- **Poor motion quality**: Ensure motion retargeting is accurate
- **W&B errors**: If you encounter W&B initialization errors, you can:
  - Make sure you've logged in with `wandb login`
  - Disable W&B with the `--no-wandb` flag
  - Check if your W&B API key is properly set in the environment

## Troubleshooting

### AttributeError: 'OnPolicyRunner' object has no attribute 'train_cfg'

This error might occur with older versions of the code. The solution is to update the WandbCallback class to use the experiment name directly instead of trying to access it through runner.train_cfg.

### ImportError: No module named 'wandb'

Install Weights & Biases with:
```bash
pip install wandb
```

### CUDA-related errors

If you encounter CUDA errors, make sure the PyTorch version in your environment matches your CUDA version. You may need to modify the PyTorch installation in environment.yml:

```yaml
- pip:
    - torch==2.7.0+cuXXX  # Replace XXX with your CUDA version (e.g., cu118)
```

## Project Structure

- `rl_train/`: Contains the main training and visualization scripts
  - `train.py`: Main training script
  - `visualize.py`: Motion visualization
  - `go2_run_retarget.py`: Motion retargeting

- `output/`: Default output directory for retargeted motions
- `logs/`: Training logs and saved models

## Requirements

See `requirements.txt` for complete package requirements or use the provided `environment.yml` file to create a conda environment with all dependencies. 

## Evaluating Trained Policies

After training a policy, you can evaluate it using the provided evaluation scripts.

> **Note on Paths**: When specifying model paths, make sure they are relative to your current working directory. If you encounter a `FileNotFoundError`, check that your path correctly points to the model file.

### Using go2_eval.py (Genesis Simulator)

The `go2_eval.py` script can automatically find and load the latest checkpoint from your training logs:

```bash
# Run with automatic checkpoint detection (finds the most recent log directory)
python policy_test/go2_eval.py

# Specify a particular motion type to load its latest checkpoint
python policy_test/go2_eval.py --motion canter

# Specify a custom evaluation duration (in seconds)
python policy_test/go2_eval.py --motion pace --duration 60

# Manually specify a model file (from root directory)
python policy_test/go2_eval.py --model rl_train/logs/go2-imitate-canter/model_499.pt
```

### Using go2_mujoco.py (MuJoCo Simulator)

The `go2_mujoco.py` script provides an alternative evaluation in the MuJoCo simulator:

```bash
# Run with the default model (looks for model_999.pt in the same directory)
python policy_test/go2_mujoco.py

# Specify a custom model path (from root directory)
python policy_test/go2_mujoco.py --model rl_train/logs/go2-imitate-pace/model_499.pt

# Adjust the forward velocity command (default is 0.5 m/s)
python policy_test/go2_mujoco.py --command 0.8
```

Both evaluation scripts provide similar functionality with different simulation backends, allowing you to see how your trained policy performs in controlling the Go2 robot. 

# Motion Imitation RL Training

This directory contains scripts for training a Go2 robot to imitate reference motions using Reinforcement Learning.

## Setup

Ensure you have the Genesis simulation environment and necessary Python packages (PyTorch, rsl-rl-lib, wandb, etc.) installed in your active Python environment.

## Training

To start a new training run from the `motion-imitation/rl_train/` directory:

```bash
python train.py --file path/to/your/motion_file.npy --envs <num_environments> --iters <num_iterations> --viz
```

**Arguments for `train.py`:**

*   `--file`: (Required) Path to the reference motion file (e.g., `data/canter.npy`, `data/trot.npy`).
*   `--envs`: Number of parallel simulation environments to use for training (default: 256). More environments can speed up data collection but require more resources.
*   `--iters`: Maximum number of training iterations (default: 1000).
*   `--viz`: (Optional) Add this flag to enable visualization of one of the training environments. This will slow down training.
*   `--no-wandb`: (Optional) Add this flag to disable Weights & Biases logging.
*   `--wandb-project`: (Optional) Specify a custom Weights & Biases project name (default: `go2-motion-imitation`).

**Example Training Command:**

```bash
# From the motion-imitation/rl_train/ directory
python train.py --file data/pace.npy --envs 256 --iters 2000 --viz
```

This command will train a policy to imitate the `pace.npy` motion, using 256 environments for 2000 iterations, with visualization enabled.
Training logs and model checkpoints will be saved under the `logs/go2-imitate-<motion_filename>/` directory.

## Evaluation

To evaluate a trained policy from the `motion-imitation/rl_train/` directory:

```bash
python ../policy_test/go2_eval.py --run_dir path/to/your/training_run_directory --model <model_checkpoint_name.pt> --duration <seconds>
```

**Arguments for `go2_eval.py`:**

*   `--run_dir`: (Required) Path to the specific training run directory that contains the model and `cfgs.pkl` file (e.g., `logs/go2-imitate-canter-grounded/`).
*   `--model`: (Optional) Name of the model checkpoint file within the `run_dir` (e.g., `model_499.pt`). If not provided, the script will attempt to load the model with the highest iteration number from the specified `run_dir`.
*   `--duration`: (Optional) Duration of the evaluation in seconds (default: 30).

**Example Evaluation Commands:**

1.  **Evaluate a specific model checkpoint:**

    ```bash
    # From the motion-imitation/rl_train/ directory
    python ../policy_test/go2_eval.py --run_dir logs/go2-imitate-canter-grounded/ --model model_499.pt --duration 60
    ```

2.  **Evaluate the latest model in a run directory:**

    ```bash
    # From the motion-imitation/rl_train/ directory
    python ../policy_test/go2_eval.py --run_dir logs/go2-imitate-canter-grounded/ --duration 60
    ```

This will load the specified policy and run it in the simulation environment with visualization, using the configurations stored during its training.

## Enhanced Evaluation Script

You can now use the new `go2_eval_enhanced.py` script to evaluate any models trained with your enhanced environment. The script can be run with:

```bash
# From the motion-imitation/rl_train/ directory
python ../policy_test/go2_eval_enhanced.py -m canter --duration 60
```

This will automatically find the latest model for the "canter" motion type in either the enhanced or original log directories. You can also specify:

- A specific model file with `--model`
- A specific run directory with `--run_dir`
- A specific experiment name with `-e/--experiment`
- A specific checkpoint iteration with `--ckpt`

The script will handle all the necessary imports and configurations to ensure the enhanced environment works correctly with the trained models. 

**Example Enhanced Evaluation Commands:**

1.  **Find and evaluate the latest model for a specific motion type:**

    ```bash
    # From the motion-imitation/rl_train/ directory
    python ../policy_test/go2_eval_enhanced.py -m trot --duration 60
    ```

2.  **Evaluate a specific experiment by name:**

    ```bash
    # From the motion-imitation/rl_train/ directory
    python ../policy_test/go2_eval_enhanced.py -e go2-enhanced-canter --duration 60
    ```

3.  **Evaluate a specific checkpoint from an experiment:**

    ```bash
    # From the motion-imitation/rl_train/ directory
    python ../policy_test/go2_eval_enhanced.py -e go2-enhanced-canter --ckpt 500 --duration 60
    ```

## Notes

*   Ensure that the paths to motion files and log directories are correct based on your current working directory.
*   The evaluation scripts rely on the `cfgs.pkl` file being present in the specified `--run_dir` to accurately replicate the training conditions.
*   For `go2_eval_enhanced.py`, always run from the `rl_train` directory using the path `../policy_test/go2_eval_enhanced.py` 

# Go2 Motion Imitation with Joint Velocity Matching

This repository contains tools for training a Go2 quadruped robot to imitate reference motions using reinforcement learning with a focus on joint velocity matching for better cyclic motion performance.

## Quick Start Guide

### Training a Motion

```bash
# Convert original motion file to the joint velocity format
cd motion-imitation/rl_train
python convert_existing_npy.py --file data/canter.npy

# Train using the joint velocity-enhanced version
python train_experiment.py --file data/canter_new.npy --envs 256 --iters 1000
```

### Visualizing Trained Results

```bash
# Navigate to the rl_train directory 
cd motion-imitation/rl_train

# Use the simple test script to visualize canter_new with specific checkpoint:
python ../policy_test/test_canter_new.py 999  # Checkpoint number 999
python ../policy_test/test_canter_new.py 500  # Or checkpoint 500

# Or use the full evaluation script with explicit options:
python ../policy_test/go2_eval_enhanced.py -e go2-enhanced-canter_new --ckpt 999
python ../policy_test/go2_eval_enhanced.py --run_dir logs/go2-enhanced-canter_new --ckpt 500

# For other motion types:
python ../policy_test/go2_eval_enhanced.py -m canter --duration 60
```

### Delete Previous Training Results

```bash
# Remove a specific training run
rm -rf motion-imitation/rl_train/logs/go2-enhanced-canter

# Or clean all training runs
rm -rf motion-imitation/rl_train/logs/go2-enhanced-*
```

## Enhanced Joint Velocity Training

The enhanced training method focuses on frame-to-frame joint velocity matching, which is critical for cyclic motions like walking and running gaits. This approach:

1. Captures the dynamic nature of motion rather than just static poses
2. Better handles short, cyclic reference motions by focusing on transitions
3. Improves motion continuity and natural flow of movement

### Key Features:

- Calculates joint velocities directly from consecutive frames
- Properly handles cyclic motion boundaries (last-to-first frame transitions)
- Prioritizes velocity matching over position matching in rewards
- Weights hip joints higher for better gait coordination

## Available Commands and Options

### Training Options

```bash
python train_experiment.py --file DATA_FILE [options]
```

Options:
- `--file`: Path to motion data file (.npy format)
- `--envs`: Number of parallel environments (default: 256)
- `--iters`: Number of training iterations (default: 1000)
- `--viz`: Enable visualization during training
- `--no-wandb`: Disable Weights & Biases logging
- `--wandb-project`: Custom W&B project name
- `--resume`: Resume from a previous checkpoint
- `--run-dir`: Training directory to resume from
- `--checkpoint`: Specific checkpoint to resume from

### Evaluation Options

```bash
python ../policy_test/go2_eval_enhanced.py [options]
```

Options:
- `-m/--motion TYPE`: Motion type (e.g., canter, trot, pace)
- `--model PATH`: Path to specific model file
- `--run_dir DIR`: Specific training run directory
- `-e/--experiment NAME`: Experiment name
- `--ckpt NUMBER`: Specific checkpoint iteration
- `--duration SECONDS`: Evaluation duration (default: 30s)

## Motion Data Preparation

The system now uses joint velocity data for better motion quality:

1. Original `.txt` motion files are converted to `.npy` files with `convert_motion_for_training.py`
2. Existing `.npy` files can be updated to include joint velocities with `convert_existing_npy.py`

```bash
# Convert txt to npy with joint velocities
python convert_motion_for_training.py

# Update existing npy files to include joint velocities
python convert_existing_npy.py --file data/your_motion.npy
```

For more detailed information, see the full documentation. 