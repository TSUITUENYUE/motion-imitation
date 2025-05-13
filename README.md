# Walkthrough: Training Go2 Walking Motion Imitation

This walkthrough provides step-by-step instructions for training a Go2 robot to imitate walking motion using the Genesis simulator.

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/motion_imitation.git
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