# Walkthrough: Training Go2 Walking Motion Imitation

This walkthrough provides step-by-step instructions for training a Go2 robot to imitate walking motion using the Genesis simulator.

## Quick Start (5 minutes)

If you want to quickly see results with a pre-trained model:

```bash
# Visualize pre-trained walking policy
python visualize_trained_policy.py --model_path logs/go2-motion-imitation/model_999.pt --duration 30
```

## Complete Walkthrough

### Step 1: Set Up Your Environment

Ensure you have the Genesis environment set up:

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

This creates `data/hound_joint_pos.pt`

### Step 4: Retarget Motion to Go2 Robot

```bash
# Retarget motion data to match Go2 robot joint structure
python retarget_hound_motion.py
```

This creates `data/hound_joint_pos_retargeted.pt`

### Step 5: Train Motion Imitation Policy

```bash
# Train the policy with a PyTorch tensor file (.pt)
python genesis_motion_imitation.py --exp_name go2_walking \
    --motion_file data/hound_joint_pos_retargeted.pt \
    --max_iterations 1000 \
    --num_envs 256

# OR directly use a NumPy array file (.npy) - recommended
python genesis_motion_imitation.py --exp_name dog_walk_npy \
    --motion_file dog_walk_retargeted.npy \
    --max_iterations 1000 \
    --num_envs 256
```

The script now directly supports both .pt and .npy files, so you can use whichever format you have available. 

Using .npy files:
- No conversion needed if you already have .npy files
- Format should be [num_frames, 12] where 12 is the number of joint angles for the Go2 robot
- Values should be in radians

The trained models will be saved to `logs/go2_walking/`

### Step 6: Visualize Trained Policy

```bash
# Visualize using the original motion file (same format as used for training)
python visualize/visualize_hound_policy.py --model_path logs/go2_walking/model_999.pt \
    --motion_file data/hound_joint_pos_retargeted.pt \
    --duration 30

# If you trained with a .npy file, use it for visualization too
python visualize/visualize_hound_policy.py --model_path logs/dog_walk_npy/model_999.pt \
    --motion_file dog_walk_retargeted.npy \
    --duration 30

# OR use the policy-only visualizer which doesn't need the reference motion
# This demonstrates that the policy works completely independently
python visualize/visualize_policy_only.py --model_path logs/dog_walk_npy/model_999.pt --duration 30
```

The first two visualization methods use the reference motion for environment setup, but the policy-only version shows that the trained policy works entirely on its own without needing the reference motion data.

### Step 7: Deploy to MuJoCo (Optional)

For deployment in MuJoCo:

```bash
# Export policy to MuJoCo-compatible format
python genesis_to_mujoco.py --model logs/go2_walking/model_999.pt \
    --output mujoco_models/go2_walking_policy.pt \
    --export_onnx

# Run in MuJoCo
python mujoco_run_go2_policy.py --model mujoco_models/go2_walking_policy.pt \
    --duration 30
```

## Additional Options

### Training with Different Parameters

```bash
# Train with different learning rate
python genesis_motion_imitation.py --exp_name go2_walking_lr0001 \
    --motion_file data/hound_joint_pos_retargeted.pt \
    --max_iterations 1000 \
    --num_envs 256 \
    --learning_rate 0.0001
```

### Visualizing During Training

You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir logs/go2_walking
```

### Using Your Own Motion Data

1. Convert your motion data to CSV format (frame × joint angles)
2. Use the convert script to create a tensor format
3. Retarget to Go2 joint space
4. Train as normal

## Common Issues

- **Out of memory errors**: Reduce `--num_envs` to use less GPU memory
- **Unstable training**: Reduce learning rate or increase iterations
- **Poor motion quality**: Ensure motion retargeting is accurate

## Next Steps

- Try different motion styles (trotting, bounding, etc.)
- Adjust reward functions for smoother motion
- Combine multiple motions into a single policy 