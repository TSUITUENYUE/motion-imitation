# Go2 Motion Imitation

This project implements a motion imitation pipeline for the Go2 quadruped robot using the Genesis simulator, focusing on joint velocity matching for high-quality cyclic motions.

## Pipeline Overview

1. **Data Preparation**: We start with mocap data from the original paper and map skeleton data from videos.
2. **Motion Retargeting**: Convert mocap data from PyBullet format to Go2 robot in Genesis.
3. **Motion Processing**: Convert motion files to training format with enhanced joint velocity tracking.
4. **Policy Training**: Train motion imitation policies using reinforcement learning.
5. **Evaluation**: Visualize and evaluate trained policies in Genesis or MuJoCo.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Convert motion data for training (if needed)
cd rl_train
python convert_motion_for_training.py

# Train a policy
python train.py --file data/canter.npy --envs 256 --iters 1000 --viz

# Evaluate a trained policy
cd ..
python policy_test/go2_eval.py --motion canter --duration 60
```

## Data Preparation and Retargeting

### Skeleton Data from Video (`export_pose`)

The `export_pose` directory contains tools for mapping skeleton data from videos:

```bash
# Export poses from video to reference format
python export_pose/export_pose.py "breed-name"
```

This creates reference joint position data that can be used for retargeting.

You can also download our generated dataset through:
 https://drive.google.com/drive/folders/1twi4mgo6hTAaQ8ftO04MjWxnEjZkwwCr?usp=drive_link

### Motion Retargeting (`retarget_motion`)

The `retarget_motion` directory handles converting mocap data from PyBullet to the Go2 robot format:

```bash
# Retarget motion data to Go2 robot
python retarget_motion/retarget_motion.py --input_file=source_motion.txt --output_file=retargeted_motion.txt
```

### Genesis Visualization

The Genesis visualizer allows you to preview retargeted motions:

```bash
# Visualize retargeted motion
python retarget_motion/genesis_visualize_retarget.py --motion_file=retargeted_motion.txt

# Visualize specific motion type
python retarget_motion/genesis_visualize_retarget.py --motion canter
```

## Motion Conversion for Training

### Converting Motion Files

Two conversion utilities prepare motion data for training:

```bash
# Convert original .txt motion files to enhanced .npy format with joint velocities
cd rl_train
python convert_motion_for_training.py

# Update existing .npy files to include joint velocities
python convert_existing_npy.py --file data/canter.npy
```

The enhanced format focuses on frame-to-frame joint velocity matching, which is critical for cyclic motions like walking and running gaits.

## Training Process

### Basic Training

```bash
cd rl_train
python train.py --file data/canter.npy --envs 256 --iters 1000
```

### Enhanced Training Options

```bash
# Train with visualization
python train.py --file data/pace.npy --envs 256 --iters 2000 --viz

# Disable Weights & Biases logging
python train.py --file data/trot.npy --envs 512 --iters 1500 --no-wandb

# Resume training from a checkpoint
python train.py --resume --run-dir logs/go2-enhanced-canter --checkpoint 500 --iters 2000
```

Training logs and models are stored in the `rl_train/logs/` directory with appropriate experiment names based on the motion file.

## Evaluation and Visualization

### Genesis Evaluation

The `go2_eval.py` script provides the primary method for evaluating trained policies:

```bash
# Evaluate using motion type
python policy_test/go2_eval.py --motion canter --duration 60

# Evaluate specific model
python policy_test/go2_eval.py --model rl_train/logs/go2-enhanced-canter/model_999.pt

# Show reward breakdown
python policy_test/go2_eval.py -m trot --show-rewards
```

### MuJoCo Visualization

The `go2_mujoco.py` script provides an alternative visualization in MuJoCo:

```bash
# Basic MuJoCo visualization
python policy_test/go2_mujoco.py --motion canter

# Adjust command velocity
python policy_test/go2_mujoco.py --command 0.8 --motion trot

# Adjust controller parameters
python policy_test/go2_mujoco.py --stiffness 100.0 --damping 5.0 --motion pace
```

**Note**: The MuJoCo visualization requires careful tuning of controller parameters as they significantly differ from Genesis. There are known issues with parameter tuning in MuJoCo that may result in different motion quality compared to Genesis.

## Advanced Configuration

### Joint Velocity Matching

Our enhanced training focuses on joint velocity matching for improved motion quality:

1. Properly handles cyclic motion boundaries (last-to-first frame transitions)
2. Prioritizes velocity matching over position matching in rewards
3. Weights hip joints higher for better gait coordination

### Available Gaits

Pre-configured motion files include:
- `canter.npy`: Cantering gait with three-beat rhythm
- `trot.npy`: Trotting gait with diagonal leg pairs moving together
- `pace.npy`: Pacing gait with lateral leg pairs moving together
- `trot2.npy`: Alternative trotting pattern
- `left_turn.npy`/`right_turn.npy`: Turning motions

## Requirements

See `requirements.txt` for complete package requirements. Key dependencies:
- Genesis World (simulation environment)
- PyTorch
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Go2 Motion Imitation](#go2-motion-imitation)
  - [Pipeline Overview](#pipeline-overview)
  - [Quick Start](#quick-start)
  - [Data Preparation and Retargeting](#data-preparation-and-retargeting)
    - [Skeleton Data from Video (`export_pose`)](#skeleton-data-from-video-export_pose)
    - [Motion Retargeting (`retarget_motion`)](#motion-retargeting-retarget_motion)
    - [Genesis Visualization](#genesis-visualization)
  - [Motion Conversion for Training](#motion-conversion-for-training)
    - [Converting Motion Files](#converting-motion-files)
  - [Training Process](#training-process)
    - [Basic Training](#basic-training)
    - [Enhanced Training Options](#enhanced-training-options)
  - [Evaluation and Visualization](#evaluation-and-visualization)
    - [Genesis Evaluation](#genesis-evaluation)
    - [MuJoCo Visualization](#mujoco-visualization)
  - [Advanced Configuration](#advanced-configuration)
    - [Joint Velocity Matching](#joint-velocity-matching)
    - [Available Gaits](#available-gaits)
  - [Requirements](#requirements)
  - [Troubleshooting](#troubleshooting)
    - [MuJoCo Visualization Issues](#mujoco-visualization-issues)
    - [Training Issues](#training-issues)

<!-- /code_chunk_output -->


- rsl-rl-lib (reinforcement learning framework)
- MuJoCo (optional for alternative visualization)

## Troubleshooting

### MuJoCo Visualization Issues

If the robot appears unstable in MuJoCo visualization:
```bash
# Try adjusting PD controller parameters
python policy_test/go2_mujoco.py --stiffness 120.0 --damping 4.0 --motion canter
```

### Training Issues

If you encounter GPU memory issues:
```bash
# Reduce environment count
python rl_train/train.py --file data/canter.npy --envs 128 --iters 1000
```

For stability issues during training:
```bash
# Enable visualization to monitor robot behavior
python rl_train/train.py --file data/canter.npy --viz
``` 