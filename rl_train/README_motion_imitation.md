# Motion Imitation Workflow

This directory contains tools to visualize retargeted motion data and train reinforcement learning policies to imitate those motions on the Unitree Go2 robot.

## Workflow Overview

The workflow consists of these main steps:

1. **Visualize the retargeted motion** to ensure it looks correct on the Go2 robot
2. **Convert the motion data** from JSON format (.txt) to the format required for training (.npy/.pt)
3. **Train a policy** using the converted motion data
   - Option A: Standard training with `train.py`
   - Option B: Enhanced paper-style training with `train_experiment.py`
4. **Evaluate the trained policy** using the provided evaluation scripts

## 1. Visualizing Motion Data

The `visualize_retarget.py` script allows you to visualize the retargeted motion data in the JSON format files (located in the `data/` directory).

```bash
python visualize_retarget.py --file data/canter.txt --time 20 --speed 0.8
```

Parameters:
- `--file`: Path to the motion file (e.g., `data/canter.txt`, `data/pace.txt`)
- `--time`: Duration of visualization in seconds (default: 20.0)
- `--speed`: Playback speed multiplier (default: 1.0)

For more details, see [README_visualization.md](README_visualization.md).

## 2. Converting Motion Data for Training

The `convert_motion_for_training.py` script converts the JSON format motion files (.txt) to the NumPy (.npy) or PyTorch (.pt) format required for training.

```bash
python convert_motion_for_training.py --file data/canter.txt
```

Parameters:
- `--file`: Path to the input motion file (.txt)
- `--output`: Path to save the output file (.npy or .pt), defaults to the same name with .npy extension
- `--format`: Output format, either "numpy" (default) or "torch"
- `--no-remap`: Skip joint remapping (not recommended)

The script automatically handles the remapping of joint angles between the visualization and training format. The visualization format uses a different joint order than the training format, so this remapping is essential.

## 2. Training a Policy

Once you have converted the motion data, you can train a policy using the `train.py` script:

```bash
python train.py --file data/canter.npy --envs 128 --iters 1500
```

Parameters:
- `--file`: Path to the converted motion file (.npy or .pt)
- `--envs`: Number of parallel training environments (default: 256)
- `--iters`: Maximum number of training iterations (default: 1000)
- `--viz`: Enable visualization during training
- `--no-wandb`: Disable Weights & Biases logging
- `--wandb-project`: W&B project name (default: go2-motion-imitation)

The training process will create a log directory with the experiment name (derived from the motion file name) containing:
- Training logs
- Model checkpoints
- TensorBoard events

## 3. Enhanced Training with Paper-Style Rewards

For more advanced training that better matches the original motion imitation paper approach, you can use the `train_experiment.py` script. This version focuses specifically on three key reward components: joint matching, velocity profile, and end-effector positions, while maintaining domain randomization for robustness.

```bash
python train_experiment.py --file data/canter.npy --envs 128 --iters 2000
```

Parameters:
- `--file`: Path to the converted motion file (.npy or .pt)
- `--envs`: Number of parallel training environments (default: 256)
- `--iters`: Maximum number of training iterations (default: 1000)
- `--viz`: Enable visualization during training
- `--no-wandb`: Disable Weights & Biases logging
- `--wandb-project`: W&B project name (default: go2-motion-imitation-enhanced)
- `--no-domain-rand`: Disable domain randomization
- `--no-curriculum`: Disable curriculum learning

### Key Differences from Standard Training

The experimental training script provides several advantages:

1. **Focused Reward Structure**: Prioritizes three key components from the original paper:
   - Joint pose matching (weight: 6.0)
   - Velocity profile matching (weight: 4.0) 
   - End-effector position matching (weight: 5.0)

2. **Domain Randomization**: Applies randomization to physics parameters (friction, gravity) for more robust policies

3. **Curriculum Learning**: Progressively increases difficulty with terrain roughness and external forces

4. **Better Stability**: Includes ground contact rewards (weight: 2.0) and forward motion rewards (weight: 3.0) for improved stability

### When to Use Train Experiment

Use `train_experiment.py` when:
- You want more accurate motion reproduction that closely follows joint trajectories
- You need more robust policies that can handle environmental variations
- Your motion capture data is well-formed and accurately reflects the desired behavior

The experimental script generally produces more natural-looking gaits with better motion tracking at the cost of slightly longer training times.

## 4. Joint Order

It's important to understand the joint ordering used in the different formats:

### Visualization Format (in JSON .txt files)
```
0: FL_hip_joint
1: FL_thigh_joint
2: FL_calf_joint
3: FR_hip_joint
4: FR_thigh_joint
5: FR_calf_joint
6: RL_hip_joint
7: RL_thigh_joint
8: RL_calf_joint
9: RR_hip_joint
10: RR_thigh_joint
11: RR_calf_joint
```

### Training Format (in .npy/.pt files)
```
0: FR_hip_joint
1: FR_thigh_joint
2: FR_calf_joint
3: FL_hip_joint
4: FL_thigh_joint
5: FL_calf_joint
6: RR_hip_joint
7: RR_thigh_joint
8: RR_calf_joint
9: RL_hip_joint
10: RL_thigh_joint
11: RL_calf_joint
```

The conversion script handles this remapping automatically.

## Naming Convention

- FR: Front Right
- FL: Front Left
- RR: Rear Right
- RL: Rear Left

For each leg, there are three joints:
- hip_joint: The joint connecting the leg to the body
- thigh_joint: The middle joint of the leg
- calf_joint: The lowest joint of the leg 

## 5. Evaluating Trained Policies

After training is complete, you can evaluate your trained policies using the provided evaluation scripts.

> **Important Note on Paths**: When specifying model paths, ensure they're relative to your current working directory. The examples below assume you're running commands from the `rl_train` directory. If you encounter `FileNotFoundError`, verify your paths are correct.

### Genesis Simulator Evaluation

Use the `go2_eval.py` script to evaluate your trained policy in the Genesis simulator:

```bash
# Automatically load the latest checkpoint from logs
python ../policy_test/go2_eval.py

# Evaluate a policy trained with train_experiment.py
python ../policy_test/go2_eval.py --motion pace --experiment_model

# Specify custom evaluation duration
python ../policy_test/go2_eval.py --motion pace --duration 60

# Manually specify a model file (from rl_train directory)
python ../policy_test/go2_eval.py --model logs/go2-imitate-canter/model_499.pt

# Evaluate an experimental model
python ../policy_test/go2_eval.py --model logs/go2-paper-rewards-canter/model_499.pt
```

The `go2_eval.py` script will automatically:
1. Find the latest checkpoint from the specified motion type's log directory
2. Load the policy and create the environment with visualization
3. Run the policy in the environment for the specified duration

### MuJoCo Simulator Evaluation

For an alternative visualization using the MuJoCo simulator, use the `go2_mujoco.py` script:

```bash
# Automatically use model_999.pt in the script directory
python ../policy_test/go2_mujoco.py

# Specify a custom model path (from rl_train directory)
python ../policy_test/go2_mujoco.py --model logs/go2-imitate-pace/model_499.pt

# Evaluate an experimental model
python ../policy_test/go2_mujoco.py --model logs/go2-paper-rewards-pace/model_499.pt

# Adjust the forward velocity command (default is 0.5 m/s)
python ../policy_test/go2_mujoco.py --command 0.8
```

This provides a useful comparison of how the policy performs in different simulation environments. 