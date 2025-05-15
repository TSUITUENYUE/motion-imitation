import mujoco
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import time
from mujoco import viewer
import re
import glob
import pickle

class ActorMLP(nn.Module):
    def __init__(self, hidden_dims=None):
        super().__init__()
        # Default architecture from genesis_motion_imitation.py
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
            
        # Match exactly the layer structure from genesis_motion_imitation.py
        layers = []
        layers.append(nn.Linear(45, hidden_dims[0]))
        layers.append(nn.ELU())
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(hidden_dims[-1], 12))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class Go2Robot:
    def __init__(self, model_path="model_999.pt"):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load MuJoCo model - use the official model from MuJoCo Menagerie
        scene_path = os.path.join(script_dir, "scene.xml")
        print(f"Loading scene from: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        
        # Modify simulation parameters to increase stability and match Genesis better
        self.model.opt.timestep = 0.002  # Default MuJoCo timestep
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON  # More stable solver
        self.model.opt.iterations = 50  # Increase solver iterations substantially (was 20)
        self.model.opt.tolerance = 1e-10  # Decrease solver tolerance
        self.model.opt.gravity[2] = -9.81  # Ensure correct gravity
        
        # Increase joint and contact stiffness
        for i in range(self.model.njnt):
            # Increase joint stiffness if parameters exist
            if hasattr(self.model, 'jnt_stiffness'):
                self.model.jnt_stiffness[i] = 40.0  # Reduced from 100.0
            
        # Try to set contact parameters if available
        if hasattr(self.model, 'geom_solref'):
            for i in range(self.model.ngeom):
                # Make contacts more stiff
                self.model.geom_solref[i, 0] = 0.002  # Time constant for contacts (slightly increased)
                
        self.data = mujoco.MjData(self.model)
        
        # Try to load config from model directory to get the correct architecture
        self.policy_cfg = self._extract_model_config(model_path)
        
        # Initialize actor network with the correct architecture
        hidden_dims = self.policy_cfg.get("actor_hidden_dims", [512, 256, 128])
        self.actor = ActorMLP(hidden_dims=hidden_dims)
        print(f"Created actor network with hidden dimensions: {hidden_dims}")
        
        # Load model weights
        self.load_checkpoint(model_path)
        
        # Initialize observation scaling factors from genesis_motion_imitation.py
        self.obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }

        # Apply PD control gains - moderate values that allow movement
        self.kp = 80.0  # Reduced from 500.0 to allow movement
        self.kd = 3.0   # Reduced from 20.0 to allow movement
        
        # Action scaling factor (from genesis_motion_imitation.py)
        self.action_scale = 0.25
        
        # Map joint names to indices for easy access
        self.joint_ids = {}
        self.actuator_ids = {}
        
        # Mapping from joint names to actuator names (in the MuJoCo model)
        self.joint_to_actuator = {
            "FR_hip_joint": "FR_hip",
            "FR_thigh_joint": "FR_thigh",
            "FR_calf_joint": "FR_calf",
            "FL_hip_joint": "FL_hip",
            "FL_thigh_joint": "FL_thigh",
            "FL_calf_joint": "FL_calf",
            "RR_hip_joint": "RR_hip",
            "RR_thigh_joint": "RR_thigh",
            "RR_calf_joint": "RR_calf",
            "RL_hip_joint": "RL_hip",
            "RL_thigh_joint": "RL_thigh",
            "RL_calf_joint": "RL_calf"
        }
        
        # Map all joints and actuators
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            self.joint_ids[name] = i
            
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            self.actuator_ids[name] = i
            
        # Use joint_names from genesis_motion_imitation.py to ensure correct order
        # This is critical for policy outputs to match the right joint!
        self.actuated_joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
        
        # Get the positions in qpos for these joints
        self.actuated_joint_qpos_idxs = [self.model.jnt_qposadr[self.joint_ids[name]] for name in self.actuated_joint_names]
        
        # Get the velocity indices in qvel for these joints
        self.actuated_joint_qvel_idxs = [self.model.jnt_dofadr[self.joint_ids[name]] for name in self.actuated_joint_names]
        
        # Default joint angles from genesis_motion_imitation.py
        self.default_dof_pos = np.zeros(12)
        for i, name in enumerate(self.actuated_joint_names):
            if "hip" in name:
                self.default_dof_pos[i] = 0.0
            elif "thigh" in name:
                if "FL" in name or "FR" in name:
                    self.default_dof_pos[i] = 0.8
                else:
                    self.default_dof_pos[i] = 1.0
            elif "calf" in name:
                self.default_dof_pos[i] = -1.5
        
        # Apply a fixed timestep
        self.dt = 0.02  # Match Genesis timestep (50Hz)
        
        # Substeps to match dt
        self.n_substeps = int(self.dt / self.model.opt.timestep)
        
        # Enable action latency simulation like in Genesis
        self.simulate_action_latency = True
        
        # Commands with proper scaling (match Genesis defaults)
        self.commands = np.array([0.5, 0.0, 0.0])  # Default forward command
        self.commands_scale = np.array([
            self.obs_scales["lin_vel"],
            self.obs_scales["lin_vel"], 
            self.obs_scales["ang_vel"]
        ])
        
        # Store actions for latency simulation - match Genesis implementation
        self.last_actions = np.zeros(12)
        self.current_actions = np.zeros(12)
        
        # Store joint positions and velocities for PD control
        self.last_dof_pos = np.zeros(12)
        self.last_dof_vel = np.zeros(12)
        
        # Initialize viewer
        self.viewer = None
        
        # Flag to print observation info on first step
        self.first_observation = True
        
        # Debug information
        self.debug_mode = True
        self.debug_steps = 0
        
        # Stability tracking
        self.stabilization_complete = False
        
        # Apply initial reset to get robot in standing position
        self.reset()
    
    def _extract_model_config(self, model_path):
        """Extract policy configuration from model directory's cfg.pkl if available."""
        policy_cfg = {"actor_hidden_dims": [512, 256, 128], "activation": "elu"}
        
        try:
            # Try to find cfg.pkl in the same directory as the model
            model_dir = os.path.dirname(os.path.abspath(model_path))
            cfg_path = os.path.join(model_dir, "cfgs.pkl")
            
            if os.path.exists(cfg_path):
                print(f"Loading config from {cfg_path}")
                with open(cfg_path, 'rb') as f:
                    configs = pickle.load(f)
                    
                    # The structure of cfgs.pkl is a list: [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg]
                    if len(configs) == 5:
                        _, _, _, _, train_cfg = configs
                        if "policy" in train_cfg:
                            policy_cfg = train_cfg["policy"]
                            print(f"Loaded policy config: {policy_cfg}")
                        else:
                            print("Warning: train_cfg does not contain policy config")
                    else:
                        print(f"Warning: cfgs.pkl has unexpected format with {len(configs)} items")
            else:
                print(f"No config file found at {cfg_path}, using default architecture")
                
                # Try to infer from model name for models in standard locations
                model_name = os.path.basename(model_path)
                if model_name.startswith("model_"):
                    # Look for experiment name in the parent directory name
                    parent_dir = os.path.basename(model_dir)
                    if parent_dir.startswith("go2-"):
                        # This looks like a standard model directory structure
                        print(f"Detected experiment: {parent_dir}")
                        
                        # If it's a paper rewards model, use that architecture
                        if "paper-rewards" in parent_dir or "enhanced" in parent_dir:
                            policy_cfg = {
                                "actor_hidden_dims": [512, 256, 128], 
                                "activation": "elu"
                            }
                            print("Using enhanced/paper rewards architecture")
                            
        except Exception as e:
            print(f"Error extracting policy config: {e}")
            print("Using default policy configuration")
            
        return policy_cfg
        
    def load_checkpoint(self, model_path):
        """Load checkpoint with much better debugging."""
        try:
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Track if loading succeeded
            model_loaded = False
            
            # Case 1: model_state_dict format (most common)
            if 'model_state_dict' in checkpoint:
                print("Found 'model_state_dict' format")
                state_dict = checkpoint['model_state_dict']
                
                # Handle 'actor.' prefix in state dict
                if any('actor.' in k for k in state_dict.keys()):
                    print("Found 'actor.' prefix in state dict, converting keys...")
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if 'actor.' in k and not 'std' in k:
                            new_key = k.replace('actor.', 'network.')
                            new_state_dict[new_key] = v
                            
                    # Try loading the converted state dict
                    try:
                        self.actor.load_state_dict(new_state_dict)
                        print(f"Successfully loaded model with converted keys ({len(new_state_dict)} parameters)")
                        model_loaded = True
                    except Exception as e:
                        print(f"Error loading converted state dict: {e}")
                else:
                    # Try direct loading
                    try:
                        self.actor.load_state_dict(state_dict)
                        print("Successfully loaded model state_dict directly")
                        model_loaded = True
                    except Exception as e:
                        print(f"Error loading direct state dict: {e}")
            
            # Case 2: Direct state_dict format
            elif isinstance(checkpoint, dict) and not ('state_dict' in checkpoint and len(checkpoint) == 1):
                print("Trying direct state dict format...")
                actor_keys = {}
                network_keys = {}
                
                # Check for any possible key patterns
                for k in checkpoint.keys():
                    if k.startswith('actor.'):
                        actor_keys[k] = k.replace('actor.', 'network.')
                    elif k.startswith('network.'):
                        network_keys[k] = k
                
                # Try loading with actor. prefix converted to network.
                if actor_keys:
                    print(f"Found {len(actor_keys)} keys with 'actor.' prefix")
                    new_state_dict = {}
                    for old_key, new_key in actor_keys.items():
                        if 'std' not in old_key:  # Skip std parameters
                            new_state_dict[new_key] = checkpoint[old_key]
                    
                    try:
                        self.actor.load_state_dict(new_state_dict)
                        print(f"Successfully loaded model with converted actor prefix ({len(new_state_dict)} parameters)")
                        model_loaded = True
                    except Exception as e:
                        print(f"Error loading with actor prefix conversion: {e}")
                
                # Try loading with network. prefix directly
                elif network_keys:
                    print(f"Found {len(network_keys)} keys with 'network.' prefix")
                    new_state_dict = {k: checkpoint[k] for k in network_keys if 'std' not in k}
                    
                    try:
                        self.actor.load_state_dict(new_state_dict)
                        print(f"Successfully loaded model with network prefix ({len(new_state_dict)} parameters)")
                        model_loaded = True
                    except Exception as e:
                        print(f"Error loading with network prefix: {e}")
                
                # Try loading the full state dict directly
                if not model_loaded:
                    try:
                        self.actor.load_state_dict(checkpoint)
                        print("Successfully loaded full state dict directly")
                        model_loaded = True
                    except Exception as e:
                        print(f"Error loading full state dict directly: {e}")
            
            # Case 3: Standard PyTorch state_dict format
            elif 'state_dict' in checkpoint and len(checkpoint) == 1:
                print("Found standard PyTorch state_dict format")
                try:
                    self.actor.load_state_dict(checkpoint['state_dict'])
                    print("Successfully loaded from standard state_dict")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading from standard state_dict: {e}")
            
            # If nothing worked, try one more approach with ActorCritic models
            if not model_loaded:
                print("Attempting to extract weights from ActorCritic model structure...")
                # Try finding actor in the state_dict by examining keys
                actor_state_dict = {}
                found_actor_weights = False
                
                if isinstance(checkpoint, dict):
                    for key in checkpoint.keys():
                        # Look for actor-related keys
                        if 'actor' in key and 'std' not in key:
                            # Extract the layer part after 'actor.'
                            if '.' in key:
                                parts = key.split('.')
                                if len(parts) >= 2:
                                    # Construct the corresponding key for our model
                                    new_key = 'network.' + '.'.join(parts[1:])
                                    actor_state_dict[new_key] = checkpoint[key]
                                    found_actor_weights = True
                
                if found_actor_weights:
                    try:
                        self.actor.load_state_dict(actor_state_dict)
                        print(f"Successfully extracted and loaded actor weights ({len(actor_state_dict)} parameters)")
                        model_loaded = True
                    except Exception as e:
                        print(f"Error loading extracted actor weights: {e}")
            
            # Final fallback - print model architecture and state dict keys for debugging
            if not model_loaded:
                print("\nFailed to load model. Printing model architecture and checkpoint keys for debugging:")
                print("\nModel architecture:")
                print(self.actor)
                
                print("\nCheckpoint keys:")
                if isinstance(checkpoint, dict):
                    for k in checkpoint.keys():
                        print(f"  {k}")
                        
                    # If there's an 'actor' key, print its structure
                    if 'actor' in checkpoint:
                        print("\nActor structure:")
                        print(checkpoint['actor'])
                else:
                    print(f"Checkpoint is not a dictionary, type: {type(checkpoint)}")
                
                raise RuntimeError("Failed to load model weights after trying multiple approaches")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
        self.actor.eval()
    
    def get_observation(self):
        """Get the current observation state vector exactly as in genesis_motion_imitation.py"""
        # Get base velocities in base frame
        base_quat = self.data.qpos[3:7].copy()  # w,x,y,z
        base_ang_vel = self.data.qvel[3:6].copy()  # world frame
        base_lin_vel = self.data.qvel[0:3].copy()  # world frame
        
        # Transform velocities to base frame
        def quat_rotate_inverse(q, v):
            q_w = q[0]
            q_vec = q[1:]
            return v + 2 * np.cross(q_vec, np.cross(q_vec, v) + q_w * v)
            
        base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel)
        base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel)
        
        # Get gravity vector in base frame
        gravity = np.array([0.0, 0.0, -1.0])
        projected_gravity = quat_rotate_inverse(base_quat, gravity)
        
        # Get joint positions and velocities - using indexed access for actuated joints
        dof_pos = np.zeros(12)
        dof_vel = np.zeros(12)
        
        for i, idx in enumerate(self.actuated_joint_qpos_idxs):
            dof_pos[i] = self.data.qpos[idx]
            
        for i, idx in enumerate(self.actuated_joint_qvel_idxs):
            dof_vel[i] = self.data.qvel[idx]
        
        # Store current positions and velocities
        self.last_dof_pos = dof_pos.copy()
        self.last_dof_vel = dof_vel.copy()
        
        # Construct observation vector exactly as in genesis_motion_imitation.py step() method
        obs = np.concatenate([
            base_ang_vel * self.obs_scales["ang_vel"],  # 3
            projected_gravity,  # 3
            self.commands * self.commands_scale,  # 3
            (dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            dof_vel * self.obs_scales["dof_vel"],  # 12
            self.last_actions,  # 12
        ])
        
        if self.first_observation:
            print("\nObservation components:")
            print(f"  base_ang_vel: {base_ang_vel} (scaled: {base_ang_vel * self.obs_scales['ang_vel']})")
            print(f"  projected_gravity: {projected_gravity}")
            print(f"  commands: {self.commands} (scaled: {self.commands * self.commands_scale})")
            print(f"  dof_pos: (shape: {dof_pos.shape}, normalized: {(dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos']})")
            print(f"  dof_vel: (shape: {dof_vel.shape}, scaled: {dof_vel * self.obs_scales['dof_vel']})")
            print(f"  last_actions: (shape: {self.last_actions.shape})")
            print(f"  Total observation shape: {obs.shape}")
            self.first_observation = False
            
            # Print quaternion for debugging
            print(f"  Base quaternion: {base_quat}")
        
        # Debug for first few steps
        if self.debug_mode and self.debug_steps < 5:
            self.debug_steps += 1
            print(f"Step {self.debug_steps} - Base lin vel: {base_lin_vel}, Command: {self.commands}")
        
        return torch.FloatTensor(obs)
    
    def _setup_friction_parameters(self):
        """Set up friction parameters to match Genesis."""
        # Try to set friction for all geoms if available
        if hasattr(self.model, 'geom_friction'):
            for i in range(self.model.ngeom):
                # Set higher friction for feet
                geom_name = self.model.geom(i).name
                if 'foot' in geom_name or 'feet' in geom_name:
                    self.model.geom_friction[i, 0] = 2.0  # Increased sliding friction
                    self.model.geom_friction[i, 1] = 0.1  # Increased torsional friction
                    self.model.geom_friction[i, 2] = 0.02  # Increased rolling friction
                else:
                    # Default friction for other geoms
                    self.model.geom_friction[i, 0] = 1.0  # Increased default friction

    def compute_torques(self, action):
        """Apply action through PD controller to compute torques (as in genesis_motion_imitation.py)"""
        # Scale actions
        scaled_action = action * self.action_scale
        
        # Get current joint positions and velocities
        dof_pos = self.last_dof_pos
        dof_vel = self.last_dof_vel
        
        # Compute position targets from scaled actions
        position_targets = scaled_action + self.default_dof_pos
        
        # Calculate torques using PD control (as in Genesis)
        torques = self.kp * (position_targets - dof_pos) - self.kd * dof_vel
        
        # Clip torques to avoid instability - more reasonable limits that don't lock motion
        torques = np.clip(torques, -200.0, 200.0)  # Reduced from 1000 to allow movement
        
        return torques
    
    def step(self, action):
        """Execute one step in the environment"""
        # Store current action
        self.current_actions = action.copy()
        
        # Apply action latency - use last actions if simulating latency, exactly as Genesis does
        exec_action = self.last_actions if self.simulate_action_latency else self.current_actions
        
        # Compute torques from actions using PD control
        torques = self.compute_torques(exec_action)
        
        # Apply torques to the robot actuators
        for i, name in enumerate(self.actuated_joint_names):
            actuator_name = self.joint_to_actuator[name]
            actuator_idx = self.actuator_ids[actuator_name]
            self.data.ctrl[actuator_idx] = torques[i]
        
        # Step the simulation with multiple substeps for better stability
        for _ in range(self.n_substeps):
            # Catch any instability
            try:
                mujoco.mj_step(self.model, self.data)
            except Exception as e:
                print(f"Error during simulation step: {e}")
                # Reset if simulation becomes unstable
                self.reset()
                break
                
            # Check for NaN values
            if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
                print("NaN detected in simulation, resetting")
                self.reset()
                break
            
        # Only update last_actions after successfully stepping the simulation
        self.last_actions = self.current_actions.copy()
        
        # Get new observation
        next_obs = self.get_observation()
        
        return next_obs
    
    def reset(self):
        """Reset the environment"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set up friction parameters
        self._setup_friction_parameters()
        
        # Reset to default pose for actuated joints
        for i, name in enumerate(self.actuated_joint_names):
            idx = self.actuated_joint_qpos_idxs[i]
            self.data.qpos[idx] = self.default_dof_pos[i]
        
        # Set the robot to a better initial height to avoid falling through floor
        self.data.qpos[2] = 0.45  # Set initial height higher (was 0.44)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Set initial orientation (w,x,y,z)
        
        # Reset velocities
        self.data.qvel[:] = 0
        
        # Reset actions
        self.last_actions[:] = 0
        self.current_actions[:] = 0
        
        # Apply forward kinematics to update the model
        mujoco.mj_forward(self.model, self.data)
        
        # Reset debug flag to print observation components once
        self.first_observation = True
        self.debug_steps = 0
        
        # Reset stabilization flag
        self.stabilization_complete = False
        
        # Perform a few stabilization steps to ensure robot doesn't fall immediately
        self._stabilize_standing_pose()
        
        return self.get_observation()
    
    def _stabilize_standing_pose(self, n_steps=100):
        """Apply a few steps of stabilization control to get the robot in a stable standing pose"""
        # Small stabilization control to maintain default joint positions
        stabilization_actions = np.zeros(12)
        
        # Apply PD controller to maintain default joint angles
        for step in range(n_steps):
            # Apply stronger stabilization control
            torques = self.compute_torques(stabilization_actions)
            
            # Apply torques directly 
            for i, name in enumerate(self.actuated_joint_names):
                actuator_name = self.joint_to_actuator[name]
                actuator_idx = self.actuator_ids[actuator_name]
                self.data.ctrl[actuator_idx] = torques[i]
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Get joint positions and velocities
            for i, idx in enumerate(self.actuated_joint_qpos_idxs):
                self.last_dof_pos[i] = self.data.qpos[idx]
                
            for i, idx in enumerate(self.actuated_joint_qvel_idxs):
                self.last_dof_vel[i] = self.data.qvel[idx]
        
        # Final height check
        height = self.data.qpos[2]
        if height < 0.3:
            print(f"Warning: Robot too low at {height:.4f}m, adjusting height...")
            self.data.qpos[2] = 0.45
            mujoco.mj_forward(self.model, self.data)

    def render(self):
        """Render the environment"""
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        
        self.viewer.sync()
        
    def print_model_info(self):
        """Print information about the model to help debug"""
        print(f"Number of bodies: {self.model.nbody}")
        print(f"Number of joints: {self.model.njnt}")
        print(f"Number of DOFs: {self.model.nv}")
        print(f"Number of actuators: {self.model.nu}")
        
        print("\nJoint information:")
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            joint_type = self.model.jnt_type[i]
            joint_type_str = ["free", "ball", "slide", "hinge"][joint_type]
            print(f"  {i}: {name} (type: {joint_type_str})")
            
        print("\nActuator information:")
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            print(f"  {i}: {name}")
            
        print("\nActuated joint mapping:")
        for i, name in enumerate(self.actuated_joint_names):
            qpos_idx = self.actuated_joint_qpos_idxs[i]
            qvel_idx = self.actuated_joint_qvel_idxs[i]
            actuator_name = self.joint_to_actuator[name]
            actuator_idx = self.actuator_ids[actuator_name]
            print(f"  {i}: {name} → {actuator_name} (qpos: {qpos_idx}, qvel: {qvel_idx}, actuator: {actuator_idx})")

def find_latest_model(motion_type=None, experiment_type=None, checkpoint_iter=None, log_dir=None):
    """Find the latest model file based on motion type and experiment type or from a specific directory."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # If specific log directory is provided, use it directly
    if log_dir:
        # Handle both absolute and relative paths
        if not os.path.isabs(log_dir):
            # Special handling for logs/ paths - check both direct logs/ and rl_train/logs/
            if log_dir.startswith("logs/"):
                # Try both rl_train/logs and direct logs folder
                direct_logs_path = os.path.join(project_root, log_dir)
                rl_train_logs_path = os.path.join(project_root, "rl_train", log_dir)
                
                # Check if either path exists
                if os.path.exists(direct_logs_path):
                    log_dir = direct_logs_path
                    print(f"Found directory at: {log_dir}")
                elif os.path.exists(rl_train_logs_path):
                    log_dir = rl_train_logs_path
                    print(f"Found directory at: {log_dir}")
                else:
                    # Will fall through to error handling below
                    log_dir = direct_logs_path
            else:
                # Path relative to script directory
                log_dir = os.path.join(script_dir, log_dir)
        
        if os.path.exists(log_dir):
            print(f"Using specified log directory: {log_dir}")
            
            # If checkpoint iteration is specified, look for that specific model
            if checkpoint_iter is not None:
                specific_model = os.path.join(log_dir, f"model_{checkpoint_iter}.pt")
                if os.path.exists(specific_model):
                    print(f"Using specified checkpoint: model_{checkpoint_iter}.pt")
                    return specific_model
                else:
                    print(f"Warning: Specified checkpoint model_{checkpoint_iter}.pt not found in {log_dir}")
                    print("Looking for nearest available checkpoint...")
                    
                    # Find the nearest available checkpoint
                    model_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
                    if not model_files:
                        print(f"No model files found in {log_dir}")
                        return None
                        
                    # Extract iteration numbers
                    iter_numbers = []
                    for file_path in model_files:
                        match = re.search(r'model_(\d+)\.pt', os.path.basename(file_path))
                        if match:
                            iter_numbers.append((int(match.group(1)), file_path))
                    
                    if not iter_numbers:
                        print(f"No valid model files found in {log_dir}")
                        return None
                        
                    # Find the closest iteration number
                    iter_numbers.sort(key=lambda x: abs(x[0] - checkpoint_iter))
                    nearest_iter, nearest_file = iter_numbers[0]
                    print(f"Found nearest checkpoint: model_{nearest_iter}.pt")
                    return nearest_file
            
            # Find the latest model file in the directory
            model_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
            if not model_files:
                print(f"No model files found in {log_dir}")
                return None
            
            # Extract iteration numbers
            iter_numbers = []
            for file_path in model_files:
                match = re.search(r'model_(\d+)\.pt', os.path.basename(file_path))
                if match:
                    iter_numbers.append((int(match.group(1)), file_path))
            
            if not iter_numbers:
                print(f"No valid model files found in {log_dir}")
                return None
            
            # Get the highest iteration model
            iter_numbers.sort(reverse=True)
            highest_iter, latest_model = iter_numbers[0]
            print(f"Found latest checkpoint: model_{highest_iter}.pt")
            
            return latest_model
        else:
            print(f"Error: Specified log directory not found: {log_dir}")
            # Print all available logs directories for user convenience
            rl_train_logs = os.path.join(project_root, "rl_train", "logs")
            logs_dir = os.path.join(project_root, "logs")
            
            available_dirs = []
            for check_dir in [rl_train_logs, logs_dir]:
                if os.path.exists(check_dir):
                    print(f"\nAvailable directories in {check_dir}:")
                    for item in sorted(os.listdir(check_dir)):
                        if os.path.isdir(os.path.join(check_dir, item)) and item.startswith("go2-"):
                            available_dirs.append(os.path.join(check_dir, item))
                            print(f"  - {item}")
            
            if available_dirs:
                print("\nTry specifying one of these paths with --dir")
                if log_dir.startswith(project_root) and "logs/go2-" in log_dir:
                    corrected_path = log_dir.replace(os.path.join(project_root, "logs"), 
                                                   os.path.join(project_root, "rl_train", "logs"))
                    dir_name = os.path.basename(log_dir)
                    if os.path.exists(corrected_path):
                        print(f"\nTry this path instead:\n  --dir rl_train/logs/{dir_name}")
            return None
    
    # Path to logs directory (automatic search mode)
    logs_dir = os.path.join(project_root, "rl_train", "logs")
    
    # If the rl_train/logs directory doesn't exist, try the direct logs directory
    if not os.path.exists(logs_dir):
        logs_dir = os.path.join(project_root, "logs")
        if not os.path.exists(logs_dir):
            print(f"Error: No logs directory found at {logs_dir} or {os.path.join(project_root, 'rl_train', 'logs')}")
            return None
    
    # Construct search pattern
    search_pattern = "go2-"
    if experiment_type:
        search_pattern += experiment_type + "-"
    if motion_type:
        search_pattern += motion_type
    
    # Search for matching directories
    matching_dirs = []
    for item in os.listdir(logs_dir):
        if item.startswith(search_pattern):
            matching_dirs.append(os.path.join(logs_dir, item))
    
    if not matching_dirs:
        print(f"No matching experiment directories found for pattern: {search_pattern}")
        return None
    
    # Sort directories by modification time (newest first)
    matching_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = matching_dirs[0]
    print(f"Found experiment directory: {os.path.basename(latest_dir)}")
    
    # If checkpoint iteration is specified, look for that specific model
    if checkpoint_iter is not None:
        specific_model = os.path.join(latest_dir, f"model_{checkpoint_iter}.pt")
        if os.path.exists(specific_model):
            print(f"Using specified checkpoint: model_{checkpoint_iter}.pt")
            return specific_model
        else:
            print(f"Warning: Specified checkpoint model_{checkpoint_iter}.pt not found in {latest_dir}")
            print("Looking for nearest available checkpoint...")
            
            # Try to find the nearest available checkpoint
            model_files = glob.glob(os.path.join(latest_dir, "model_*.pt"))
            if not model_files:
                print(f"No model files found in {latest_dir}")
                return None
                
            # Extract iteration numbers
            iter_numbers = []
            for file_path in model_files:
                match = re.search(r'model_(\d+)\.pt', os.path.basename(file_path))
                if match:
                    iter_numbers.append((int(match.group(1)), file_path))
            
            if not iter_numbers:
                print(f"No valid model files found in {latest_dir}")
                return None
                
            # Find the closest iteration number
            iter_numbers.sort(key=lambda x: abs(x[0] - checkpoint_iter))
            nearest_iter, nearest_file = iter_numbers[0]
            print(f"Found nearest checkpoint: model_{nearest_iter}.pt")
            return nearest_file
    
    # Find the latest model file in the directory
    model_files = glob.glob(os.path.join(latest_dir, "model_*.pt"))
    if not model_files:
        print(f"No model files found in {latest_dir}")
        return None
    
    # Extract iteration numbers
    iter_numbers = []
    for file_path in model_files:
        match = re.search(r'model_(\d+)\.pt', os.path.basename(file_path))
        if match:
            iter_numbers.append((int(match.group(1)), file_path))
    
    if not iter_numbers:
        print(f"No valid model files found in {latest_dir}")
        return None
    
    # Get the highest iteration model
    iter_numbers.sort(reverse=True)
    highest_iter, latest_model = iter_numbers[0]
    print(f"Found latest checkpoint: model_{highest_iter}.pt")
    
    return latest_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Go2 robot in MuJoCo with trained policy")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--motion", type=str, default=None, help="Motion type to load (e.g., canter, pace)")
    parser.add_argument("--experiment", type=str, default=None, 
                    help="Experiment type (e.g., enhanced, imitate, paper-rewards)")
    parser.add_argument("--dir", type=str, default=None,
                    help="Exact directory path where checkpoint is located (e.g., logs/go2-enhanced-canter)")
    parser.add_argument("--checkpoint", type=int, default=None, 
                    help="Specific checkpoint iteration to load (e.g., 500 for model_500.pt)")
    parser.add_argument("--pattern", action="store_true", help="Use predefined motion pattern")
    parser.add_argument("--info", action="store_true", help="Print model information and exit")
    parser.add_argument("--command", type=float, default=0.5, help="Forward velocity command")
    parser.add_argument("--stiffness", type=float, default=80.0, 
                    help="PD controller stiffness (kp value, default: 80.0)")
    parser.add_argument("--damping", type=float, default=3.0,
                    help="PD controller damping (kd value, default: 3.0)")
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine model path
    model_path = None
    if args.model:
        # Use provided model path
        model_path = args.model
        # If not an absolute path, make it relative to script directory
        if not os.path.isabs(model_path):
            model_path = os.path.join(script_dir, args.model)
    elif args.dir:
        # Use the specific log directory provided
        model_path = find_latest_model(log_dir=args.dir, checkpoint_iter=args.checkpoint)
    elif args.motion:
        # Find latest model for specified motion type and experiment type
        model_path = find_latest_model(motion_type=args.motion, experiment_type=args.experiment, 
                                      checkpoint_iter=args.checkpoint)
    else:
        # Use default model path
        default_model = os.path.join(script_dir, "model_999.pt")
        if os.path.exists(default_model):
            model_path = default_model
        else:
            # Try to find any model if default doesn't exist
            model_path = find_latest_model(checkpoint_iter=args.checkpoint)
    
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Could not find a valid model file. Please specify with --model or --motion")
        return
        
    print(f"Using model: {model_path}")
    
    # Initialize robot and environment
    robot = Go2Robot(model_path=model_path)
    
    # Update command if specified
    robot.commands[0] = args.command
    
    # Update PD controller parameters if specified
    if args.stiffness != 80.0:
        print(f"Using custom stiffness (kp): {args.stiffness}")
        robot.kp = args.stiffness
    else:
        print(f"Using default stiffness (kp): {robot.kp}")
    
    if args.damping != 3.0:
        print(f"Using custom damping (kd): {args.damping}")
        robot.kd = args.damping
    else:
        print(f"Using default damping (kd): {robot.kd}")
    
    # Print model information if requested
    if args.info:
        robot.print_model_info()
        return
    
    # Main control loop
    obs = robot.reset()
    last_time = time.time()
    
    # For debugging policy behavior
    print(f"Starting control loop with forward command: {robot.commands[0]}")
    print(f"Using PD control with kp={robot.kp}, kd={robot.kd}")
    
    try:
        # Run simulation steps
        while True:
            # Control the rate to avoid consuming too much CPU
            current_time = time.time()
            if current_time - last_time < robot.dt:
                time.sleep(0.001)
                continue
                
            last_time = current_time
            
            # Handle pattern mode
            if args.pattern:
                print("Pattern mode is disabled in this version.")
                return
            
            # Use trained model
            with torch.no_grad():
                try:
                    # Convert observation to tensor
                    if not isinstance(obs, torch.Tensor):
                        obs_tensor = torch.FloatTensor(obs)
                    else:
                        obs_tensor = obs
                        
                    # Get action from policy
                    action = robot.actor(obs_tensor).detach().numpy()
                    
                    # Debug actions for the first few steps
                    if robot.debug_steps < 6:
                        print(f"Action output: min={action.min():.4f}, max={action.max():.4f}, mean={action.mean():.4f}")
                except Exception as e:
                    print(f"Error during policy inference: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                    
            # Execute action 
            try:
                obs = robot.step(action)
            except Exception as e:
                print(f"Error during step: {e}")
                break
            
            # Render
            robot.render()
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        if robot.viewer is not None:
            robot.viewer.close()

if __name__ == "__main__":
    main() 