import mujoco
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import time
from mujoco import viewer

class ActorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Match exactly the layer structure from genesis_motion_imitation.py
        self.network = nn.Sequential(
            nn.Linear(45, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12)
        )
        
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
        self.model.opt.iterations = 20  # Increase solver iterations
        self.model.opt.tolerance = 1e-10  # Decrease solver tolerance
        
        self.data = mujoco.MjData(self.model)
        
        # Initialize actor network
        self.actor = ActorMLP()
        
        # Load model weights
        self.load_checkpoint(model_path)
        
        # Initialize observation scaling factors from genesis_motion_imitation.py
        self.obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }

        # Apply PD control gains as in genesis_motion_imitation.py
        # Increase kp slightly to match Genesis stiffness
        self.kp = 30.0  # Increased from 20.0
        self.kd = 1.0   # Increased from 0.5
        
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
        
        # Apply initial reset to get robot in standing position
        self.reset()
        
    def load_checkpoint(self, model_path):
        """Load checkpoint with much better debugging."""
        try:
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # Check if we need to convert from PPO's naming scheme
                if any('actor.' in k for k in state_dict.keys()):
                    print("Found 'actor.' prefix in state dict, renaming keys to match our model...")
                    # Map from the standard PPO format to our network format
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if 'actor.' in k and not 'std' in k:
                            new_key = k.replace('actor.', 'network.')
                            new_state_dict[new_key] = v
                    
                    # Load the converted state dict
                    if new_state_dict:
                        print(f"Loading {len(new_state_dict)} converted parameters")
                        try:
                            self.actor.load_state_dict(new_state_dict)
                            print("Successfully loaded model with converted layer names")
                        except Exception as e:
                            print(f"Error loading converted state dict: {e}")
                    else:
                        print("No valid keys found after conversion")
                else:
                    print("No 'actor.' prefix found, loading directly...")
                    try:
                        self.actor.load_state_dict(state_dict)
                        print("Successfully loaded model directly")
                    except Exception as e:
                        print(f"Error loading direct state dict: {e}")
            else:
                print("No 'model_state_dict' key found in checkpoint")
                if len(checkpoint.keys()) == 1 and list(checkpoint.keys())[0] == 'state_dict':
                    # Try standard PyTorch format
                    print("Trying standard PyTorch state_dict format...")
                    try:
                        self.actor.load_state_dict(checkpoint['state_dict'])
                        print("Successfully loaded from standard state_dict")
                    except Exception as e:
                        print(f"Error loading from standard state_dict: {e}")
                else:
                    print("Unknown checkpoint format")
        except Exception as e:
            print(f"Error loading model: {e}")
            
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
            print(f"  base_ang_vel: {base_ang_vel}")
            print(f"  projected_gravity: {projected_gravity}")
            print(f"  commands: {self.commands}")
            print(f"  dof_pos shape: {dof_pos.shape}")
            print(f"  dof_vel shape: {dof_vel.shape}")
            print(f"  last_actions shape: {self.last_actions.shape}")
            print(f"  Total observation shape: {obs.shape}")
            self.first_observation = False
        
        return torch.FloatTensor(obs)
    
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
        
        # Clip torques to avoid instability
        torques = np.clip(torques, -100.0, 100.0)
        
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
        
        # Reset to default pose for actuated joints
        for i, name in enumerate(self.actuated_joint_names):
            idx = self.actuated_joint_qpos_idxs[i]
            self.data.qpos[idx] = self.default_dof_pos[i]
        
        # Set the robot to a better initial height to avoid falling through floor
        self.data.qpos[2] = 0.42  # Set initial height
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
        
        # Perform a few stabilization steps to ensure robot doesn't fall immediately
        self._stabilize_standing_pose()
        
        return self.get_observation()
    
    def _stabilize_standing_pose(self, n_steps=100):
        """Apply a few steps of stabilization control to get the robot in a stable standing pose"""
        # Small stabilization control to get robot in stable position
        stabilization_actions = np.zeros(12)
        
        # Apply PD controller to maintain default joint angles
        for _ in range(n_steps):
            # Apply stabilization control
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
            
        print("Robot stabilized in standing position")
    
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Go2 robot in MuJoCo with trained policy")
    parser.add_argument("--model", type=str, default="model_999.pt", help="Path to model checkpoint")
    parser.add_argument("--pattern", action="store_true", help="Use predefined motion pattern")
    parser.add_argument("--info", action="store_true", help="Print model information and exit")
    parser.add_argument("--command", type=float, default=0.5, help="Forward velocity command")
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)
    
    # Initialize robot and environment
    robot = Go2Robot(model_path=model_path)
    
    # Update command if specified
    robot.commands[0] = args.command
    
    # Print model information if requested
    if args.info:
        print("Model information display is disabled in this version.")
        return
    
    # Main control loop
    obs = robot.reset()
    last_time = time.time()
    
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
                action = robot.actor(obs).numpy()
                    
            # Execute action 
            obs = robot.step(action)
            
            # Render
            robot.render()
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        if robot.viewer is not None:
            robot.viewer.close()

if __name__ == "__main__":
    main() 