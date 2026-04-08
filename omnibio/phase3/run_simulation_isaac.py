# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the Active Inference Planner in Isaac Lab.
It loads a simple environment with a ground plane, obstacles, and a moving agent.
The agent is controlled by the custom Active Inference Planner logic.
"""

import argparse
import numpy as np
import os
import sys

# Append parent directory for common modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isaaclab.app import AppLauncher

# Create the argument parser
parser = argparse.ArgumentParser(description="Active Inference Planner in Isaac Lab")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
args_cli = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# Import Isaac Lab libraries
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationContext

# Import common modules
from common.behavior_tree import Blackboard, Status
from common.cost_planner import CostPlanner, distance_to_goal_cost, obstacle_avoidance_cost, control_effort_cost
from common.active_inference_planner import ActiveInferencePlanner

@configclass
class AgentSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with the agent and obstacles."""

    # Ground Plane
    ground = sim_utils.GroundPlaneCfg()

    # Lights
    dome_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))

    # Agent (Simple Sphere)
    agent = RigidObjectCfg(
        prim_path="/World/Agent",
        spawn=sim_utils.SphereCfg(radius=0.5, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    # Obstacles (Fixed Spheres)
    # We will spawn them manually or define them here if static.
    # For dynamic spawning, we can use prim utils, but defining config is cleaner.
    obstacle_1 = RigidObjectCfg(
        prim_path="/World/Obstacles/Obs1",
        spawn=sim_utils.SphereCfg(radius=0.5, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 5.0, 0.5)),
    )
    obstacle_2 = RigidObjectCfg(
        prim_path="/World/Obstacles/Obs2",
        spawn=sim_utils.SphereCfg(radius=0.5, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 2.0, 0.5)),
    )
    obstacle_3 = RigidObjectCfg(
        prim_path="/World/Obstacles/Obs3",
        spawn=sim_utils.SphereCfg(radius=0.5, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 7.0, 0.5)),
    )

    # Goal Marker (Visual only)
    goal_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/Goal",
        markers={
            "goal": sim_utils.SphereCfg(radius=0.3, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
        },
    )


def main():
    """Main function to run the simulation."""
    
    # Initialize simulation context
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    
    # Setup scene
    scene_cfg = AgentSceneCfg(num_envs=1, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    
    # --- Planner Setup ---
    initial_state = np.array([0.0, 0.0]) # 2D state (x, y)
    goal = np.array([10.0, 10.0])
    obstacles = [np.array([5.0, 5.0]), np.array([8.0, 2.0]), np.array([3.0, 7.0])]
    
    ai_planner = ActiveInferencePlanner(goal=goal, initial_state=initial_state, num_rollouts=50, noise_level=1.5)
    
    # Cost functions
    cost_functions = {
        'distance': distance_to_goal_cost,
        'obstacle': lambda state, action, goal: obstacle_avoidance_cost(state, action, goal, obstacles),
        'control': control_effort_cost
    }
    cost_planner = CostPlanner(cost_functions=list(cost_functions.values()))
    
    # Set goal marker position
    # VisualizationMarkers usually take torch tensors
    goal_pos_3d = torch.tensor([[goal[0], goal[1], 0.5]], device=sim.device)
    scene["goal_marker"].visualize(translations=goal_pos_3d)

    print("[INFO] Starting simulation loop...")
    
    while simulation_app.is_running():
        # Step simulation
        sim.step()
        
        # Update scene buffers
        scene.update(dt=sim.get_physics_dt())

        # Get agent state
        # Agent is a RigidObject, data is in scene["agent"].data
        # Position: (num_envs, 3)
        agent_pos_3d = scene["agent"].data.root_pos_w[0].cpu().numpy()
        agent_pos_2d = agent_pos_3d[:2] # Extract (x, y)
        
        # Check if goal reached
        if np.linalg.norm(agent_pos_2d - goal) < 0.5:
            print("[INFO] Goal reached!")
            # Reset agent to start
            root_state = scene["agent"].data.default_root_state.clone()
            scene["agent"].write_root_state_to_sim(root_state)
            scene["agent"].reset()
            sim.reset() # Optional: full reset
            continue

        # --- Active Inference Step ---
        # 1. Update beliefs (in this simple case, belief = current state)
        ai_planner.update_beliefs(agent_pos_2d)
        
        # 2. Generate rollouts
        rollouts = ai_planner.step(agent_pos_2d)
        
        # 3. Evaluate rollouts
        costs = []
        for action in rollouts:
            # Predict next state (simple Euler integration for planning)
            simulated_next_state = agent_pos_2d + action * 0.5
            costs.append(cost_planner.calculate_total_cost(simulated_next_state, action, goal))
        
        # 4. Select best action
        best_action_index = np.argmin(costs)
        best_action = rollouts[best_action_index]
        
        # 5. Apply action to agent
        # We control velocity directly for simplicity (kinematic control)
        # Velocity in 3D: (vx, vy, 0)
        velocity_command = np.array([best_action[0], best_action[1], 0.0]) * 2.0 # Scale for speed
        
        # Set velocity (requires writing to sim state)
        # Note: For physics-based control, we should apply forces. 
        # But for "simulating a code", setting velocity is fine for kinematic agents.
        # RigidObject data.root_vel_w is (num_envs, 6) [lin_vel, ang_vel]
        
        # We need to construct the full velocity tensor
        vel_tensor = torch.zeros((1, 6), device=sim.device)
        vel_tensor[0, 0] = velocity_command[0]
        vel_tensor[0, 1] = velocity_command[1]
        
        # Apply directly to root velocity
        scene["agent"].write_root_velocity_to_sim(vel_tensor)

    print("[INFO] Simulation finished.")
    simulation_app.close()

if __name__ == "__main__":
    main()
