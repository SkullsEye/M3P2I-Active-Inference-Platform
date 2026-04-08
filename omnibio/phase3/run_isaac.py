from isaacsim import SimulationApp

# Start the simulation context (must be done before other imports)
simulation_app = SimulationApp({"headless": False})

import sys
import os
import numpy as np

# Add parent directory to path to find 'common'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.behavior_tree import Blackboard
from common.active_inference_planner import ActiveInferencePlanner
from common.cost_planner import CostPlanner, distance_to_goal_cost, obstacle_avoidance_cost, control_effort_cost

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualSphere, FixedCylinder
from omni.isaac.core.utils.types import ArticulationAction

def main():
    # 1. Initialize World
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Lighting
    # Add a distant light to illuminate the scene
    from omni.isaac.core.prims import XFormPrim
    from pxr import UsdLux, Sdf, Gf
    
    stage = world.stage
    light_prim_path = "/World/DistantLight"
    light_prim = UsdLux.DistantLight.Define(stage, light_prim_path)
    light_prim.CreateIntensityAttr(3000)  # Brightness
    light_prim.CreateAngleAttr(0.53)      # Sun angle (for shadows)
    
    # Orient the light
    xform = XFormPrim(light_prim_path)
    xform.set_world_pose(orientation=np.array([0.5, 0.5, 0.5, 0.5])) # Simple rotation

    # 2. Setup Task Elements
    initial_state = np.array([0.0, 0.0, 0.5]) # x, y, z
    goal_pos = np.array([10.0, 10.0, 0.5])
    
    # Robot (represented as a Cube for now)
    robot = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Robot",
            name="robot",
            position=initial_state,
            scale=np.array([0.5, 0.5, 0.5]),
            color=np.array([0.0, 0.0, 1.0]), # Blue
            mass=1.0
        )
    )

    # Goal (Visual Sphere)
    world.scene.add(
        VisualSphere(
            prim_path="/World/Goal",
            name="goal",
            position=goal_pos,
            radius=0.5,
            color=np.array([1.0, 0.0, 0.0]) # Red
        )
    )

    # Obstacles
    obstacles_2d = [np.array([5.0, 5.0]), np.array([8.0, 2.0]), np.array([3.0, 7.0])]
    for i, obs in enumerate(obstacles_2d):
        world.scene.add(
            FixedCylinder(
                prim_path=f"/World/Obstacle_{i}",
                name=f"obstacle_{i}",
                position=np.array([obs[0], obs[1], 0.5]),
                radius=0.5,
                height=1.0,
                color=np.array([1.0, 0.0, 0.0]) # Red obstacles
            )
        )

    # 3. Setup Planner
    # Planner expects 2D arrays, we'll map 3D world to 2D planner
    planner = ActiveInferencePlanner(
        goal=goal_pos[:2], 
        initial_state=initial_state[:2], 
        num_rollouts=50, 
        noise_level=1.5
    )
    
    cost_functions = {
        'distance': distance_to_goal_cost,
        'obstacle': lambda s, a, g: obstacle_avoidance_cost(s, a, g, obstacles_2d),
        'control': control_effort_cost
    }
    cost_planner = CostPlanner(cost_functions=list(cost_functions.values()))

    # 4. Run Simulation
    world.reset()
    
    while simulation_app.is_running():
        # Physics Step
        world.step(render=True)
        if not world.is_playing():
            continue

        # Get current state (Ground Truth for this simple test)
        # Note: In a real scenario, use sensors or robot.get_world_pose()
        current_pose, _ = robot.get_world_pose()
        current_state_2d = current_pose[:2]
        
        # Planner Step
        # The existing planner code returns 'rollouts' (list of actions)
        rollouts = planner.step(current_state_2d)
        
        # Evaluate costs to find best action
        costs = []
        for action in rollouts:
            # Predict next state (simple kinematic model: next = curr + action * dt)
            # Assuming dt=0.5 from original code
            next_state_est = current_state_2d + action * 0.5
            costs.append(cost_planner.calculate_total_cost(next_state_est, action, goal_pos[:2]))
            
        best_idx = np.argmin(costs)
        best_action = rollouts[best_idx]
        
        # Apply Action
        # Since 'robot' is a DynamicCuboid, we can apply velocity
        # Mapping 2D action (velocity) to 3D velocity [vx, vy, 0]
        velocity_command = np.array([best_action[0], best_action[1], 0.0])
        
        # Scale for simulation speed (planner assumed steps, physics runs @ 60hz)
        # We might need to tune this gain
        robot.set_linear_velocity(velocity_command * 5.0) 
        
        # Update planner internal belief
        planner.update_beliefs(current_state_2d)

        # Goal Check
        if np.linalg.norm(current_state_2d - goal_pos[:2]) < 0.5:
            print("Goal Reached!")
            # Respawn or Stop? Let's just pause.
            world.pause()

    simulation_app.close()

if __name__ == "__main__":
    main()
