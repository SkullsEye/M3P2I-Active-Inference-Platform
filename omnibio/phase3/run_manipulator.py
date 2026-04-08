from isaacsim import SimulationApp

# Start the simulation context
simulation_app = SimulationApp({"headless": False})

import sys
import os
import numpy as np

# Add parent directory to path to find 'common'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.active_inference_planner import ActiveInferencePlanner

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualSphere, FixedCuboid
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.types import ArticulationAction

def main():
    # 1. Initialize World
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Lighting
    from pxr import UsdLux, Gf
    stage = world.stage
    light_prim = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    light_prim.CreateIntensityAttr(3000)
    light_prim.CreateAngleAttr(0.53)

    # 2. Setup Scene
    # Table (centered at X=0.5, Y=0.0)
    # Z=0.4 (center), Scale=0.8 (height) -> Top Surface is at Z=0.8
    table_pos = np.array([0.5, 0.0, 0.4])
    table = world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=table_pos, 
            scale=np.array([0.6, 1.0, 0.8]),
            color=np.array([0.7, 0.7, 0.7])
        )
    )

    # Cube to Pick (Target Object)
    # On top of table (Z=0.8) + Half Cube Height (0.025) = 0.825
    cube_initial_pos = np.array([0.5, 0.0, 0.825]) 
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=cube_initial_pos,
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([0.0, 1.0, 0.0]), # Green
            mass=0.1
        )
    )

    # Robot (Franka Panda)
    # Place robot base on top of table (Z=0.8).
    # Shifted back (-X) so it can reach the cube at X=0.5.
    robot_pos = np.array([-0.1, 0.0, 0.8]) 
    robot = world.scene.add(
        Franka(
            prim_path="/World/Franka",
            name="franka",
            position=robot_pos
        )
    )

    # Target Goal (Where to place the cube)
    # On the table, shifted to the side (+Y)
    goal_pos = np.array([0.5, 0.4, 0.825]) 
    world.scene.add(
        VisualSphere(
            prim_path="/World/TargetGoal",
            name="target_goal",
            position=goal_pos,
            radius=0.03,
            color=np.array([1.0, 0.0, 0.0]), # Red
            visible=True
        )
    )

    world.reset()
    
    # 3. Setup Controller (RMPFlow)
    # RMPFlow handles collision avoidance and smooth motion to a target
    rmp_controller = RMPFlowController(name="target_follower", robot_articulation=robot)
    
    # Task Logic (Simple State Machine)
    task_phase = "approach"
    gripper_timer = 0
    
    # Default orientation (gripper pointing down)
    # Note: RMPflow expects quaternions as [w, x, y, z] usually, but let's check standard Isaac.
    # Usually [0, 1, 0, 0] works for "down" with Franka.
    target_orientation = np.array([0.0, 1.0, 0.0, 0.0]) 

    print("Starting simulation loop...")
    
    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue

        # Get Current End-Effector Position
        ee_pos = robot.end_effector.get_world_pose()[0]
        cube_pos = cube.get_world_pose()[0] # Should be stable or grabbed
        
        # Determine Target based on Phase
        target_pos = None
        
        if task_phase == "approach":
            target_pos = cube_pos + np.array([0, 0, 0.15]) # Hover 15cm above
            if np.linalg.norm(ee_pos - target_pos) < 0.05:
                task_phase = "descend"
                print("Phase: Descending")

        elif task_phase == "descend":
            target_pos = cube_pos + np.array([0, 0, 0.005]) # Touch slightly above center
            if np.linalg.norm(ee_pos - target_pos) < 0.02:
                task_phase = "grasp"
                gripper_timer = 0
                print("Phase: Grasping")
                
        elif task_phase == "grasp":
            target_pos = cube_pos # Stay put
            robot.gripper.close()
            gripper_timer += 1
            if gripper_timer > 60: # Wait 1 sec (at 60fps)
                task_phase = "lift"
                print("Phase: Lifting")

        elif task_phase == "lift":
            target_pos = cube_initial_pos + np.array([0, 0, 0.3]) # Lift high
            if np.linalg.norm(ee_pos - target_pos) < 0.05:
                task_phase = "move_to_goal"
                print("Phase: Moving to Goal")

        elif task_phase == "move_to_goal":
            target_pos = goal_pos + np.array([0, 0, 0.1]) # Hover over goal
            if np.linalg.norm(ee_pos - target_pos) < 0.05:
                task_phase = "place"
                print("Phase: Placing")
                
        elif task_phase == "place":
            target_pos = goal_pos + np.array([0, 0, 0.02]) # Lower
            if np.linalg.norm(ee_pos - target_pos) < 0.02:
                task_phase = "release"
                gripper_timer = 0
                print("Phase: Releasing")
        
        elif task_phase == "release":
            target_pos = goal_pos
            robot.gripper.open()
            gripper_timer += 1
            if gripper_timer > 60:
                task_phase = "done"
                print("Phase: Done")
        
        elif task_phase == "done":
            target_pos = goal_pos + np.array([0, 0, 0.2]) # Retract

        # Execute Control
        if target_pos is not None:
            actions = rmp_controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=target_orientation
            )
            robot.apply_action(actions)

    simulation_app.close()

if __name__ == "__main__":
    main()
