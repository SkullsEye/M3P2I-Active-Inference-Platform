import numpy as np
import os
import sys

# Add the parent directory to the path to find the 'common' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.behavior_tree import Sequence, Action, Blackboard, Status, plot_behavior_tree
from common.cost_planner import CostPlanner, distance_to_goal_cost, obstacle_avoidance_cost, control_effort_cost, rectangle_obstacle_cost
from common.active_inference_planner import ActiveInferencePlanner
from common import plotting_utils

def main():
    """
    This script runs a planning process and, once complete, generates a
    series of plots and data files for analysis.
    """
    # --- 1. Setup ---
    SAVE_DIR = "../graphs/phase1_run"
    
    # Environment
    initial_state = np.array([0.0, 0.0])
    goal = np.array([10.0, 10.0])
    circ_obstacles = [np.array([5.0, 5.0]), np.array([8.0, 2.0]), np.array([3.0, 7.0])]
    rect_obstacles = [] # Phase 1 does not have interactive rectangles
    
    # Blackboard
    blackboard = Blackboard()
    blackboard.set("state", initial_state)
    
    # History tracking
    path_history = [initial_state.copy()]
    cost_history = []
    
    # --- 2. Planners ---
    ai_planner = ActiveInferencePlanner(goal=goal, initial_state=initial_state, num_rollouts=50, noise_level=1.5)
    
    cost_functions = {
        'distance': distance_to_goal_cost,
        'circ_obstacle': lambda state, action, goal: obstacle_avoidance_cost(state, action, goal, circ_obstacles),
        'rect_obstacle': lambda state, action, goal: rectangle_obstacle_cost(state, action, goal, rect_obstacles),
        'control': control_effort_cost
    }
    cost_planner = CostPlanner(cost_functions=list(cost_functions.values()))

    # --- 3. Behavior Tree ---
    class NavigateWithRollouts(Action):
        def tick(self, blackboard):
            state = blackboard.get("state")
            
            if np.linalg.norm(state - goal) < 0.5:
                print("Goal reached!")
                return Status.SUCCESS
            
            rollouts = ai_planner.step(state)
            costs = [cost_planner.calculate_total_cost(state + action * 0.5, action, goal) for action in rollouts]
            best_action_index = np.argmin(costs)
            best_action = rollouts[best_action_index]
            
            # Store cost history
            dist_cost = cost_functions['distance'](state, best_action, goal)
            circ_obs_cost = cost_functions['circ_obstacle'](state, best_action, goal)
            rect_obs_cost = cost_functions['rect_obstacle'](state, best_action, goal)
            ctrl_cost = cost_functions['control'](state, best_action, goal)
            total_cost = costs[best_action_index]
            cost_history.append([total_cost, dist_cost, circ_obs_cost, rect_obs_cost, ctrl_cost])

            # Apply action
            new_state = state + best_action * 0.5
            blackboard.set("state", new_state)
            path_history.append(new_state.copy())
            ai_planner.update_beliefs(new_state)

            print(f"Step: {len(path_history)-1}, State: {np.round(state, 2)}, Best Cost: {total_cost:.2f}")
            return Status.RUNNING

    robot_behavior = Sequence("NavigateToGoal", [
        NavigateWithRollouts("NavigateWithRollouts")
    ])

    # --- 4. Simulation Loop ---
    print("Starting Phase 1: Planning and Graph Generation...")
    for i in range(100):
        status = robot_behavior.tick(blackboard)
        if status == Status.SUCCESS or status == Status.FAILURE:
            break
    
    # --- 5. Generate Outputs ---
    print(f"\n--- Saving outputs to {SAVE_DIR} ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    plot_behavior_tree(robot_behavior, os.path.join(SAVE_DIR, "behavior_tree.png"))
    plotting_utils.save_simulation_graphs(
        save_dir=SAVE_DIR,
        path_history=path_history,
        cost_history=cost_history,
        goal=goal,
        circ_obstacles=circ_obstacles,
        rect_obstacles=rect_obstacles
    )
    print("\nPhase 1 finished.")

if __name__ == "__main__":
    main()
