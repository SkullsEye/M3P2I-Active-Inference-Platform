import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

# Add the parent directory to the path to find the 'common' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.behavior_tree import Sequence, Action, Blackboard, Status
from common.cost_planner import CostPlanner, distance_to_goal_cost, obstacle_avoidance_cost, control_effort_cost
from common.active_inference_planner import ActiveInferencePlanner

# --- Simulation Setup ---
# Environment
initial_state = np.array([0.0, 0.0])
goal = np.array([10.0, 10.0])
obstacles = [np.array([5.0, 5.0]), np.array([8.0, 2.0]), np.array([3.0, 7.0])]

# Blackboard
blackboard = Blackboard()
blackboard.set("state", initial_state)
blackboard.set("goal", goal)
blackboard.set("obstacles", obstacles)

# Planners
ai_planner = ActiveInferencePlanner(goal=goal, initial_state=initial_state, num_rollouts=50, noise_level=1.5)
cost_functions = {
    'distance': distance_to_goal_cost,
    'obstacle': lambda state, action, goal: obstacle_avoidance_cost(state, action, goal, obstacles),
    'control': control_effort_cost
}
cost_planner = CostPlanner(cost_functions=list(cost_functions.values()))

# Behavior Tree
class NavigateWithRollouts(Action):
    def tick(self, blackboard):
        state = blackboard.get("state")
        goal = blackboard.get("goal")

        if np.linalg.norm(state - goal) < 0.5:
            print("Goal reached!")
            return Status.SUCCESS
        
        rollouts = ai_planner.step(state)
        
        costs = []
        rollout_ends = []
        for action in rollouts:
            simulated_next_state = state + action * 0.5
            costs.append(cost_planner.calculate_total_cost(simulated_next_state, action, goal))
            rollout_ends.append(simulated_next_state)

        best_action_index = np.argmin(costs)
        best_action = rollouts[best_action_index]
        
        new_state = state + best_action * 0.5
        blackboard.set("state", new_state)
        blackboard.set("current_rollouts", [{'start': state, 'end': end} for end in rollout_ends])
        ai_planner.update_beliefs(new_state)

        return Status.RUNNING

robot_behavior = Sequence("NavigateToGoal", [
    NavigateWithRollouts("NavigateWithRollouts")
])

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 12)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Phase 2: Live Simulation")

# Plot static elements
ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
for obs in obstacles:
    circle = plt.Circle(obs, 0.5, color='r', alpha=0.5)
    ax.add_patch(circle)

# Plot dynamic elements
robot_path, = ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
robot_marker, = ax.plot([], [], 'bo', markersize=8)
rollout_lines = [ax.plot([], [], 'c-', alpha=0.2)[0] for _ in range(ai_planner.num_rollouts)]

path_history = [initial_state.copy()]

def init():
    robot_path.set_data([], [])
    robot_marker.set_data([], [])
    for line in rollout_lines:
        line.set_data([], [])
    ax.legend()
    return [robot_path, robot_marker] + rollout_lines

def update(frame):
    print(f"Step: {frame}")
    status = robot_behavior.tick(blackboard)

    current_state = blackboard.get("state")
    path_history.append(current_state.copy())
    
    # Update robot path
    path_data = np.array(path_history)
    robot_path.set_data(path_data[:, 0], path_data[:, 1])
    robot_marker.set_data([current_state[0]], [current_state[1]])

    # Update rollouts
    current_rollouts = blackboard.get("current_rollouts")
    for i, line in enumerate(rollout_lines):
        if i < len(current_rollouts):
            rollout = current_rollouts[i]
            line.set_data([rollout['start'][0], rollout['end'][0]], [rollout['start'][1], rollout['end'][1]])
        else:
            line.set_data([], [])
            
    if status == Status.SUCCESS:
        ani.event_source.stop()

    return [robot_path, robot_marker] + rollout_lines

ani = FuncAnimation(fig, update, frames=range(100),
                    init_func=init, blit=True, repeat=False, interval=100)

plt.show()
