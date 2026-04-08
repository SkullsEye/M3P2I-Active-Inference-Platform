import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import os
import sys

# Add the parent directory to the path to find the 'common' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.behavior_tree import Sequence, Action, Blackboard, Status
from common.cost_planner import CostPlanner, distance_to_goal_cost, obstacle_avoidance_cost, control_effort_cost
from common.active_inference_planner import ActiveInferencePlanner

# --- Simulation State and Setup ---
class SimulationState:
    def __init__(self):
        # Environment
        self.initial_state = np.array([0.0, 0.0])
        self.goal = np.array([10.0, 10.0])
        self.obstacles = [np.array([5.0, 5.0]), np.array([8.0, 2.0]), np.array([3.0, 7.0])]
        # Planners
        self.ai_planner = ActiveInferencePlanner(goal=self.goal, initial_state=self.initial_state, num_rollouts=50, noise_level=1.5)
        self.cost_functions = {
            'distance': distance_to_goal_cost,
            'obstacle': lambda state, action, goal: obstacle_avoidance_cost(state, action, goal, self.obstacles),
            'control': control_effort_cost
        }
        self.cost_planner = CostPlanner(cost_functions=list(self.cost_functions.values()))
        # Blackboard
        self.blackboard = Blackboard()
        self.blackboard.set("state", self.initial_state)
        # History
        self.path_history = [self.initial_state.copy()]
        self.cost_history = []
        # Intermediate state for 2-stage animation
        self.best_action = None
        self.all_rollouts = []
        self.status = Status.RUNNING

sim_state = SimulationState()

# Behavior Tree
class PlanAndExecute(Action):
    def tick(self, blackboard):
        # This BT node is now a placeholder, as the logic is in the animation loop
        return sim_state.status

robot_behavior = Sequence("NavigateToGoal", [
    PlanAndExecute("PlanAndExecute")
])

# --- Animation Setup ---
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 3, figure=fig)

ax_nav = fig.add_subplot(gs[:, 0:2]) # Main navigation plot
ax_cost = fig.add_subplot(gs[0, 2])  # Cost plot
ax_text = fig.add_subplot(gs[1, 2])  # Text info plot

# Setup Navigation Plot
ax_nav.set_xlim(-2, 12)
ax_nav.set_ylim(-2, 12)
ax_nav.set_aspect('equal')
ax_nav.grid(True)
ax_nav.set_title("Navigation")
ax_nav.plot(sim_state.goal[0], sim_state.goal[1], 'r*', markersize=15, label='Goal')
for obs in sim_state.obstacles:
    ax_nav.add_patch(plt.Circle(obs, 0.5, color='r', alpha=0.5))
robot_path, = ax_nav.plot([], [], 'b-', linewidth=2, label='Robot Path')
robot_marker, = ax_nav.plot([], [], 'bo', markersize=8)
rollout_lines = [ax_nav.plot([], [], 'gray', alpha=0.5)[0] for _ in range(sim_state.ai_planner.num_rollouts)]
best_rollout_line, = ax_nav.plot([], [], 'g-', linewidth=2)
ax_nav.legend()

# Setup Cost Plot
ax_cost.set_xlim(0, 100)
ax_cost.set_ylim(0, 20)
ax_cost.grid(True)
ax_cost.set_title("Costs")
ax_cost.set_xlabel("Step")
cost_lines = {
    'total': ax_cost.plot([], [], label='Total')[0],
    'distance': ax_cost.plot([], [], label='Distance')[0],
    'obstacle': ax_cost.plot([], [], label='Obstacle')[0],
    'control': ax_cost.plot([], [], label='Control')[0],
}
ax_cost.legend()

# Setup Text Info Plot
ax_text.axis('off')
ax_text.set_title("Info")
info_text = ax_text.text(0.05, 0.95, '', verticalalignment='top', fontsize=10, fontfamily='monospace')


def init():
    # This function is required for blitting but we can leave it empty
    # as we are setting initial data manually.
    return []

def update(frame):
    is_planning_phase = frame % 2 == 0
    sim_step = frame // 2

    current_state = sim_state.blackboard.get("state")
    
    if sim_state.status != Status.RUNNING:
        ani.event_source.stop()
        return []

    if is_planning_phase:
        # --- PLANNING PHASE ---
        info_text.set_text(f"""Step: {sim_step}
Phase: Planning...
State: {np.round(current_state, 2)}""")

        # 1. Plan rollouts
        sim_state.all_rollouts = sim_state.ai_planner.step(current_state)
        
        # 2. Evaluate rollouts
        costs, rollout_ends = [], []
        for action in sim_state.all_rollouts:
            next_state = current_state + action * 0.5
            costs.append(sim_state.cost_planner.calculate_total_cost(next_state, action, sim_state.goal))
            rollout_ends.append(next_state)

        # 3. Find best action and store it
        best_action_index = np.argmin(costs)
        sim_state.best_action = sim_state.all_rollouts[best_action_index]
        
        # 4. Update cost plot data
        dist_cost = sim_state.cost_functions['distance'](current_state, sim_state.best_action, sim_state.goal)
        obs_cost = sim_state.cost_functions['obstacle'](current_state, sim_state.best_action, sim_state.goal)
        ctrl_cost = sim_state.cost_functions['control'](current_state, sim_state.best_action, sim_state.goal)
        total_cost = costs[best_action_index]
        sim_state.cost_history.append([total_cost, dist_cost, obs_cost, ctrl_cost])
        
        cost_data = np.array(sim_state.cost_history)
        steps = range(len(cost_data))
        cost_lines['total'].set_data(steps, cost_data[:, 0])
        cost_lines['distance'].set_data(steps, cost_data[:, 1])
        cost_lines['obstacle'].set_data(steps, cost_data[:, 2])
        cost_lines['control'].set_data(steps, cost_data[:, 3])
        ax_cost.relim()
        ax_cost.autoscale_view()

        # 5. Draw all rollouts
        for i, line in enumerate(rollout_lines):
            end_pos = rollout_ends[i]
            line.set_data([current_state[0], end_pos[0]], [current_state[1], end_pos[1]])
        
        # 6. Highlight the best one
        best_rollout_end = rollout_ends[best_action_index]
        best_rollout_line.set_data([current_state[0], best_rollout_end[0]], [current_state[1], best_rollout_end[1]])

    else:
        # --- EXECUTION PHASE ---
        info_text.set_text(f"""Step: {sim_step}
Phase: Executing...
State: {np.round(current_state, 2)}""")

        # 1. Clear rollout lines for a "flash" effect
        for line in rollout_lines:
            line.set_data([], [])
        best_rollout_line.set_data([], [])

        # 2. Apply the stored best action
        new_state = current_state + sim_state.best_action * 0.5
        sim_state.blackboard.set("state", new_state)
        sim_state.path_history.append(new_state.copy())
        sim_state.ai_planner.update_beliefs(new_state)

        # 3. Update robot path plot
        path_data = np.array(sim_state.path_history)
        robot_path.set_data(path_data[:, 0], path_data[:, 1])
        robot_marker.set_data([new_state[0]], [new_state[1]])

        # 4. Check for goal
        if np.linalg.norm(new_state - sim_state.goal) < 0.5:
            print("Goal reached!")
            sim_state.status = Status.SUCCESS

    # The list of all artists to be re-drawn
    artists = [robot_path, robot_marker, best_rollout_line, info_text] + rollout_lines + list(cost_lines.values())
    return artists

ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=False, repeat=False, interval=200)
plt.tight_layout()
plt.show()
