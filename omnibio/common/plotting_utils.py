import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import os

def plot_navigation_2d(ax, path_history, goal, circ_obstacles, rect_obstacles, rollout_history=None):
    """
    Plots the 2D navigation view on a given matplotlib axes.
    """
    ax.clear()
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Navigation")

    # Plot rollouts if provided
    if rollout_history:
        for rollouts in rollout_history:
            for rollout in rollouts:
                start_pos = rollout['start']
                end_pos = rollout['end']
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'c-', alpha=0.1)

    # Plot main path
    path = np.array(path_history)
    ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')

    # Plot obstacles
    for obs in circ_obstacles:
        ax.add_patch(plt.Circle(obs, 0.5, color='r', alpha=0.5))
    for rect in rect_obstacles:
        ax.add_patch(Rectangle((rect['center'][0] - rect['size'][0]/2, rect['center'][1] - rect['size'][1]/2), 
                               rect['size'][0], rect['size'][1], color='purple', alpha=0.5))
    
    ax.legend()

def plot_costs(ax, cost_history):
    """
    Plots the cost functions over time on a given matplotlib axes.
    """
    ax.clear()
    ax.set_title("Costs")
    ax.set_xlabel("Step")
    ax.grid(True)

    if not cost_history:
        ax.set_ylim(0, 20)
        return

    costs = np.array(cost_history)
    ax.plot(costs[:, 0], label='Total')
    ax.plot(costs[:, 1], label='Distance')
    ax.plot(costs[:, 2], label='Circ Obstacle')
    ax.plot(costs[:, 3], label='Rect Obstacle')
    ax.plot(costs[:, 4], label='Control')
    ax.legend()
    ax.relim()
    ax.autoscale_view()

def save_simulation_graphs(save_dir, path_history, cost_history, goal, circ_obstacles, rect_obstacles):
    """
    Saves the final simulation state as a set of graphs.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Save 2D Nav Plot ---
    fig_nav, ax_nav = plt.subplots(figsize=(10, 10))
    plot_navigation_2d(ax_nav, path_history, goal, circ_obstacles, rect_obstacles)
    fig_nav.savefig(os.path.join(save_dir, "2d_navigation.png"))
    plt.close(fig_nav)

    # --- Save Cost Plot ---
    fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
    plot_costs(ax_cost, cost_history)
    fig_cost.savefig(os.path.join(save_dir, "cost_functions.png"))
    plt.close(fig_cost)
    
    # --- Save 3D Nav Plot ---
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    path = np.array(path_history)
    ax_3d.plot(path[:, 0], path[:, 1], 0, 'b-', label='Robot Path')
    ax_3d.scatter(path[0, 0], path[0, 1], 0, c='g', s=100, label='Start')
    ax_3d.scatter(goal[0], goal[1], 0, c='r', marker='*', s=200, label='Goal')
    for obs in circ_obstacles:
        theta = np.linspace(0, 2 * np.pi, 100)
        x = obs[0] + 0.5 * np.cos(theta)
        y = obs[1] + 0.5 * np.sin(theta)
        z = np.zeros_like(x)
        ax_3d.plot(x, y, z, color='r', alpha=0.5)
    # 3D rects are complex, skipping for static plot
    ax_3d.set_xlabel('X Position')
    ax_3d.set_ylabel('Y Position')
    ax_3d.set_title('3D Robot Navigation Path')
    ax_3d.legend()
    fig_3d.savefig(os.path.join(save_dir, "3d_navigation.png"))
    plt.close(fig_3d)

    # --- Save MATLAB data ---
    try:
        from scipy.io import savemat
        mat_data = {
            'path': np.array(path_history),
            'costs': np.array(cost_history),
            'goal': np.array(goal),
            'circ_obstacles': np.array(circ_obstacles),
            'rect_obstacles': rect_obstacles
        }
        savemat(os.path.join(save_dir, "simulation_data.mat"), mat_data)
        print(f"Data saved for MATLAB in {save_dir}")
    except ImportError:
        print(f"Skipping MATLAB export in {save_dir}: `scipy` is not installed.")

    print(f"Simulation graphs saved in {save_dir}")
