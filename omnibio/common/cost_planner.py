import numpy as np

class CostPlanner:
    def __init__(self, cost_functions):
        self.cost_functions = cost_functions

    def calculate_total_cost(self, state, action, goal):
        total_cost = 0
        for cost_function in self.cost_functions:
            total_cost += cost_function(state, action, goal)
        return total_cost

# Example Cost Functions

def distance_to_goal_cost(state, action, goal):
    """
    Calculates the distance between the current state and the goal.
    """
    return np.linalg.norm(state - goal)

def obstacle_avoidance_cost(state, action, goal, obstacles):
    """
    Calculates a cost for being too close to obstacles.
    """
    cost = 0
    for obstacle in obstacles:
        distance_to_obstacle = np.linalg.norm(state - obstacle)
        if distance_to_obstacle < 0.5:  # Obstacle radius
            cost += 1 / (distance_to_obstacle + 1e-6)
    return cost

def control_effort_cost(state, action, goal):
    """
    Penalizes large control actions.
    """
    return np.linalg.norm(action)

def rectangle_obstacle_cost(state, action, goal, rect_obstacles):
    """
    Calculates a cost for being too close to rectangular obstacles.
    A rectangle is defined as a dict {'center': [x, y], 'size': [w, h]}
    """
    cost = 0
    for rect in rect_obstacles:
        center = rect['center']
        size = rect['size']
        
        # Calculate distance from point to rectangle
        dx = max(abs(state[0] - center[0]) - size[0] / 2, 0)
        dy = max(abs(state[1] - center[1]) - size[1] / 2, 0)
        distance_to_rect = np.sqrt(dx**2 + dy**2)
        
        # Add cost if inside or very close
        if distance_to_rect < 0.1: # A small threshold
            cost += 1 / (distance_to_rect + 1e-6)
            
    return cost

def escape_local_minimum_cost(state, action, goal, recent_positions):
    """
    Adds a cost for staying too close to the recent path.
    This encourages the robot to escape from local minima.
    """
    if not recent_positions:
        return 0
    
    avg_recent_pos = np.mean(recent_positions, axis=0)
    distance_from_avg = np.linalg.norm(state - avg_recent_pos)
    
    # If the potential state is very close to the average of recent positions,
    # it's not helping us escape, so add a high cost.
    if distance_from_avg < 1.0: # 1.0 is the radius of the "stuck" zone
        return 10 * (1.0 - distance_from_avg)
    return 0
