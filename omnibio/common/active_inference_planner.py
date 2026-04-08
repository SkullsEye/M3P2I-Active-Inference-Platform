import numpy as np

class ActiveInferencePlanner:
    def __init__(self, goal, initial_state, num_rollouts=10, noise_level=0.5):
        self.goal = goal
        self.state = initial_state
        self.beliefs = self.initialize_beliefs()
        self.num_rollouts = num_rollouts
        self.noise_level = noise_level

    def initialize_beliefs(self):
        """
        Initialize beliefs about the state of the world.
        For simplicity, we'll assume the initial beliefs are the same as the state.
        """
        return self.state.copy()

    def update_beliefs(self, new_observation):
        """
        Update beliefs based on a new observation.
        In a real implementation, this would involve a more complex Bayesian update.
        """
        self.beliefs = new_observation.copy()
        self.state = self.beliefs # In this simple case, state is what we believe it is

    def plan_rollouts(self):
        """
        Plan a set of rollouts (possible actions) to evaluate.
        """
        direction_to_goal = self.goal - self.state
        if np.linalg.norm(direction_to_goal) > 0:
            base_action = direction_to_goal / np.linalg.norm(direction_to_goal)
        else:
            base_action = np.zeros_like(self.state)

        rollouts = [base_action]
        for _ in range(self.num_rollouts - 1):
            noise = np.random.normal(0, self.noise_level, size=base_action.shape)
            noisy_action = base_action + noise
            if np.linalg.norm(noisy_action) > 0:
                noisy_action /= np.linalg.norm(noisy_action)
            rollouts.append(noisy_action)
        
        return np.array(rollouts)

    def step(self, observation):
        """
        Perform one step of the active inference loop.
        """
        self.update_beliefs(observation)
        rollouts = self.plan_rollouts()
        return rollouts
