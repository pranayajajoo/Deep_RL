import numpy as np

def compute_epsilon_greedy_action_probs(q_vals, epsilon):
	"""Takes in Q-values and produces epsilon-greedy action probabilities

	where ties are broken evenly.

	Args:
	    q_vals: a numpy array of action values
	    epsilon: epsilon-greedy epsilon in ([0,1])
	     
	Returns:
	    numpy array of action probabilities
	"""
	assert len(q_vals.shape) == 1
     
	action_probabilities = np.ones_like(q_vals) * epsilon / len(q_vals)
	max_actions = np.where(q_vals == np.max(q_vals))[0]
	action_probabilities[max_actions] += (1 - epsilon) / len(max_actions)
	
	assert action_probabilities.shape == q_vals.shape
	return action_probabilities	



class ConstantEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, epsilon, num_actions, min_epsilon, decay_rate):
        self.epsilon = epsilon
        self.num_actions = num_actions
        # added to decay epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select_action(self, action_values) -> int:
        max_actions = np.where(action_values == np.max(action_values))[0]
        action_probs = compute_epsilon_greedy_action_probs(action_values, self.epsilon)
        return np.random.choice(len(action_probs), p=action_probs)
    
    
class AdaptiveConstantEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, epsilon, num_actions, min_epsilon, decay_rate):
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select_action(self, action_values) -> int:
        max_actions = np.where(action_values == np.max(action_values))[0]
        action_probs = compute_epsilon_greedy_action_probs(action_values, self.epsilon)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        return np.random.choice(len(action_probs), p=action_probs)