import numpy as np

def compute_q_values(state_action_features, weights):
    """Takes in Q-values and produces epsilon-greedy action probabilities

    where ties are broken evenly.

    Args:
        state_action_features: a numpy array of state-action features
        weights: a numpy array of weights
         
    Returns:
        scalar numpy Q-value
    """
    q_vals = np.dot(state_action_features, weights)
    return q_vals

def get_action_values(obs, feature_extractor, weights, num_actions):
    """Applies feature_extractor to observation and produces action values

    Args:
        obs: observation
        feature_extractor: extracts features for a state-action pair
        weights: a numpy array of weights
        num_actions: an integer number of actions
         
    Returns:
        a numpy array of Q-values
    """
    action_values = np.zeros(num_actions)
    for action in range(num_actions):
        features = feature_extractor(obs, action)
        action_values[action] = compute_q_values(features, weights)
    return action_values

class SemiGradientSARSA:
    """Class that implements Linear Semi-gradient SARSA."""

    def __init__(self,
                 num_state_action_features,
                 num_actions,
                 feature_extractor,
                 step_size,
                 explorer,
                 discount,
                 initial_weight_value=5.0,
                 n_step = 12):
        self.num_state_action_features = num_state_action_features
        self.num_actions = num_actions
        self.explorer = explorer
        self.step_size = step_size
        self.feature_extractor = feature_extractor
        self.w = np.full(num_state_action_features, initial_weight_value)
        self.discount = discount
        self.prev_state = None
        self.prev_action = None
        self.n_step = n_step
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def update_q(self, obs, action, reward, next_obs, next_action, terminated):

        # updating buffer
        self.state_buffer.append(self.prev_state)
        self.action_buffer.append(self.prev_action)
        self.reward_buffer.append(reward)
        if max(len(self.state_buffer), len(self.action_buffer), len(self.reward_buffer)) > self.n_step:
            self.state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)       
        


        if len(self.state_buffer) == self.n_step:
            G = 0
            for i in range(self.n_step):
                G += self.reward_buffer[i] * (self.discount ** i)
            if not terminated:
                next_fea = self.feature_extractor(next_obs, next_action)
                G = G + (self.discount ** self.n_step) * np.dot(next_fea, self.w)
    
            curr_fea = self.feature_extractor(obs, action)
            curr_q =  np.dot(curr_fea, self.w )

            self.w += self.step_size * (G - curr_q) * curr_fea
    

    def act(self, obs) -> int:
        """Returns an integer 
        """
        self.prev_state = obs
        q_vals = get_action_values(obs, self.feature_extractor, self.w, self.num_actions)
        action = self.explorer.select_action(q_vals) # replace this line
        self.prev_action = action
        return action
        

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        state = self.prev_state
        action = self.prev_action
        next_state = obs

        
        next_action = self.act(obs)
        self.update_q(state, action, reward, next_state, next_action, terminated)
