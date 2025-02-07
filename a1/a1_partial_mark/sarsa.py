import numpy as np


class SARSA:
    """Class that implements SARSA."""

    def __init__(self,
                 num_states,
                 num_actions,
                 step_size,
                 explorer,
                 discount=0.99,):
        self.explorer = explorer
        self.step_size = step_size
        self.q = np.zeros((num_states, num_actions))
        self.discount = discount
        self.prev_state = None
        self.prev_action = None


    def update_q(self, obs, action, reward, next_obs, next_action, terminated):
        target = reward + self.discount * self.q[next_obs, next_action] * (not terminated)
        self.q[obs, action] += self.step_size * (target - self.q[obs, action])    

    def act(self, obs: int) -> int:
        """Returns an integer 
        """
        q_vals = self.q[obs]
        action = self.explorer.select_action(action_values = q_vals) # replace this line
        return action
        

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        if self.prev_state is not None and self.prev_action is not None:
            next_action = self.act(obs)
            self.update_q(self.prev_state, self.prev_action, reward, obs, next_action, terminated) 
            self.prev_action = next_action
        else:
            self.prev_action = self.act(obs)
        self.prev_state = obs
