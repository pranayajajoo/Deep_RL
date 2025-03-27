import collections
import numpy as np

class ReplayBuffer:

    def __init__(self, buffer_size, discount=0.99, n_step=1):
        self.buffer = collections.deque([], maxlen=buffer_size)
        self.discount = discount
        self.n_step = n_step
        
    def __len__(self):
        return len(self.buffer)
    

    def append(self, state, action, reward, next_state, terminated, truncated):
        transition = {'state': state,
                      'action': action,
                      'reward': reward,
                      'next_state': next_state,
                      'discount': self.discount,
                      'terminated': terminated,
                      'truncated': truncated}
        self.buffer.append(transition)

    def create_multistep_transition(self, index):
        
        transition = self.buffer[index]
        n_step_transition = transition.copy()
        for i in range(1, self.n_step):
            if index + i >= len(self.buffer):
                break
            next_transition = self.buffer[index + i]
            n_step_transition['reward'] += next_transition['reward'] * (self.discount ** i)
            n_step_transition['next_state'] = next_transition['next_state']
            n_step_transition['discount'] *= next_transition['discount']
            n_step_transition['terminated'] = next_transition['terminated']
            if n_step_transition['terminated'] or n_step_transition['truncated']:
                break
        return n_step_transition
    
    def sample(self, n_transitions):
        assert len(self.buffer) >= n_transitions
        batch_indices = np.random.choice(len(self.buffer), size=n_transitions, replace=False)
        batch = [self.create_multistep_transition(index) for index in batch_indices]
        return batch
