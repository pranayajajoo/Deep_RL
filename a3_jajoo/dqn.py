import numpy as np
import torch
import copy
import collections


def target_network_refresh(q_network):
    target_network = copy.deepcopy(q_network)
    return target_network

class DQN:
    """Class that implements Deep Q-networks."""

    def __init__(self,
                 q_network,
                 optimizer,
                 replay_buffer,
                 explorer,
                 discount,
                 gradient_updates_per_target_refresh,
                 gradient_update_frequency=1,
                 input_preprocessor= lambda x: x,
                 minibatch_size=32,
                 min_replay_size_before_updates=32,
                 track_statistics=False,
                 reward_phi=lambda reward: reward):
        self.optimizer = optimizer
        
        self.replay_buffer = replay_buffer
        self.explorer = explorer
        self.discount = discount
        self.gradient_updates_per_target_refresh = gradient_updates_per_target_refresh
        self.gradient_update_frequency = gradient_update_frequency
        self.input_preprocessor = input_preprocessor
        self.minibatch_size = minibatch_size
        self.min_replay_size_before_updates = min_replay_size_before_updates
        self.track_statistics = track_statistics
        self.reward_phi = reward_phi

        self.prev_state = None
        self.prev_action = None
        self.steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = q_network.to(self.device)
        self.target_network = target_network_refresh(self.q_network).to(self.device)


    def act(self, obs) -> int:
        """Returns an integer 
        """
        self.prev_state = obs
        state_tensor = self.input_preprocessor(torch.from_numpy(self.prev_state).float()).to(self.device)
        q_action_values = self.q_network(state_tensor).detach().cpu().numpy().squeeze()
        action = self.explorer.select_action(q_action_values)
        self.prev_action = action
        return action, q_action_values[action]
    
    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        batched_rewards = batched_rewards.to(self.device)
        batched_next_states = batched_next_states.to(self.device)
        batched_discounts = batched_discounts.to(self.device)
        batch_terminated = batch_terminated.to(self.device)

        q_values_next = self.target_network(batched_next_states)
        max_q_values_next = torch.max(q_values_next, dim=1)[0]
        target = batched_rewards + (batched_discounts * max_q_values_next * (~batch_terminated))
        return target


    def gradient_update(self):
        
        minibatch = self.replay_buffer.sample(self.minibatch_size)
        batch_states = torch.stack([transition['state'] for transition in minibatch]).to(self.device)
        batch_actions = torch.tensor([transition['action'] for transition in minibatch], dtype=torch.long).to(self.device)
        batch_rewards = torch.tensor([transition['reward'] for transition in minibatch], dtype=torch.float32).to(self.device)
        batch_next_states = torch.stack([transition['next_state'] for transition in minibatch]).to(self.device)
        batch_discounts = torch.tensor([transition['discount'] for transition in minibatch], dtype=torch.float32).to(self.device)
        batch_terminated = torch.tensor([transition['terminated'] for transition in minibatch], dtype=torch.bool).to(self.device)
        with torch.no_grad():
            targets = self.compute_targets(batch_rewards, batch_next_states, batch_discounts, batch_terminated).detach()
        q_values = self.q_network(self.input_preprocessor(batch_states).to(self.device))
        q_values_selected = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1) 
        loss = torch.nn.MSELoss()(q_values_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.
        Returns:
            None
        """
        self.steps += 1
        reward = self.reward_phi(reward)
        state = self.input_preprocessor(torch.from_numpy(self.prev_state).float().to(self.device))
        action = self.prev_action
        next_state = self.input_preprocessor(torch.from_numpy(obs).float().to(self.device))
        self.replay_buffer.append(state, action, reward, next_state, terminated, truncated)
        
        if len(self.replay_buffer.buffer) >= self.min_replay_size_before_updates and self.steps % self.gradient_update_frequency == 0 :
            self.gradient_update()
        if self.steps % self.gradient_updates_per_target_refresh == 0:
            self.target_network = target_network_refresh(self.q_network).to(self.device)
