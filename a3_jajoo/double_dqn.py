import dqn
import torch

class DoubleDQN(dqn.DQN):

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):

        batched_rewards = batched_rewards.to(self.device)
        batched_next_states = self.input_preprocessor(batched_next_states).to(self.device)
        batched_discounts = batched_discounts.to(self.device)
        batch_terminated = batch_terminated.to(self.device)
        next_action_online = torch.argmax(self.q_network(batched_next_states), dim = 1)
        with torch.no_grad():
            q_values_next = self.target_network(batched_next_states)
            max_q_values_next = torch.max(q_values_next, dim=1).values
            target = batched_rewards + batched_discounts * max_q_values_next * (1 - batch_terminated.float())
        return target
