import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from actor import Actor
from critic import Critic

class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount,
            tau,

            initial_policy_noise,
            decay_steps,
            
            noise_clip,
            policy_update_fequency,


            batch_size,
            replay_buffer
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.initial_policy_noise = initial_policy_noise
        self.noise_clip = noise_clip
        self.policy_update_fequency = policy_update_fequency

        self.prev_state = None
        self.prav_action = None
        self.steps = 0
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        
        self.decay_steps = decay_steps
        # self.current_noise = self.initial_policy_noise
        self.current_noise = 0.2

    
    def compute_targets(self, batched_rewards, batched_actions, batched_next_states, batched_discounts, batched_terminated):
        with torch.no_grad():
            noise =  (torch.randn_like(batched_actions) * self.current_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(batched_next_states) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(batched_next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = batched_rewards + ((~batched_terminated) * self.discount * target_Q)
        return target_Q

    def select_action(self, state):

        # Linearly decaying policy noise
        # decay_ratio = min(1.0, self.steps / self.decay_steps)
        # self.current_noise = max(self.initial_policy_noise * (1 - decay_ratio), 0.0001)

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.prev_action = self.actor(state).cpu().data.numpy().flatten()
        self.prev_action += np.random.normal(0, self.current_noise, size=self.prev_action.shape)
        return np.clip(self.prev_action, -self.max_action, self.max_action)
    
    def actor_gradient_update(self, batch_states, batch_actions, batch_next_states, batch_discounts, batch_terminated):
        actor_loss = -self.critic.q1(batch_states, self.actor(batch_states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    

    def critic_gradient_update(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_discounts, batch_terminated):
        # targets
        with torch.no_grad():
            targets = self.compute_targets(batch_rewards, batch_actions, batch_next_states, batch_discounts, batch_terminated).detach()
        # current q estimates
        current_Q1, current_Q2 = self.critic(batch_states, batch_actions)
        # critic loss = (onestepreward + discount * next state return) - current state return
        loss = nn.functional.mse_loss(current_Q1, targets) + nn.functional.mse_loss(current_Q2, targets)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step() 


    
    def process_transition(self, state, action, reward, next_state, terminated, truncated):
        self.steps += 1
        state = torch.FloatTensor(state.reshape(1, -1))
        action = torch.FloatTensor(action.reshape(1, -1))
        next_state = torch.FloatTensor(next_state.reshape(1, -1))
        reward = torch.FloatTensor([reward])
        terminated = torch.FloatTensor([terminated])
        truncated = torch.FloatTensor([truncated])

        self.replay_buffer.append(state, action, reward, next_state, terminated, truncated)

        if len(self.replay_buffer) >= self.batch_size:
            minibatch = self.replay_buffer.sample(self.batch_size)
            batch_states = torch.cat([transition['state'] for transition in minibatch], dim=0).to(self.device)
            batch_actions = torch.cat([transition['action'] for transition in minibatch], dim=0).to(self.device)
            batch_rewards = torch.tensor([transition['reward'] for transition in minibatch], dtype=torch.float32).unsqueeze(1).to(self.device)
            batch_next_states = torch.cat([transition['next_state'] for transition in minibatch], dim=0).to(self.device)
            batch_discounts = torch.tensor([transition['discount'] for transition in minibatch], dtype=torch.float32).unsqueeze(1).to(self.device)
            batch_terminated = torch.tensor([transition['terminated'].item() for transition in minibatch], dtype=torch.bool).unsqueeze(1).to(self.device)


            if batch_states == None:
                return
            

            self.critic_gradient_update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_discounts, batch_terminated)

            if self.steps % self.policy_update_fequency == 0:
                self.actor_gradient_update(batch_states, batch_actions, batch_next_states, batch_discounts, batch_terminated)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
