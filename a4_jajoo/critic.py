import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(1)  # ensure shape [batch_size, 1]
        # import ipdb; ipdb.set_trace()
        state_action = torch.cat([state, action], dim=1)

        q1 = torch.relu(self.l1(state_action))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(state_action))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
    def q1(self, state, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(1)  # ensure shape [batch_size, 1]
        state_action  = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(state_action))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    

class PPOCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        return self.critic(x)