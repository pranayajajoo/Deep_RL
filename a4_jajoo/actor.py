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

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, obs) -> int:
        x = torch.relu(self.l1(obs))
        x = torch.relu(self.l2(x))
        x = self.l3(x)

        return self.max_action * torch.tanh(x)


class PPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # Change log_std initialization
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Changed from (1, action_dim) to (action_dim,)

    def forward(self, x):
        mean = self.actor_mean(x)
        std = torch.exp(self.log_std)  # Removed expand_as since log_std is now the right shape
        dist = Normal(mean, std)
        return dist
    

    