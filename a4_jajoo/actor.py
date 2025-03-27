import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def target_network_refresh(q_network):
    target_network = copy.deepcopy(q_network)
    return target_network

class Actor(nn.Module):
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
    