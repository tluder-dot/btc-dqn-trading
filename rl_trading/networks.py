"""Q-network (same architecture as DQNetwork in TD2/TD3)."""

import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):
    """MLP mapping a state to one Q-value per action: Q_θ(s, ·).

    Two hidden layers with ReLU, linear output (Q-values are unbounded, so no
    activation on the last layer).
    """

    def __init__(self, state_size, n_actions, hidden_size=128):
        super(DQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
