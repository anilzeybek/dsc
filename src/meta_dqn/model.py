import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_1: int, hidden_2: int) -> None:
        super(QNetwork, self).__init__()

        self.obs_size = obs_dim
        self.action_size = action_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        self.fc1 = nn.Linear(self.obs_size, self.hidden_1)
        self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.fc3 = nn.Linear(self.hidden_2, self.action_size)

    def change_last_layer(self, new_size: int) -> None:
        self.fc3 = nn.Linear(self.hidden_2, new_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
