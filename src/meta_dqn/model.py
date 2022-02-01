import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_1: int, hidden_2: int) -> None:
        super(QNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        self.fc1 = nn.Linear(self.obs_dim, self.hidden_1)
        self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.fc3 = nn.Linear(self.hidden_2, self.action_dim)

    def add_node_to_output(self, new_dim: int) -> None:
        last_layer_weight = self.state_dict()['fc3.weight']
        last_layer_bias = self.state_dict()['fc3.bias']

        self.fc3 = nn.Linear(self.hidden_2, new_dim)

        self.fc3.weight.data[:-1] = last_layer_weight
        self.fc3.bias.data[:-1] = last_layer_bias

        self.fc3.bias.data[-1] += 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
