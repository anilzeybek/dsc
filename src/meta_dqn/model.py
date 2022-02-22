import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1, hidden_2):
        super(QNetwork, self).__init__()
        self.hidden_2 = hidden_2

        self.fc1 = nn.Linear(obs_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, action_dim)

    def add_node_to_output(self, new_dim):
        last_layer_weight = self.state_dict()['fc3.weight']
        last_layer_bias = self.state_dict()['fc3.bias']

        self.fc3 = nn.Linear(self.hidden_2, new_dim)

        self.fc3.weight.data[:-1] = last_layer_weight
        self.fc3.bias.data[:-1] = last_layer_bias

        self.fc3.bias.data[-1] += 100

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
