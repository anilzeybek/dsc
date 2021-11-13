import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size, act_limits, hidden_1, hidden_2):
        super(PolicyNetwork, self).__init__()
        self.act_limits = torch.as_tensor(act_limits, dtype=torch.float32)

        self.fc1 = nn.Linear(obs_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(300, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return torch.tanh(x) * self.act_limits


class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_1, hidden_2):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_size + action_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
