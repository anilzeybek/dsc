import torch
import torch.nn as nn
import torch.nn. functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim, hidden_1=256, hidden_2=256, action_bounds_high=1):
        super(Actor, self).__init__()
        self.action_bounds_high = torch.Tensor(action_bounds_high)

        self.fc1 = nn.Linear(obs_dim + goal_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.tanh(x) * self.action_bounds_high


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim, hidden_1=256, hidden_2=256):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(obs_dim + goal_dim + action_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 1)

        self.fc4 = nn.Linear(obs_dim + goal_dim + action_dim, hidden_1)
        self.fc5 = nn.Linear(hidden_1, hidden_2)
        self.fc6 = nn.Linear(hidden_2, 1)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)

        q1 = F.relu(self.fc1(obs_action))
        q1 = F.relu(self.fc2(q1))
        output1 = self.fc3(q1)

        q2 = F.relu(self.fc4(obs_action))
        q2 = F.relu(self.fc5(q2))
        output2 = self.fc6(q2)

        return output1, output2
