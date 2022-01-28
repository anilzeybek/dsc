from torch import nn
import torch
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, goal_dim: int, hidden_1=256, hidden_2=256,
                 max_action=1):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.max_action = max_action

        self.fc1 = nn.Linear(self.obs_dim + self.goal_dim, self.hidden_1)
        self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.output = nn.Linear(self.hidden_2, self.action_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = torch.tanh(self.output(x)) * self.max_action

        return output


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, goal_dim: int, hidden_1=256, hidden_2=256):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        self.fc1 = nn.Linear(self.obs_dim + self.goal_dim + self.action_dim, self.hidden_1)
        self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.output1 = nn.Linear(self.hidden_2, 1)

        self.fc3 = nn.Linear(self.obs_dim + self.goal_dim + self.action_dim, self.hidden_1)
        self.fc4 = nn.Linear(self.hidden_1, self.hidden_2)
        self.output2 = nn.Linear(self.hidden_2, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        obs_action = torch.cat([obs, action], dim=-1)

        q1 = F.relu(self.fc1(obs_action))
        q1 = F.relu(self.fc2(q1))
        output1 = self.output1(q1)

        q2 = F.relu(self.fc3(obs_action))
        q2 = F.relu(self.fc4(q2))
        output2 = self.output2(q2)

        return output1, output2
