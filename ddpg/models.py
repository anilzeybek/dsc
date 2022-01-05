from torch import nn
import torch
from torch.nn import functional as f


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, goal_dim: int, hidden_1=256, hidden_2=256,
                 action_bounds=(-1, 1)):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.action_bounds = torch.Tensor(action_bounds)

        self.fc1 = nn.Linear(self.state_dim + self.goal_dim, self.hidden_1)
        self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.output = nn.Linear(self.hidden_2, self.action_dim)

    def forward(self, x: torch.Tensor):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        output = torch.tanh(self.output(x)) * self.action_bounds[1]

        return output


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, goal_dim: int, hidden_1=256, hidden_2=256):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        self.fc1 = nn.Linear(self.state_dim + self.goal_dim + self.action_dim, self.hidden_1)
        self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.output = nn.Linear(self.hidden_2, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        x = f.relu(self.fc1(torch.cat([x, a], dim=-1)))
        x = f.relu(self.fc2(x))
        output = self.output(x)

        return output
