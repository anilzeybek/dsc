from torch import nn
import torch
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, n_hidden1=256, n_hidden2=256, n_hidden3=256, action_bounds=[-1, 1]):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.action_bounds = torch.Tensor(action_bounds)

        self.fc1 = nn.Linear(self.state_dim + self.goal_dim, self.n_hidden1)
        self.fc2 = nn.Linear(self.n_hidden1, self.n_hidden2)
        self.fc3 = nn.Linear(self.n_hidden2, self.n_hidden3)
        self.output = nn.Linear(self.n_hidden3, self.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = torch.tanh(self.output(x)) * self.action_bounds[1]

        return output


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, n_hidden1=256, n_hidden2=256, n_hidden3=256, action_size=1):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_dim + self.goal_dim +
                             self.action_size, self.n_hidden1)
        self.fc2 = nn.Linear(self.n_hidden1, self.n_hidden2)
        self.fc3 = nn.Linear(self.n_hidden2, self.n_hidden3)
        self.output = nn.Linear(self.n_hidden3, 1)

    def forward(self, x, a):
        x = F.relu(self.fc1(torch.cat([x, a], dim=-1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.output(x)

        return output
