import torch.nn as nn
import torch.nn.functional as F


class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden_1=64, hidden_2=64, hidden_3=32):
        super(DuelingNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim + goal_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

        self.v1 = nn.Linear(hidden_2, hidden_3)
        self.v2 = nn.Linear(hidden_3, 1)

        self.a1 = nn.Linear(hidden_2, hidden_3)
        self.a2 = nn.Linear(hidden_3, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = F.relu(self.v1(x))
        v = self.v2(v)

        a = F.relu(self.a1(x))
        a = self.a2(a)

        Q = v + a - a.mean(dim=0, keepdim=True)
        return Q
