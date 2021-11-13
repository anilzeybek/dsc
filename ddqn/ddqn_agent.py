import json
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .model import QNetwork
from .replay_buffer import ReplayBuffer
from copy import deepcopy


class DDQNAgent:
    def __init__(self, obs_size, action_size) -> None:
        self.obs_size = obs_size
        self.action_size = action_size
        self.hyperparams = self.read_hyperparams()['agent_over_actions']

        self.Q_network = QNetwork(obs_size, action_size, self.hyperparams['hidden_1'], self.hyperparams['hidden_2'])
        self.target_network = deepcopy(self.Q_network)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.hyperparams['lr'])

        self.eps = self.hyperparams['eps_start']
        self.memory = ReplayBuffer(self.hyperparams['buffer_size'])
        self.t_step = 0
        self.learn_count = 0

    def read_hyperparams(self) -> Dict[str, Any]:
        with open('hyperparams.json') as f:
            hyperparams = json.load(f)
            return hyperparams

    def act(self, obs):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_size)
        else:
            obs = torch.from_numpy(obs).unsqueeze(0)
            action_values = self.Q_network(obs)
            return torch.argmax(action_values).item()

    def step(self, obs, action, reward, next_obs, done):
        self.memory.store_transition(obs, action, reward, next_obs, done)

        self.t_step = (self.t_step + 1) % self.hyperparams['update_every']
        if self.t_step == 0 and len(self.memory) > self.hyperparams['batch_size']:
            experiences = self.memory.sample(self.hyperparams['batch_size'])
            self._learn(experiences)

        if done:
            self._update_eps()

    def _update_eps(self):
        self.eps = max(self.hyperparams['eps_end'], self.hyperparams['eps_decay'] * self.eps)

    def _learn(self, experiences):
        observations, actions, rewards, next_observations, dones = experiences

        Q_current = self.Q_network(observations).gather(1, actions)

        with torch.no_grad():
            a = self.Q_network(next_observations).argmax(1).unsqueeze(1)
            Q_target_next = self.target_network(next_observations).gather(1, a)
            Q_target = rewards + self.hyperparams['gamma'] * Q_target_next * (1 - dones)

        loss = F.mse_loss(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % self.hyperparams['sync_target_every'] == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())
