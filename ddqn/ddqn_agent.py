import json
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from option import Option
from .model import QNetwork
from .replay_buffer import ReplayBuffer
from copy import deepcopy


class DDQNAgent:
    def __init__(self, obs_size, option_repertoire) -> None:
        self.obs_size = obs_size
        self.action_size = len(option_repertoire)
        self.option_repertoire = option_repertoire
        self.hyperparams = self.read_hyperparams()['agent_over_actions']

        self.Q_network = QNetwork(self.obs_size, self.action_size, self.hyperparams['hidden_1'], self.hyperparams['hidden_2'])
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

    def act(self, obs) -> Option:
        selectable_options = map(lambda o: o.initiation_classifier.check(obs), self.option_repertoire)
        selectable_indexes = np.argwhere(selectable_options)

        if np.random.rand() < self.eps:
            selected_index = np.random.choice(selectable_indexes)
        else:
            with torch.no_grad():
                obs = torch.from_numpy(obs)
                action_values = self.Q_network(obs).numpy()
                action_values[selectable_indexes] = -np.inf

                selected_index = action_values.argmax()

        return self.option_repertoire[selected_index]

    def step(self, obs, action, reward_list, next_obs, done) -> None:
        self.memory.store_transition(obs, action, reward_list, next_obs, done)

        self.t_step = (self.t_step + 1) % self.hyperparams['update_every']
        if self.t_step == 0 and len(self.memory) > self.hyperparams['batch_size']:
            experiences = self.memory.sample(self.hyperparams['batch_size'])
            self._learn(experiences)

        if done:
            self._update_eps()

    def _update_eps(self) -> None:
        self.eps = max(self.hyperparams['eps_end'], self.hyperparams['eps_decay'] * self.eps)

    def _learn(self, experiences) -> None:
        observations, actions, rewards_lists, next_observations, dones = experiences
        # TODO: rewards_lists definetly wrong

        Q_current = self.Q_network(observations).gather(1, actions)
        with torch.no_grad():
            a = self.Q_network(next_observations).argmax(1)
            Q_target_next = self.target_network(next_observations).gather(1, a)

            discounted_reward = 0
            for i in range(len(rewards_lists)):
                discounted_reward += (self.hyperparams['gamma'] ** i) * rewards_lists[i]

            Q_target = discounted_reward + (self.hyperparams['gamma'] ** len(rewards_lists)) * Q_target_next * (1 - dones)

        loss = F.mse_loss(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % self.hyperparams['sync_target_every'] == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())
