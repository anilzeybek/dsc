import json
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from option import Option
from .model import QNetwork
from cpprb import ReplayBuffer
from copy import deepcopy


class MetaDQNAgent:
    def __init__(self, obs_dim, action_dim) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hyperparams = self._read_hyperparams()['agent_over_options']

        self.Q_network = QNetwork(self.obs_dim, self.action_dim,
                                  self.hyperparams['hidden_1'], self.hyperparams['hidden_2'])
        self.target_network = deepcopy(self.Q_network)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.hyperparams['lr'])

        self.eps = self.hyperparams['eps_start']
        self.rb = ReplayBuffer(self.hyperparams['buffer_size'], env_dict={
            "obs": {"shape": obs_dim},
            "action": {"shape": action_dim},
            "discounted_reward": {},
            "next_obs": {"shape": obs_dim},
            "length": {},
            "done": {},
        })
        self.t_step = 0
        self.learn_count = 0

    def add_option(self) -> None:
        self.action_dim += 1
        self.Q_network.change_last_layer(self.action_dim)
        self.target_network = deepcopy(self.Q_network)

    @staticmethod
    def _read_hyperparams() -> Dict[str, Any]:
        with open('hyperparams.json') as file:
            hyperparams = json.load(file)
            return hyperparams

    def act(self, obs: np.ndarray, option_repertoire: List[Option], train_mode=True) -> int:
        selectable_indexes = []
        for i, o in enumerate(option_repertoire):
            if o.init_classifier.check(obs):
                selectable_indexes.append(i)

        # TODO: following is not feasible (removing the global from selectable) for optimistic init of options
        if len(selectable_indexes) > 1:
            selectable_indexes = selectable_indexes[1:]

        if np.random.rand() < self.eps and train_mode:
            selected_index = np.random.choice(selectable_indexes)
        else:
            with torch.no_grad():
                obs = torch.from_numpy(obs).float()
                action_values = self.Q_network(obs).numpy()
                action_values[np.setdiff1d(np.arange(len(action_values)), selectable_indexes)] = -np.inf

                selected_index = action_values.argmax()

        return selected_index

    def step(self, obs: np.ndarray, action: int, reward_list: List[float], next_obs: np.ndarray, done: bool) -> None:
        discounted_reward = 0
        for i, reward in enumerate(reward_list):
            discounted_reward += (self.hyperparams['gamma'] ** (i + 1)) * reward

        self.rb.add(obs=obs, action=action, discounted_reward=discounted_reward,
                    next_obs=next_obs, done=done, length=len(reward_list))

        self.t_step = (self.t_step + 1) % self.hyperparams['update_every']
        if self.t_step == 0:
            self._train()
        if done:
            self._update_eps()

    def _update_eps(self) -> None:
        self.eps = max(self.hyperparams['eps_end'], self.hyperparams['eps_decay'] * self.eps)

    def _train(self) -> None:
        sample = self.rb.sample(self.hyperparams['batch_size'])

        observations = torch.Tensor(sample['obs'])
        actions = torch.Tensor(sample['action']).long()
        discounted_rewards = torch.Tensor(sample['discounted_reward'])
        next_observations = torch.Tensor(sample['next_obs'])
        lengths = torch.Tensor(sample['length']).long()
        dones = torch.Tensor(sample['done']).long()

        q_current = self.Q_network(observations).gather(1, actions)
        with torch.no_grad():
            a = self.Q_network(next_observations).argmax(1).unsqueeze(1)
            q_target_next = self.target_network(next_observations).gather(1, a)

            q_target = discounted_rewards + torch.pow(self.hyperparams['gamma'], lengths) * q_target_next * (1 - dones)

        loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % self.hyperparams['sync_target_every'] == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())

    def save(self) -> None:
        torch.save(self.Q_network.state_dict(), "./saved_trainings/agent_over_options.pth")

    def load(self) -> None:
        self.Q_network.load_state_dict(torch.load("./saved_trainings/agent_over_options.pth"))
        self.eps = 0  # since we load, it should be 0 because we are not training anymore
