import torch
import json
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from .model import QNetwork
from .memory import Memory
from copy import deepcopy


class DQNAgent:
    def __init__(self, state_dim, action_dim, goal_dim, compute_reward_func):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.compute_reward_func = compute_reward_func

        self.hyperparams = self._read_hyperparams()['local_agent_discrete']

        self.eps = self.hyperparams['eps_start']
        self.k_future = self.hyperparams['k_future']

        self.model = QNetwork(self.state_dim, self.action_dim, goal_dim=self.goal_dim,
                                    hidden_1=self.hyperparams['hidden_1'], hidden_2=self.hyperparams['hidden_2'],
                                    hidden_3=self.hyperparams['hidden_3'])
        self.model_target = deepcopy(self.model)
        self.tau = self.hyperparams['tau']
        self.gamma = self.hyperparams['gamma']

        self.capacity = self.hyperparams['buffer_size']
        self.memory = Memory(self.capacity, self.k_future, self.compute_reward_func)

        self.batch_size = self.hyperparams['batch_size']
        self.lr = self.hyperparams['lr']
        self.model_optimizer = Adam(self.model.parameters(), self.lr)

    def _read_hyperparams(self):
        with open('hyperparams.json') as f:
            hyperparams = json.load(f)
            return hyperparams

    def act(self, state, goal, train_mode=True):
        if train_mode and np.random.rand() < self.eps:
            action = np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                x = np.concatenate([state, goal])
                obs = torch.from_numpy(x).float()
                action = self.model(obs).numpy().argmax()

        return action

    def update_eps(self):
        self.eps = max(self.hyperparams['eps_end'], self.hyperparams['eps_decay'] * self.eps)

    def store(self, episode_dict):
        self.memory.add(episode_dict)

    @staticmethod
    def soft_update_networks(local_model, target_model, tau=0.05):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    def train(self):
        states, actions, rewards, next_states, goals = self.memory.sample(self.batch_size)

        inputs = np.concatenate([states, goals], axis=1)
        next_inputs = np.concatenate([next_states, goals], axis=1)

        inputs = torch.Tensor(inputs).float()
        rewards = torch.Tensor(rewards).float()
        next_inputs = torch.Tensor(next_inputs).float()
        actions = torch.Tensor(actions).long()

        Q_current = self.model(inputs).gather(1, actions)
        with torch.no_grad():
            a = self.model(next_inputs).argmax(1).unsqueeze(1)
            Q_target_next = self.model_target(next_inputs).gather(1, a)
            Q_target = rewards + self.gamma * Q_target_next

        loss = F.mse_loss(Q_current, Q_target)

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def load_global_weights(self, global_agent):
        self.model.load_state_dict(global_agent.model.state_dict())
        self.model_target = deepcopy(self.model)

    def save_weights(self, name):
        torch.save({"state_dict": self.model.state_dict()}, f"weights/{name}.pth")

    def load_weights(self, name):
        checkpoint = torch.load(f"weights/{name}.pth")
        state_dict = checkpoint["actor_state_dict"]
        self.model.load_state_dict(state_dict)

    def update_networks(self):
        self.soft_update_networks(self.model, self.model_target, self.tau)
