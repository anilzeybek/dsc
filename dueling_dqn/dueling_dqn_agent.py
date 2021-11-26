import torch
import json
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from .model import DuelingNetwork
from .memory import Memory
from copy import deepcopy


class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, goal_dim, compute_reward_func):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.compute_reward_func = compute_reward_func

        self.hyperparams = self._read_hyperparams()['local_agent_discrete']

        self.k_future = self.hyperparams['k_future']

        self.model = DuelingNetwork(self.state_dim, self.action_dim, goal_dim=self.goal_dim,
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
        state = np.expand_dims(state, axis=0)
        goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([state, goal], axis=1)
            x = torch.from_numpy(x).float()
            action = self.model(x)[0].numpy()

        if train_mode:
            # TODO: here will change
            action += self.action_bounds[1] / 5 * np.random.randn(self.action_dim)
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

            random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                               size=self.action_dim)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return action

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

        inputs = torch.Tensor(inputs)
        rewards = torch.Tensor(rewards)
        next_inputs = torch.Tensor(next_inputs)
        actions = torch.Tensor(actions)

        Q_current = self.model(inputs).gather(1, actions)
        with torch.no_grad():
            a = self.model(next_states).argmax(1).unsqueeze(1)
            Q_target_next = self.model_target(next_inputs).gather(1, a)
            Q_target = rewards + self.gamma * Q_target_next

        loss = F.mse_loss(Q_current, Q_target)

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def load_global_weights(self, global_network):
        self.model.load_state_dict(global_network.state_dict())
        self.model_target = deepcopy(self.model)

    def save_weights(self, name):
        torch.save({"state_dict": self.model.state_dict()}, f"weights/{name}.pth")

    def load_weights(self, name):
        checkpoint = torch.load(f"weights/{name}.pth")
        state_dict = checkpoint["actor_state_dict"]
        self.model.load_state_dict(state_dict)

    def update_networks(self):
        self.soft_update_networks(self.model, self.model_target, self.tau)
