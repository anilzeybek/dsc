import torch
import json
import numpy as np
from torch.optim import Adam
from .models import Actor, Critic
from .memory import Memory
from copy import deepcopy


class DDPGAgent:
    def __init__(self, state_dim, action_dim, goal_dim, action_bounds, compute_reward_func):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.action_bounds = action_bounds
        self.compute_reward_func = compute_reward_func

        self.hyperparams = self._read_hyperparams()['local_agent']

        self.k_future = self.hyperparams['k_future']

        self.actor = Actor(self.state_dim, action_dim=self.action_dim, goal_dim=self.goal_dim, action_bounds=self.action_bounds)
        self.critic = Critic(self.state_dim, action_size=self.action_dim, goal_dim=self.goal_dim)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.tau = self.hyperparams['tau']
        self.gamma = self.hyperparams['gamma']

        self.capacity = self.hyperparams['buffer_size']
        self.memory = Memory(self.capacity, self.k_future, self.compute_reward_func)

        self.batch_size = self.hyperparams['batch_size']
        self.actor_lr = self.hyperparams['actor_lr']
        self.critic_lr = self.hyperparams['critic_lr']
        self.actor_optimizer = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), self.critic_lr)

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
            action = self.actor(x)[0].numpy()

        if train_mode:
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

        with torch.no_grad():
            target_q = self.critic_target(next_inputs, self.actor_target(next_inputs))
            target_returns = rewards + self.gamma * target_q
            target_returns = torch.clamp(target_returns, -1 / (1 - self.gamma), 0)

        q_eval = self.critic(inputs, actions)
        critic_loss = (target_returns - q_eval).pow(2).mean()

        a = self.actor(inputs)
        actor_loss = -self.critic(inputs, a).mean()
        actor_loss += a.pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save_weights(self, env_name):
        torch.save({"actor_state_dict": self.actor.state_dict()}, f"weights/{env_name}.pth")

    def load_weights(self, env_name):
        checkpoint = torch.load(f"weights/{env_name}.pth")
        actor_state_dict = checkpoint["actor_state_dict"]
        self.actor.load_state_dict(actor_state_dict)

    def update_networks(self):
        self.soft_update_networks(self.actor, self.actor_target, self.tau)
        self.soft_update_networks(self.critic, self.critic_target, self.tau)
