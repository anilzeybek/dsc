from typing import Any, Dict
import numpy as np
import torch
import torch.optim as optim
import json
from .model import QNetwork, PolicyNetwork
from .replay_buffer import ReplayBuffer
from copy import deepcopy


class DDPGAgent:
    def __init__(self, obs_dim, act_dim, act_limits):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limits = act_limits
        self.hyperparams = self.read_hyperparams()['local_agent']
        self.t = 0

        self.actor_network = PolicyNetwork(obs_dim, act_dim, act_limits, self.hyperparams["hidden_1"], self.hyperparams["hidden_2"])
        self.actor_target = deepcopy(self.actor_network)

        self.critic_network = QNetwork(obs_dim, act_dim, self.hyperparams["hidden_1"], self.hyperparams["hidden_2"])
        self.critic_target = deepcopy(self.critic_network)

        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.hyperparams["buffer_size"])

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.hyperparams["lr_actor"])
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.hyperparams["lr_critic"])

    def read_hyperparams(self) -> Dict[str, Any]:
        with open('hyperparams.json') as f:
            hyperparams = json.load(f)
            return hyperparams

    def act(self, obs, noise=0.1):
        if self.t > self.hyperparams["start_steps"]:
            with torch.no_grad():
                a = self.actor_network(torch.as_tensor(obs, dtype=torch.float32)).numpy()
                a += noise * np.random.randn(self.act_dim)
                return np.clip(a, -self.act_limits, self.act_limits)
        else:
            return np.random.uniform(low=-self.act_limits, high=self.act_limits, size=(self.act_dim,))

    def step(self, obs, action, reward, next_obs, done):
        self.t += 1
        self.replay_buffer.store_transition(obs, action, reward, next_obs, done)

        if self.t >= self.hyperparams["update_after"] and self.t % self.hyperparams["update_every"] == 0:
            for _ in range(self.hyperparams["update_every"]):
                batch = self.replay_buffer.sample(self.hyperparams["batch_size"])
                self._learn(data=batch)

    def load_global_weights(self, global_actor_network, global_critic_network):
        self.actor_network.load_state_dict(global_actor_network.state_dict())
        self.actor_target = deepcopy(self.actor_network)

        self.critic_network.load_state_dict(global_critic_network.state_dict())
        self.critic_target = deepcopy(self.critic_network)

    def _compute_loss_q(self, data):
        obs, action, reward, next_obs, done = data
        Q_current = self.critic_network(obs, action)

        with torch.no_grad():
            Q_target_next = self.critic_target(next_obs, self.actor_target(next_obs))
            Q_target = reward + self.hyperparams["gamma"] * Q_target_next * (1 - done)

        loss = ((Q_current - Q_target) ** 2).mean()
        return loss

    def _compute_loss_pi(self, data):
        obs = data[0]
        Q = self.critic_network(obs, self.actor_network(obs))

        return -Q.mean()

    def _learn(self, data):
        self.critic_optimizer.zero_grad()
        loss_Q = self._compute_loss_q(data)
        loss_Q.backward()
        self.critic_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.critic_network.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        loss_pi = self._compute_loss_pi(data)
        loss_pi.backward()
        self.actor_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.critic_network.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_network.parameters(), self.actor_target.parameters()):
                p_targ.data.mul_(self.hyperparams["polyak"])
                p_targ.data.add_((1 - self.hyperparams["polyak"]) * p.data)

            for p, p_targ in zip(self.critic_network.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(self.hyperparams["polyak"])
                p_targ.data.add_((1 - self.hyperparams["polyak"]) * p.data)
