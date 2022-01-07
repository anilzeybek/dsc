from typing import List, Callable, Dict, Any
import torch
import json
import numpy as np
from torch.optim import Adam
from .models import Actor, Critic
from copy import deepcopy
from torch import nn
from cpprb import ReplayBuffer


class DDPGAgent:
    def __init__(self, obs_dim: int, action_dim: int, goal_dim: int, action_bounds: List[float],
                 compute_reward_func: Callable) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.action_bounds = action_bounds
        self.compute_reward_func = compute_reward_func

        self.hyperparams = self._read_hyperparams()['local_agent_continuous']

        self.actor = Actor(self.obs_dim, action_dim=self.action_dim, goal_dim=self.goal_dim,
                           hidden_1=self.hyperparams['hidden_1'], hidden_2=self.hyperparams['hidden_2'],
                           action_bounds=self.action_bounds)
        self.critic = Critic(self.obs_dim, action_dim=self.action_dim, goal_dim=self.goal_dim,
                             hidden_1=self.hyperparams['hidden_1'], hidden_2=self.hyperparams['hidden_2'])
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), self.hyperparams['actor_lr'])
        self.critic_optimizer = Adam(self.critic.parameters(), self.hyperparams['critic_lr'])

        self.rb = ReplayBuffer(self.hyperparams['buffer_size'], env_dict={
            "obs": {"shape": obs_dim},
            "action": {"shape": action_dim},
            "reward": {},
            "next_obs": {"shape": obs_dim},
            "goal": {"shape": goal_dim},
        })
        self.store_count = 0

    @staticmethod
    def _read_hyperparams() -> Dict[str, Any]:
        with open('hyperparams.json') as f:
            hyperparams = json.load(f)
            return hyperparams

    def act(self, obs: np.ndarray, goal: np.ndarray, train_mode=True) -> np.ndarray:
        obs = np.expand_dims(obs, axis=0)
        goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([obs, goal], axis=1)
            x = torch.from_numpy(x).float()
            action = self.actor(x)[0].numpy()

        if train_mode:
            action += self.action_bounds[1] / 5 * np.random.randn(self.action_dim)
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

            random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                               size=self.action_dim)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return action

    def store(self, episode_dict: Dict[str, List[np.ndarray]]) -> None:
        # TODO: slow, fix it
        episode_len = len(episode_dict['obs'])
        for t in range(episode_len):
            obs = episode_dict['obs'][t]
            action = episode_dict['action'][t]
            reward = episode_dict['reward'][t]
            next_obs = episode_dict['next_obs'][t]
            goal = episode_dict['desired_goal'][t]
            next_achieved = episode_dict['next_achieved_goal'][t]

            self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, goal=goal)
            self.store_count += 1
            for _ in range(self.hyperparams['k_future']):
                future_idx = np.random.randint(low=t, high=episode_len)
                new_goal = episode_dict['next_achieved_goal'][future_idx]
                new_reward = self.compute_reward_func(next_achieved, new_goal, None)[0]
                self.rb.add(obs=obs, action=action, reward=new_reward, next_obs=next_obs, goal=new_goal)
                self.store_count += 1

        if self.store_count >= 250:
            self.rb.on_episode_end()
            self.store_count = 0

    @staticmethod
    def soft_update_networks(local_model: nn.Module, target_model: nn.Module, tau=0.05) -> None:
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    def train(self) -> None:
        sample = self.rb.sample(self.hyperparams['batch_size'])

        inputs = np.concatenate([sample['obs'], sample['goal']], axis=1)
        next_inputs = np.concatenate([sample['next_obs'], sample['goal']], axis=1)
        dones = sample['reward'] + 1

        inputs_ = torch.Tensor(inputs)
        rewards_ = torch.Tensor(sample['reward'])
        next_inputs_ = torch.Tensor(next_inputs)
        actions_ = torch.Tensor(sample['action'])
        dones_ = torch.Tensor(dones).long()

        with torch.no_grad():
            target_q = self.critic_target(next_inputs_, self.actor_target(next_inputs_))
            target_returns = rewards_ + self.hyperparams['gamma'] * target_q * (1 - dones_)
            target_returns = torch.clamp(target_returns, -1 / (1 - self.hyperparams['gamma']), 0)

        q_eval = self.critic(inputs_, actions_)
        critic_loss = (target_returns - q_eval).pow(2).mean()

        a = self.actor(inputs_)
        actor_loss = -self.critic(inputs_, a).mean()
        actor_loss += a.pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def load_global_weights(self, global_agent_actor: Actor, global_agent_critic: Critic) -> None:
        self.actor.load_state_dict(global_agent_actor.state_dict())
        self.critic.load_state_dict(global_agent_critic.state_dict())

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def update_networks(self) -> None:
        self.soft_update_networks(self.actor, self.actor_target, self.hyperparams['tau'])
        self.soft_update_networks(self.critic, self.critic_target, self.hyperparams['tau'])
