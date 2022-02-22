import torch
import json
import numpy as np
import torch.optim as optim
from .models import Actor, Critic
from copy import deepcopy
from cpprb import ReplayBuffer
import torch.nn.functional as F


class TD3Agent:
    def __init__(self, obs_dim, action_dim, goal_dim, action_bounds, compute_reward_func):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.action_bounds = action_bounds
        self.compute_reward_func = compute_reward_func

        self.hyperparams = self._read_hyperparams()['local_agent_td3']

        self.actor = Actor(self.obs_dim, action_dim=self.action_dim, goal_dim=self.goal_dim,
                           hidden_1=self.hyperparams['hidden_1'], hidden_2=self.hyperparams['hidden_2'],
                           action_bounds_high=action_bounds['high'])
        self.actor_target = deepcopy(self.actor)


        self.critic = Critic(self.obs_dim, action_dim=self.action_dim, goal_dim=self.goal_dim,
                             hidden_1=self.hyperparams['hidden_1'], hidden_2=self.hyperparams['hidden_2'])
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hyperparams['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hyperparams['critic_lr'])

        self.rb = ReplayBuffer(self.hyperparams['buffer_size'], env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "goal": {"shape": self.goal_dim}
        })
        self.total_it = 0

    @staticmethod
    def _read_hyperparams():
        with open('hyperparams.json') as f:
            hyperparams = json.load(f)
            return hyperparams

    def act(self, obs, goal, train_mode=True):
        with torch.no_grad():
            x = torch.from_numpy(np.concatenate([obs, goal])).float()
            action = self.actor(x).numpy()

        if train_mode:
            action += self.action_bounds['high'] / 5 * np.random.randn(self.action_dim)
            action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])

            random_actions = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                               size=self.action_dim)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def store(self, episode_dict):
        episode_len = len(episode_dict['obs'])
        for t in range(episode_len):
            obs = episode_dict['obs'][t]
            action = episode_dict['action'][t]
            reward = episode_dict['reward'][t]
            next_obs = episode_dict['next_obs'][t]
            goal = episode_dict['desired_goal'][t]
            next_achieved = episode_dict['next_achieved_goal'][t]

            self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, goal=goal)

            if episode_len > 1:
                for _ in range(self.hyperparams['k_future']):
                    future_idx = np.random.randint(low=t, high=episode_len)

                    new_goal = episode_dict['achieved_goal'][future_idx]
                    new_reward = self.compute_reward_func(next_achieved, new_goal, None)[0]

                    self.rb.add(obs=obs, action=action, reward=new_reward, next_obs=next_obs, goal=new_goal)

        self.rb.on_episode_end()

    def train(self):
        try:
            sample = self.rb.sample(self.hyperparams['batch_size'])
        except ValueError:
            # means not enough sample
            return

        self.total_it += 1

        input = np.concatenate([sample['obs'], sample['goal']], axis=1)
        next_input = np.concatenate([sample['next_obs'], sample['goal']], axis=1)
        done = sample['reward'] + 1

        input = torch.Tensor(input)
        action = torch.Tensor(sample['action'])
        reward = torch.Tensor(sample['reward'])
        next_input = torch.Tensor(next_input)
        done = torch.Tensor(done).long()

        q_current1, q_current2 = self.critic(input, action)
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.hyperparams['policy_noise']
            ).clamp(-self.hyperparams['noise_clip'], self.hyperparams['noise_clip'])

            next_actions = (
                self.actor_target(next_input) + noise
            ).clamp(torch.from_numpy(self.action_bounds['low']), torch.from_numpy(self.action_bounds['high']))

            q1_target_next, q2_target_next = self.critic_target(next_input, next_actions)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            q_target = reward + self.hyperparams['gamma'] * q_target_next * (1 - done)

        critic_loss = F.mse_loss(q_current1, q_target) + F.mse_loss(q_current2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.hyperparams['policy_freq'] == 0:
            actor_loss = -self.critic(input, self.actor(input))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.hyperparams['tau'] * param.data +
                                        (1 - self.hyperparams['tau']) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.hyperparams['tau'] * param.data +
                                        (1 - self.hyperparams['tau']) * target_param.data)

    def load_global_weights(self, global_agent_actor, global_agent_critic):
        self.actor.load_state_dict(global_agent_actor.state_dict())
        self.critic.load_state_dict(global_agent_critic.state_dict())

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
