from collections import namedtuple, deque
import numpy as np
import random
import torch


class ReplayBuffer:
    def __init__(self, buffer_size) -> None:
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["obs", "action", "reward_list", "next_obs", "done"])

    def store_transition(self, obs, action, reward_list, next_obs, done) -> None:
        e = self.experience(obs, action, reward_list, next_obs, done)
        self.memory.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)

        observations = torch.from_numpy(np.vstack([e.obs for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        # TODO: this can be wrong, because rewards are list now
        rewards = torch.from_numpy(np.vstack([e.reward_list for e in experiences if e is not None])).float()
        next_observations = torch.from_numpy(np.vstack([e.next_obs for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return observations, actions, rewards, next_observations, dones

    def __len__(self) -> int:
        return len(self.memory)
