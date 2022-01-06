from typing import Callable, Dict, List, Tuple
import numpy as np
from copy import deepcopy


class Memory:
    def __init__(self, capacity: int, k_future: int, compute_reward_func: Callable) -> None:
        self.capacity = capacity
        self.memory: List[Dict[str, List[np.ndarray]]] = []
        self.compute_reward_func = compute_reward_func

        self.future_p = 1 - (1. / (1 + k_future))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ep_indices = np.random.randint(0, len(self.memory), batch_size)
        time_indices = np.random.randint(0, len(self.memory[0]["next_obs"]), batch_size)
        obs_s = []
        actions = []
        desired_goals = []
        next_obs_s = []
        next_achieved_goals = []

        for episode, timestep in zip(ep_indices, time_indices):
            obs_s.append(deepcopy(self.memory[episode]["obs"][timestep]))
            actions.append(deepcopy(self.memory[episode]["action"][timestep]))
            desired_goals.append(deepcopy(self.memory[episode]["desired_goal"][timestep]))
            next_achieved_goals.append(deepcopy(self.memory[episode]["next_achieved_goal"][timestep]))
            next_obs_s.append(deepcopy(self.memory[episode]["next_obs"][timestep]))

        obs_s_ = np.vstack(obs_s)
        actions_ = np.vstack(actions)
        desired_goals_ = np.vstack(desired_goals)
        next_achieved_goals_ = np.vstack(next_achieved_goals)
        next_obs_s_ = np.vstack(next_obs_s)

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (len(self.memory[0]["next_obs"]) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + 1 + future_offset)[her_indices]

        future_ag = []
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(deepcopy(self.memory[episode]["achieved_goal"][f_offset]))
        future_ag_ = np.vstack(future_ag)

        desired_goals_[her_indices] = future_ag_
        rewards = np.expand_dims(self.compute_reward_func(next_achieved_goals_, desired_goals_, None), 1)

        return self.clip_obs(obs_s_), actions_, rewards, self.clip_obs(next_obs_s_), self.clip_obs(desired_goals_)

    def add(self, transition: Dict[str, List[np.ndarray]]) -> None:
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def __len__(self) -> int:
        return len(self.memory)

    @staticmethod
    def clip_obs(x: np.ndarray) -> np.ndarray:
        return np.clip(x, -300, 300)
