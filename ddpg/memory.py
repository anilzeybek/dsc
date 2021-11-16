import numpy as np
from copy import deepcopy


class Memory:
    def __init__(self, capacity, k_future, compute_reward_func):
        self.capacity = capacity
        self.memory = []
        self.compute_reward_func = compute_reward_func

        self.future_p = 1 - (1. / (1 + k_future))

    def sample(self, batch_size):
        ep_indices = np.random.randint(0, len(self.memory), batch_size)
        time_indices = np.random.randint(0, len(self.memory[0]["next_state"]), batch_size)
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []

        for episode, timestep in zip(ep_indices, time_indices):
            states.append(deepcopy(self.memory[episode]["state"][timestep]))
            actions.append(deepcopy(self.memory[episode]["action"][timestep]))
            desired_goals.append(deepcopy(self.memory[episode]["desired_goal"][timestep]))
            next_achieved_goals.append(deepcopy(self.memory[episode]["next_achieved_goal"][timestep]))
            next_states.append(deepcopy(self.memory[episode]["next_state"][timestep]))

        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (len(self.memory[0]["next_state"]) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + 1 + future_offset)[her_indices]

        future_ag = []
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(deepcopy(self.memory[episode]["achieved_goal"][f_offset]))
        future_ag = np.vstack(future_ag)

        desired_goals[her_indices] = future_ag
        rewards = np.expand_dims(self.compute_reward_func(next_achieved_goals, desired_goals, None), 1)

        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)
