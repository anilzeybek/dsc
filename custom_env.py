from gym import spaces
import numpy as np
from copy import deepcopy


class CustomEnv:
    def __init__(self):
        self.observation_space = {
            "observation": spaces.Box(0.0, np.inf, (3,), np.float32),
            "achieved_goal": spaces.Box(0.0, np.inf, (3,), np.float32),
            "desired_goal": spaces.Box(0.0, np.inf, (3,), np.float32),
        }
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

        self.start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.goal_pos = np.array([20.0, 30.0, 40.0], dtype=np.float32)
        self.current_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.step_count = 0
        self.max_step_count = 100

    def compute_reward(self, achieved, desired, _):
        return (1 * self._is_close(achieved, desired)) - 1

    def reset(self):
        self.step_count = 0
        self.current_pos = deepcopy(self.start_pos)
        return self.env_info()

    def termination(self, obs):
        if self._is_close(obs, self.goal_pos):
            return True

        return False

    def env_info(self):
        return {
            "observation": deepcopy(self.current_pos),
            "achieved_goal": deepcopy(self.current_pos),
            "desired_goal": deepcopy(self.goal_pos)
        }

    def _is_close(self, pos1, pos2):
        if pos1.ndim == 1 and pos2.ndim == 1:
            pos1 = np.expand_dims(pos1, axis=0)
            pos2 = np.expand_dims(pos2, axis=0)

        return np.linalg.norm(pos1 - pos2, axis=1) < 1

    def step(self, action):
        assert self.action_space.contains(action), "action is not valid"
        self.step_count += 1

        self.current_pos[0] += action[0]
        if self.current_pos[0] < 0:
            self.current_pos[0] = 0

        self.current_pos[1] += action[1]
        if self.current_pos[1] < 0:
            self.current_pos[1] = 0

        self.current_pos[2] += action[2]
        if self.current_pos[2] < 0:
            self.current_pos[2] = 0

        goal_achieved = self.termination(self.current_pos)
        reward = 0 if goal_achieved else -1

        done = self.step_count >= self.max_step_count
        return self.env_info(), reward, done, {}

    def render(self):
        pass
