from ddpg.ddpg_agent import DDPGAgent
from classifier import Classifier


class Option:
    def __init__(self, budget: int, env, parent) -> None:
        self.budget = budget
        self.env = env
        self.parent = parent

        self.initiation_classifier = Classifier()
        self.agent = DDPGAgent()

    def execute(self, obs):
        # TODO: check if the self.env is same across everywhere
        t = 0
        rewards = []

        done = False
        while not done and t < self.budget:
            action = self.agent.act(obs)
            next_obs, _, _, _ = self.env.step(action)

            done = self.initiation_classifier.check(next_obs)
            reward = 1 if done else 0
            rewards.append(reward)

            self.agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            t += 1

        return rewards, obs
