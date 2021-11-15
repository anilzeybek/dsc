from ddpg.ddpg_agent import DDPGAgent
from classifier import Classifier


class Option:
    def __init__(self, budget: int, env, parent_option, N, K) -> None:
        self.budget = budget
        self.env = env
        self.parent_option = parent_option
        self.N = N
        self.K = K

        self.agent = DDPGAgent()
        self.initiation_classifier = Classifier()
        if parent_option:
            self.termination_classifier = parent_option.initiation_classifier
        else:
            self.termination_classifier = Classifier(goal_or_global=True, env_termination_checker=env._task.termination)

        self.successful_observations_to_train = []
        self.initiation_classifier_trained = False

    def execute(self, obs):
        # TODO: check if the self.env is same across everywhere
        t = 0
        reward_list = []

        local_done = False
        while not local_done and t < self.budget:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            reward_list.append(reward)

            local_done = self.termination_classifier.check(next_obs)
            local_reward = 1 if local_done else 0

            self.agent.step(obs, action, local_reward, next_obs, local_done)
            obs = next_obs
            t += 1

        successful_observations = None
        if local_done:
            successful_observations = self.agent.replay_buffer.memory[-self.K-2:-self.K+2]

        return obs, reward_list, done, successful_observations

    def learn_initiation_classifier(self, successful_observations) -> None:
        self.successful_observations_to_train.append(successful_observations)
        if len(self.successful_observations_to_train) == self.N:
            self.initiation_classifier.train_one_class(self.successful_observations_to_train)

        # TODO: 2-class?
