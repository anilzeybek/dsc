from ddpg.ddpg_agent import DDPGAgent
from classifier import Classifier


class Option:
    def __init__(self, budget: int, env, this_is_global_option, this_is_goal_option, parent_option, max_refine, N, K) -> None:
        self.budget = budget
        self.env = env
        self.this_is_global_option = this_is_global_option
        self.this_is_goal_option = this_is_goal_option
        self.parent_option = parent_option
        self.max_refine = max_refine
        self.N = N
        self.K = K

        # it cant be both global and goal
        assert not (this_is_global_option and this_is_goal_option)

        if self.this_is_global_option or self.this_is_goal_option:
            assert parent_option is None

        self.agent = DDPGAgent()
        if parent_option:
            assert parent_option.initiation_classifier_created

            self.termination_classifier = parent_option.initiation_classifier
            # TODO: böylece type değiştiremeyiz, parent'ınki de değişir, ama deepcopy de olmaz çünkü parent init değişirse self term de değişsin istiyoruz
            self.termination_classifier.type_ = "termination"
        else:
            # This means self is either goal or global option
            self.termination_classifier = Classifier(type_="termination", for_global_option=self.this_is_global_option,
                                                     for_goal_option=self.this_is_goal_option, env_termination_checker=env._task.termination)

        self.initiation_classifier = Classifier(type_="initiation")
        self.initiation_classifier_created = False
        self.initiation_classifier_refined = False

        if self.this_is_global_option:
            self.initiation_classifier_created = True
            self.initiation_classifier_refined = True

        self.successful_observations_to_create_initiation_classifier = []
        self.refine_count = 0

    def execute(self, obs):
        assert self.initiation_classifier_created

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

        # those successful_observations are not for self, it is for option_without_initiation_classifier
        successful_observations = None
        if local_done:
            successful_observations = self.agent.replay_buffer.memory[-self.K-2:-self.K+2]

        return obs, reward_list, done, successful_observations

    def create_initiation_classifier(self, successful_observations) -> None:
        assert not self.initiation_classifier_created

        self.successful_observations_to_create_initiation_classifier.append(successful_observations)
        if len(self.successful_observations_to_create_initiation_classifier) == self.N:
            self.initiation_classifier.train_one_class(self.successful_observations_to_create_initiation_classifier)
            self.initiation_classifier_created = True

    def refine_inititation_classifier(self, positive_examples, negative_examples):
        assert self.initiation_classifier_created

        # TODO: this class should called by self
        if not self.initiation_classifier_refined:
            # TODO:

            self.refine_count += 1
            if self.refine_count == self.max_refine:
                self.initiation_classifier_refined = True
