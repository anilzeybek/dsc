from copy import deepcopy
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

        # TODO: act limits check
        self.agent = DDPGAgent(self.env.observation_space.shape[0], self.env.action_space.shape[0], 1)
        if parent_option:
            assert parent_option.initiation_classifier_created

            self.termination_classifier = deepcopy(parent_option.initiation_classifier)
            self.termination_classifier.type_ = "termination"

            self.initiation_classifier = Classifier(type_="initiation")
        else:
            # This means self is either goal or global option
            # TODO: env_term_checker should be env._task.termination
            self.termination_classifier = Classifier(type_="termination", for_global_option=self.this_is_global_option,
                                                     for_goal_option=self.this_is_goal_option, env_termination_checker=lambda x: True)

            self.initiation_classifier = Classifier(type_="initiation", for_global_option=self.this_is_global_option, for_goal_option=self.this_is_goal_option)
        self.initiation_classifier_created = False
        self.initiation_classifier_refined = False

        if self.this_is_global_option:
            self.initiation_classifier_created = True
            self.initiation_classifier_refined = True

        self.successful_observations_to_create_initiation_classifier = []
        self.good_examples_to_refine = []
        self.bad_examples_to_refine = []
        self.refine_count = 0

    def execute(self, obs):
        assert self.initiation_classifier_created

        starting_obs = deepcopy(obs)
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
            self.good_examples_to_refine.append(starting_obs)
            successful_observations = self.agent.replay_buffer.memory[-self.K-2:-self.K+2]
        else:
            self.bad_examples_to_refine.append(starting_obs)

        if not self.initiation_classifier_refined and len(self.good_examples_to_refine) > self.max_refine:
            self.refine_inititation_classifier()
        return obs, reward_list, done, successful_observations

    def create_initiation_classifier(self, successful_observations) -> None:
        # there shouldn't be any actions for initiation classifier if we call this function
        assert not self.initiation_classifier_created
        assert not self.initiation_classifier_refined

        self.successful_observations_to_create_initiation_classifier.append(successful_observations)
        if len(self.successful_observations_to_create_initiation_classifier) == self.N:
            self.initiation_classifier.train_one_class(self.successful_observations_to_create_initiation_classifier)
            self.initiation_classifier_created = True

    def refine_inititation_classifier(self):
        assert self.initiation_classifier_created
        assert not self.initiation_classifier_refined

        if not self.initiation_classifier_refined:
            self.initiation_classifier.train_two_class(self.good_examples_to_refine, self.bad_examples_to_refine)
            self.initiation_classifier_refined = True
