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

        self.agent = DDPGAgent(3, self.env.action_space.shape[0], env.observation_space["desired_goal"].shape[0],
                               [env.action_space.low[0], env.action_space.high[0]], env.compute_reward)
        if parent_option:
            assert parent_option.initiation_classifier_created

            self.termination_classifier = deepcopy(parent_option.initiation_classifier)
            self.termination_classifier.type_ = "termination"

            self.initiation_classifier = Classifier(type_="initiation")
        else:
            # This means self is either goal or global option
            self.termination_classifier = Classifier(type_="termination", for_global_option=self.this_is_global_option,
                                                     for_goal_option=self.this_is_goal_option, env_termination_checker=env.termination)

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

    def execute(self, env_dict):
        assert self.initiation_classifier_created

        starting_obs = deepcopy(env_dict["observation"])
        t = 0

        episode_dict = {
            "state": [],
            "action": [],
            "achieved_goal": [],
            "desired_goal": [],
            "next_state": [],
            "next_achieved_goal": []
        }
        obs = env_dict["observation"]
        achieved_goal = env_dict["achieved_goal"]
        desired_goal = env_dict["desired_goal"]

        reward_list = []

        local_done = False
        while not local_done and t < self.budget:
            action = self.agent.act(obs, desired_goal)
            next_env_dict, reward, done, _ = self.env.step(action)
            reward_list.append(reward)

            next_obs = next_env_dict["observation"]
            next_achieved_goal = next_env_dict["achieved_goal"]
            next_desired_goal = next_env_dict["desired_goal"]

            local_done = self.termination_classifier.check(next_obs)

            episode_dict["state"].append(obs)
            episode_dict["action"].append(action)
            episode_dict["achieved_goal"].append(achieved_goal)
            episode_dict["desired_goal"].append(desired_goal)

            obs = next_obs
            achieved_goal = next_achieved_goal
            desired_goal = next_desired_goal

            t += 1

        episode_dict["state"].append(obs)
        episode_dict["achieved_goal"].append(achieved_goal)
        episode_dict["desired_goal"].append(desired_goal)
        episode_dict["next_state"] = episode_dict["state"][1:]
        episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]

        self.agent.store(deepcopy(episode_dict))
        if len(self.agent.memory) > self.agent.batch_size:
            self.agent.train()
            self.agent.update_networks()

        # those successful_observations are not for self, it is for option_without_initiation_classifier
        successful_observation = None
        if local_done:
            self.good_examples_to_refine.append(starting_obs)
            if len(self.agent.memory) > self.K:
                successful_observation = self.agent.memory.memory[-self.K].state[0]
        else:
            self.bad_examples_to_refine.append(starting_obs)

        if not self.initiation_classifier_refined and len(self.good_examples_to_refine) > self.max_refine:
            self.refine_inititation_classifier()

        return next_env_dict, reward_list, done, successful_observation

    def create_initiation_classifier(self, successful_observation) -> None:
        # there shouldn't be any actions for initiation classifier if we call this function
        assert not self.initiation_classifier_created
        assert not self.initiation_classifier_refined

        if successful_observation is not None:
            self.successful_observations_to_create_initiation_classifier.append(successful_observation)

        if len(self.successful_observations_to_create_initiation_classifier) == self.N:
            self.initiation_classifier.train_one_class(self.successful_observations_to_create_initiation_classifier)
            self.initiation_classifier_created = True
            return True

        return False

    def refine_inititation_classifier(self):
        assert self.initiation_classifier_created
        assert not self.initiation_classifier_refined

        if not self.initiation_classifier_refined:
            self.initiation_classifier.train_two_class(self.good_examples_to_refine, self.bad_examples_to_refine)
            self.initiation_classifier_refined = True
