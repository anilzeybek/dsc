from copy import deepcopy
from ddpg.ddpg_agent import DDPGAgent
from dueling_dqn.dueling_dqn_agent import DuelingDQNAgent
from classifier import Classifier


class Option:
    def __init__(self, name, action_type, budget: int, env, parent_option, min_examples_to_refine, N, K) -> None:
        self.name = name
        self.action_type = action_type
        self.budget = budget
        self.env = env
        self.parent_option = parent_option
        self.min_examples_to_refine = min_examples_to_refine
        self.N = N
        self.K = K
        self._frozen = None

        assert self.action_type in ['discrete', 'continuous'], "action_type must be either discrete or continuous"
        if self.name == "global" or self.name == "goal":
            assert parent_option is None, "global and goal options cant have parent option"

        if self.action_type == "continuous":
            self.agent = DDPGAgent(3, self.env.action_space.shape[0], env.observation_space["desired_goal"].shape[0],
                                   [env.action_space.low[0], env.action_space.high[0]], env.compute_reward)
        else:
            self.agent = DuelingDQNAgent(3, self.env.action_space.shape[0], env.observation_space["desired_goal"].shape[0],
                                         env.computer_reward)

        if parent_option:
            assert self.name != "global" or self.name != "goal"
            assert parent_option.initiation_classifier_created, "if parent provided, its initiation classifier should be created"

            self.termination_classifier = deepcopy(parent_option.initiation_classifier)
            self.termination_classifier.one_class_svm = parent_option.initiation_classifier.one_class_svm  # we want both of them to point same memory
            self.termination_classifier.for_global_option = False
            self.termination_classifier.for_goal_option = False
            self.termination_classifier.type_ = "termination"

            self.initiation_classifier = Classifier(type_="initiation")
        else:
            # This means self is either goal or global option
            this_is_global_option = (self.name == "global")
            this_is_goal_option = (self.name == "goal")

            self.termination_classifier = Classifier(type_="termination", for_global_option=this_is_global_option,
                                                     for_goal_option=this_is_goal_option, env_termination_checker=env.termination)

            self.initiation_classifier = Classifier(type_="initiation", for_global_option=this_is_global_option, for_goal_option=this_is_goal_option)
        self.initiation_classifier_created = False
        self.initiation_classifier_refined = False

        if self.name == "global":
            self.initiation_classifier_created = True
            self.initiation_classifier_refined = True

        self.successful_observations_to_create_initiation_classifier = []
        self.good_examples_to_refine = []
        self.bad_examples_to_refine = []
        print(f"option {self.name}: generated")

    def execute(self, env_dict):
        assert self.initiation_classifier_created, "to execute an option, its initiation classifier must be at least created"

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
        desired_goal = env_dict["desired_goal"] if self.name == "global" or self.name == "goal" else self.termination_classifier.sample()

        reward_list = []

        local_done = False
        while t < self.budget:
            action = self.agent.act(obs, desired_goal, train_mode=not self._frozen)
            next_env_dict, reward, done, _ = self.env.step(action)
            reward_list.append(reward)

            next_obs = next_env_dict["observation"]
            next_achieved_goal = next_env_dict["achieved_goal"]
            next_desired_goal = next_env_dict["desired_goal"] if self.name == "global" or self.name == "goal" else deepcopy(desired_goal)

            local_done = self.termination_classifier.check(next_obs)

            episode_dict["state"].append(obs)
            episode_dict["action"].append(action)
            episode_dict["achieved_goal"].append(achieved_goal)
            episode_dict["desired_goal"].append(desired_goal)

            obs = next_obs
            achieved_goal = next_achieved_goal
            desired_goal = next_desired_goal

            t += 1

            if self._frozen and local_done:
                break

        if not self._frozen:
            episode_dict["state"].append(obs)
            episode_dict["achieved_goal"].append(achieved_goal)
            episode_dict["desired_goal"].append(desired_goal)
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]

            self.agent.store(deepcopy(episode_dict))
            for _ in range(self.budget):
                self.agent.train()

            self.agent.update_networks()

            if self.name != "global":
                if local_done:
                    self.good_examples_to_refine.append(starting_obs)
                else:
                    self.bad_examples_to_refine.append(starting_obs)

                # TODO: from now on, refining shouldn't be limited to just 1, a couple of per option maybe?
                if not self.initiation_classifier_refined and len(self.good_examples_to_refine) >= self.min_examples_to_refine and len(self.bad_examples_to_refine) >= self.min_examples_to_refine:
                    self.refine_inititation_classifier()

        return next_env_dict, reward_list, done

    def create_initiation_classifier(self, successful_observation, initial_state) -> None:
        # initial_state is required because if list contains only it, it fails

        assert not self.initiation_classifier_created, "if you call this function, initiation classifier must be untouched"
        assert not self.initiation_classifier_refined, "if you call this function, initiation classifier must be untouched"

        if successful_observation is not None:
            self.successful_observations_to_create_initiation_classifier.append(successful_observation)

        if len(self.successful_observations_to_create_initiation_classifier) == self.N:
            self.initiation_classifier.train_one_class(self.successful_observations_to_create_initiation_classifier, initial_state)
            self.initiation_classifier_created = True
            print(f"option {self.name}: initiation classifier created")
            return True

        return False

    def refine_inititation_classifier(self):
        assert self.initiation_classifier_created, "to refine an initiation classifier, it must be created"
        assert not self.initiation_classifier_refined, "you can't refine an already refined initiation classifier"

        self.initiation_classifier.train_two_class(self.good_examples_to_refine, self.bad_examples_to_refine)
        self.initiation_classifier_refined = True
        print(f"option {self.name}: initiation classifier refined")

    def freeze(self):
        self._frozen = True
        self.successful_observations_to_create_initiation_classifier = []
        self.good_examples_to_refine = []
        self.bad_examples_to_refine = []

    def save(self):
        assert self._frozen, "option should be frozen to be saved"
