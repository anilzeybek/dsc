from __future__ import annotations
from copy import deepcopy
from typing import Dict, Tuple, List, Optional
import numpy as np
from td3.td3_agent import TD3Agent
from classifier import Classifier
import gym


class Option:
    def __init__(self, name: str, budget: int, env: gym.Env, parent_option: Optional[Option],
                 min_examples_to_refine: int, req_num_to_create_init: int) -> None:
        self.name = name
        self.budget = budget
        self.env = env
        self.parent_option = parent_option
        self.min_examples_to_refine = min_examples_to_refine
        self.req_num_to_create_init = req_num_to_create_init

        if self.name == "global" or self.name == "goal":
            assert parent_option is None, "global and goal options cant have parent option"

        self.agent = TD3Agent(env.observation_space["observation"].shape[0], env.action_space.shape[0],
                              env.observation_space["desired_goal"].shape[0],
                              {"low": env.action_space.low, "high": env.action_space.high}, env.compute_reward)

        if parent_option:
            assert self.name != "global" or self.name != "goal"
            assert parent_option.init_classifier_created, \
                "if parent provided, its init classifier should be created"

            self.termination_classifier = deepcopy(parent_option.init_classifier)
            self.termination_classifier.one_class_svm = parent_option.init_classifier.one_class_svm
            self.termination_classifier.for_global_option = False
            self.termination_classifier.for_goal_option = False
            self.termination_classifier.type_ = "termination"

            self.init_classifier = Classifier(type_="init")
        else:
            # This means self is either goal or global option
            this_is_global_option = (self.name == "global")
            this_is_goal_option = (self.name == "goal")

            self.termination_classifier = Classifier(type_="termination", for_global_option=this_is_global_option,
                                                     for_goal_option=this_is_goal_option,
                                                     env_termination_checker=env.termination)

            self.init_classifier = Classifier(type_="init", for_global_option=this_is_global_option,
                                              for_goal_option=this_is_goal_option)
        self.init_classifier_created = False
        self.init_classifier_refined = False

        if self.name == "global":
            self.init_classifier_created = True
            self.init_classifier_refined = True

        self.successful_observations_to_create_init_classifier = []
        self.good_examples_to_refine = []
        self.bad_examples_to_refine = []
        print(f"option {self.name}: generated")

    def execute(self, env_dict: Dict[str, np.ndarray], render=False, train_mode=True) -> \
            Tuple[Dict[str, np.ndarray], List[float], bool]:

        assert self.init_classifier_created, \
            "to execute an option, its init classifier must be at least created"

        starting_obs = env_dict["observation"]
        t = 0

        episode_dict = {
            "obs": [],
            "action": [],
            "reward": [],
            "achieved_goal": [],
            "desired_goal": [],
            "next_obs": [],
            "next_achieved_goal": []
        }
        obs = env_dict["observation"]
        achieved_goal = env_dict["achieved_goal"]
        desired_goal = env_dict["desired_goal"] if self.name == "global" or self.name == "goal" else \
            self.env.obs_to_achieved(self.termination_classifier.sample())

        reward_list = []

        goal_achieved = False
        done = False
        next_env_dict = {}
        while t < self.budget:
            action = self.agent.act(obs, desired_goal, train_mode=train_mode)
            assert self.env.action_space.contains(action)

            next_env_dict, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()

            next_obs = next_env_dict["observation"]
            next_achieved_goal = next_env_dict["achieved_goal"]
            next_desired_goal = next_env_dict[
                "desired_goal"] if self.name == "global" or self.name == "goal" else desired_goal

            reward_list.append(reward)
            if not goal_achieved:
                # if goal_achieved becomes true once, it should stay true
                goal_achieved = self.termination_classifier.check(next_obs)

            episode_dict["obs"].append(obs)
            episode_dict["action"].append(action)
            episode_dict["reward"].append(reward)
            episode_dict["achieved_goal"].append(achieved_goal)
            episode_dict["desired_goal"].append(desired_goal)
            episode_dict["next_obs"].append(next_obs)
            episode_dict["next_achieved_goal"].append(next_achieved_goal)

            obs = next_obs
            achieved_goal = next_achieved_goal
            desired_goal = next_desired_goal

            t += 1

            if not train_mode and goal_achieved:
                break

        if train_mode:
            self.agent.store(episode_dict)
            for _ in range(self.budget):
                self.agent.train()

            if self.name != "global":
                if goal_achieved:
                    self.good_examples_to_refine.append(starting_obs)
                else:
                    self.bad_examples_to_refine.append(starting_obs)

                if not self.init_classifier_refined and \
                        len(self.good_examples_to_refine) >= self.min_examples_to_refine and \
                        len(self.bad_examples_to_refine) >= self.min_examples_to_refine:
                    self.refine_init_classifier()

        return next_env_dict, reward_list, done

    def create_init_classifier(self, successful_observation: np.ndarray, initial_obs: np.ndarray) -> bool:
        # initial_obs is required because if list contains only it, it fails

        assert not self.init_classifier_created or not self.init_classifier_refined, \
            "if you call this function, init classifier must be untouched"

        if successful_observation is not None:
            self.successful_observations_to_create_init_classifier.append(successful_observation)

        if len(self.successful_observations_to_create_init_classifier) == self.req_num_to_create_init:
            self.init_classifier.train_one_class(self.successful_observations_to_create_init_classifier,
                                                 initial_obs)
            self.init_classifier_created = True
            print(f"option {self.name}: init classifier created")
            return True

        return False

    def refine_init_classifier(self) -> None:
        assert self.init_classifier_created, "to refine an init classifier, it must be created"
        assert not self.init_classifier_refined, "you can't refine an already refined init classifier"

        self.init_classifier.train_two_class(self.good_examples_to_refine, self.bad_examples_to_refine)
        self.init_classifier_refined = True
        print(f"option {self.name}: init classifier refined")

    def freeze(self) -> None:
        del self.successful_observations_to_create_init_classifier
        del self.good_examples_to_refine
        del self.bad_examples_to_refine
        del self.env
        del self.agent.rb
        del self.agent.compute_reward_func
        del self.init_classifier.env_termination_checker
        del self.termination_classifier.env_termination_checker
