import json
from copy import deepcopy
from meta_dqn.meta_dqn_agent import MetaDQNAgent
from option import Option
from typing import Any, Dict
from custom_env import CustomEnv
import os
import pickle
import sys
import gym
from time import time


def read_hyperparams() -> Dict[str, Any]:
    with open('hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def is_initial_state_covered(initial_state, option_repertoire):
    for o in option_repertoire:
        if o.initiation_classifier.check(initial_state):
            print("------------Initial state covered!")
            return True

    return False


def eval():
    env = CustomEnv()
    with open(f"./train_results/options.pickle", 'rb') as f:
        option_repertoire = pickle.load(f)

    for o in option_repertoire:
        o.env = env

    agent_over_options = MetaDQNAgent(obs_size=3, action_size=len(option_repertoire))
    agent_over_options.load()

    while True:
        env_dict = env.reset()
        done = False
        total_reward = 0

        while not done:
            option_index = agent_over_options.act(env_dict['observation'], option_repertoire)
            next_env_dict, reward_list, done = option_repertoire[option_index].execute(env_dict)
            total_reward += sum(reward_list)
            env_dict = deepcopy(next_env_dict)

        print(total_reward)


def train():
    # environment's initial state must be always same!
    # in this regard, mujoco_maze package's maze_env.py and point.py files modified
    start = time()

    hyperparams = read_hyperparams()
    env = CustomEnv()
    initial_state = deepcopy(env.reset()['observation'])
    initial_state_covered = False

    action_type = 'discrete' if env.action_space == gym.spaces.Discrete else 'continuous'
    global_option = Option("global", action_type, budget=1, env=env, parent_option=None, min_examples_to_refine=hyperparams['min_examples_to_refine'],
                           N=hyperparams['N'], K=hyperparams['K'])
    goal_option = Option("goal", action_type, budget=hyperparams['budget'], env=env, parent_option=None,
                         min_examples_to_refine=hyperparams['min_examples_to_refine'], N=hyperparams['N'], K=hyperparams['K'])

    option_repertoire = [global_option]
    option_without_initiation_classifier = goal_option

    agent_over_options = MetaDQNAgent(obs_size=3, action_size=len(option_repertoire))
    agent_no = 2  # to match the option index

    for episode_num in range(hyperparams['max_episodes']):
        env_dict = env.reset()
        done = False
        obs_history = []

        while not done:
            option_index = agent_over_options.act(env_dict['observation'], option_repertoire)
            if option_repertoire[option_index].name == "global":
                obs_history.append(env_dict['observation'])

            next_env_dict, reward_list, done = option_repertoire[option_index].execute(env_dict)
            agent_over_options.step(env_dict['observation'], option_index, reward_list, next_env_dict['observation'], done)

            env_dict = deepcopy(next_env_dict)

            if not initial_state_covered:
                if option_without_initiation_classifier.termination_classifier.check(env_dict['observation']) and not option_without_initiation_classifier.initiation_classifier_created:
                    try:
                        k_steps_before = obs_history[-hyperparams['K']]
                    except IndexError:
                        k_steps_before = obs_history[0]

                    created = option_without_initiation_classifier.create_initiation_classifier(k_steps_before, initial_state)
                    if created:
                        option_without_initiation_classifier.agent.load_global_weights(global_option.agent.actor, global_option.agent.critic)
                        agent_over_options.add_option()
                        option_repertoire.append(option_without_initiation_classifier)
                        option_without_initiation_classifier = Option(str(agent_no), action_type, hyperparams['budget'], env=env, parent_option=option_without_initiation_classifier,
                                                                      min_examples_to_refine=hyperparams['min_examples_to_refine'], N=hyperparams['N'], K=hyperparams['K'])
                        agent_no += 1
                        initial_state_covered = is_initial_state_covered(initial_state, option_repertoire[1:])

        print(f"{episode_num}/{hyperparams['max_episodes']}")

        # if time() - start >= 150:
        #     break

    end = time()
    print("training completed, elapsed time: ", end-start)

    for o in option_repertoire:
        o.freeze()
    os.makedirs("./train_results", exist_ok=True)

    with open(f'./train_results/options.pickle', 'wb') as f:
        pickle.dump(option_repertoire, f)

    agent_over_options.save()


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "eval":
        print("----EVAL----")
        eval()
    else:
        print("----TRAIN----")
        train()


if __name__ == "__main__":
    main()
