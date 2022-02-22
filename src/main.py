from collections import defaultdict
import json
import numpy as np
import random
import torch
from meta_dqn.meta_dqn_agent import MetaDQNAgent
from option import Option
import os
import pickle
import gym
import mujoco_maze
from time import time
import argparse
import matplotlib.pyplot as plt


def read_hyperparams():
    with open('hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def is_initial_obs_covered(initial_obs, option_repertoire):
    for o in option_repertoire:
        if o.init_classifier.check(initial_obs):
            print("------------Initial obs covered!")
            return True

    return False


def test(env):
    print("----TEST----")

    with open(f"./saved_trainings/options.pickle", 'rb') as f:
        option_repertoire = pickle.load(f)

    for o in option_repertoire:
        o.env = env
        if o.name == "global" or o.name == "goal":
            o.termination_classifier.env_termination_checker = o.env.termination

    agent_over_options = MetaDQNAgent(obs_dim=env.observation_space["observation"].shape[0],
                                      action_dim=len(option_repertoire))
    agent_over_options.load()

    while True:
        evaluate(env, agent_over_options, option_repertoire, render=True)


def evaluate(env, agent_over_options, option_repertoire, render=False):
    print("\n---EVALUATING:")

    env_dict = env.reset()
    done = False
    total_reward = 0

    last_was_global = False
    while not done:
        option_index = agent_over_options.act(env_dict['observation'], option_repertoire, train_mode=False)

        if not (last_was_global and option_repertoire[option_index].name == "global"):
            print(option_repertoire[option_index].name)

        last_was_global = option_repertoire[option_index].name == "global"
        next_env_dict, reward_list, done = option_repertoire[option_index].execute(
            env_dict, render=render, train_mode=False
        )

        total_reward += sum(reward_list)
        env_dict = next_env_dict

    print(f"{total_reward}\n")


def train(env, global_only=False):
    print("----TRAIN----")
    start = time()
    hyperparams = read_hyperparams()

    initial_obs = env.reset()['observation']
    initial_obs_covered = False

    global_option = Option("global", budget=1, env=env, parent_option=None,
                           min_examples_to_refine=hyperparams['min_examples_to_refine'],
                           req_num_to_create_init=hyperparams['req_num_to_create_init'])

    if not global_only:
        goal_option = Option("goal", budget=hyperparams['budget'], env=env, parent_option=None,
                             min_examples_to_refine=hyperparams['min_examples_to_refine'],
                             req_num_to_create_init=hyperparams['req_num_to_create_init'])
        option_without_init_classifier = goal_option

    agent_no = 2  # to match the option index
    option_repertoire = [global_option]
    agent_over_options = MetaDQNAgent(obs_dim=env.observation_space["observation"].shape[0],
                                      action_dim=len(option_repertoire))

    all_rewards = []
    for episode_num in range(1, hyperparams['max_episodes']+1):
        env_dict = env.reset()
        done = False
        obs_history = []
        this_episode_used = False
        episode_reward = 0
        last_executed_option_name = ""

        while not done:
            option_index = agent_over_options.act(env_dict['observation'], option_repertoire)
            if not global_only and option_repertoire[option_index].name == "global":
                obs_history.append(env_dict['observation'])

            next_env_dict, reward_list, done = option_repertoire[option_index].execute(env_dict)
            episode_reward += sum(reward_list)
            agent_over_options.step(env_dict['observation'], option_index, reward_list,
                                    next_env_dict['observation'], done)

            if (last_executed_option_name == "global" and option_repertoire[option_index].name != "global") or done:
                # global can't store episode dict its inside because it doesn't know when it terminated, so stored here
                option_repertoire[0].agent.store(option_repertoire[0].globals_episode_dict)
                option_repertoire[0].globals_episode_dict = defaultdict(lambda: [])

            last_executed_option_name = option_repertoire[option_index].name
            env_dict = next_env_dict

            if not global_only and \
                    not initial_obs_covered and \
                    not this_episode_used and \
                    option_without_init_classifier.termination_classifier.check(env_dict['observation']):

                try:
                    k_steps_before = obs_history[-hyperparams['num_steps_before']]
                except IndexError:
                    k_steps_before = obs_history[0]

                this_episode_used = True
                created = option_without_init_classifier.create_init_classifier(k_steps_before, initial_obs)
                if created:
                    option_without_init_classifier.agent.load_global_weights(global_option.agent.actor,
                                                                             global_option.agent.critic)

                    agent_over_options.add_option()
                    option_repertoire.append(option_without_init_classifier)

                    initial_obs_covered = is_initial_obs_covered(initial_obs, option_repertoire[1:])
                    if not initial_obs_covered:
                        option_without_init_classifier = Option(
                            str(agent_no),
                            hyperparams['budget'],
                            env=env,
                            parent_option=option_without_init_classifier,
                            min_examples_to_refine=hyperparams['min_examples_to_refine'],
                            req_num_to_create_init=hyperparams['req_num_to_create_init'],
                        )
                        agent_no += 1

        all_rewards.append(episode_reward)
        if episode_num % 10 == 0:
            evaluate(env, agent_over_options, option_repertoire)

        print(f"{episode_num}/{hyperparams['max_episodes']}")

        # if time() - start >= 150:
        #     break

    end = time()
    print("training completed, elapsed time: ", end - start)

    os.makedirs("./saved_trainings", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    all_rewards_ = np.array(all_rewards)
    smoothed_all_rewards = np.mean(all_rewards_.reshape(-1, 10), axis=1)

    plt.plot(smoothed_all_rewards)
    plt.title(f"budget: {hyperparams['budget']}")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("plots/plot.png")

    for o in option_repertoire:
        o.freeze()

    with open(f'./saved_trainings/options.pickle', 'wb') as f:
        pickle.dump(option_repertoire, f)

    agent_over_options.save()


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--dynamic_goal', default=False, action='store_true')
    parser.add_argument('--global_only', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    env = gym.make("Point4Rooms-v1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    if args.dynamic_goal:
        env.dynamic_goal()

    if args.test:
        test(env)
    else:
        train(env, args.global_only)


if __name__ == "__main__":
    main()
