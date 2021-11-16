import json
from copy import deepcopy
from dqn.dqn_agent import DQNAgent
from option import Option
from typing import Any, Dict
from custom_env import CustomEnv


def read_hyperparams() -> Dict[str, Any]:
    with open('hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def initial_state_covered(initial_state, option_repertoire):
    for o in option_repertoire:
        if o.initiation_classifier.check(initial_state):
            return True

    return False


def main() -> None:
    # environment's initial state must be always same!
    # in this regard, mujoco_maze package's maze_env.py and point.py files modified
    hyperparams = read_hyperparams()
    env = CustomEnv()
    initial_state = deepcopy(env.reset()['observation'])

    global_option = Option(budget=1, env=env, this_is_global_option=True, this_is_goal_option=False, parent_option=None, max_refine=hyperparams['max_refine'], N=hyperparams['N'], K=hyperparams['K'])
    goal_option = Option(budget=hyperparams['budget'], env=env, this_is_global_option=False, this_is_goal_option=True,
                         parent_option=None, max_refine=hyperparams['max_refine'], N=hyperparams['N'], K=hyperparams['K'])

    option_repertoire = [global_option]
    option_without_initiation_classifier = goal_option

    agent_over_options = DQNAgent(obs_size=3, action_size=len(option_repertoire))

    for episode_num in range(hyperparams['max_episodes']):
        env_dict = env.reset()
        done = False

        while not done:
            option_index = agent_over_options.act(env_dict['observation'], option_repertoire)
            next_env_dict, reward_list, done, successful_observation = option_repertoire[option_index].execute(env_dict)
            agent_over_options.step(env_dict['observation'], option_index, reward_list, next_env_dict['observation'], done)

            env_dict = deepcopy(next_env_dict)

            if option_without_initiation_classifier.termination_classifier.check(env_dict['observation']) and not initial_state_covered(initial_state, option_repertoire[1:]):
                if not option_without_initiation_classifier.initiation_classifier_created:
                    created = option_without_initiation_classifier.create_initiation_classifier(successful_observation)
                    if created:
                        option_without_initiation_classifier.agent.load_global_weights(global_option.agent.actor_network, global_option.agent.critic_network)
                        agent_over_options.add_option()
                        option_repertoire.append(option_without_initiation_classifier)

                if option_without_initiation_classifier.initiation_classifier_refined:
                    option_without_initiation_classifier = Option(hyperparams['budget'], env=env, parent_option=option_without_initiation_classifier,
                                                                  max_refine=hyperparams['max_refine'], N=hyperparams['N'], K=hyperparams['K'])

        print(f"{episode_num}/{hyperparams['max_episodes']}")


if __name__ == "__main__":
    main()
