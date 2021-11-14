import gym
import mujoco_maze
import json
from ddqn.ddqn_agent import DDQNAgent
from option import Option
from typing import Any, Dict


def read_hyperparams() -> Dict[str, Any]:
    with open('hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def initial_state_covered(initial_state, option_repertoire):
    # initial state is not in any option's initiation classifier
    for o in option_repertoire:
        if o.initiation_classifier.check(initial_state):
            return True


def main() -> None:
    hyperparams = read_hyperparams()
    env = gym.make("Point4Rooms-v1")
    initial_state = env.reset()

    global_option = Option(budget=1, env=env, parent=None, N=hyperparams['N'], K=hyperparams['K'])
    goal_option = Option(hyperparams['budget'], env=env, parent=None, N=hyperparams['N'], K=hyperparams['K'])

    option_repertoire = [global_option]
    untrained_option = goal_option

    agent_over_options = DDQNAgent(obs_size=env.observation_space.shape[0], option_repertoire=option_repertoire)

    obs = env.reset()
    done = False
    while not done:
        selected_option = agent_over_options.act(obs)
        next_obs, reward_list, done, successful_observations = selected_option.execute(obs)
        agent_over_options.step(obs, selected_option, reward_list, next_obs, done)

        if untrained_option.termination_classifier.check(next_obs) and not initial_state_covered(initial_state, option_repertoire):
            untrained_option.learn_initiation_classifier(successful_observations)
            if untrained_option.initiation_classifier_trained:
                # TODO: agent_over_options.add
                option_repertoire.append(untrained_option)
                untrained_option = Option(hyperparams['budget'], env=env, parent=untrained_option)


if __name__ == "__main__":
    main()
