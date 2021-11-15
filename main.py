import gym
import mujoco_maze
import json
from dqn.dqn_agent import DQNAgent
from option import Option
from typing import Any, Dict


def read_hyperparams() -> Dict[str, Any]:
    with open('hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def initial_state_covered(initial_state, option_repertoire):
    # repertoire shouldn't cover global agent, since its init classifier is true everywhere
    repertoire_without_global = list(filter(lambda o: not o.this_is_global_option, option_repertoire))

    for o in repertoire_without_global:
        if o.initiation_classifier.check(initial_state):
            return True


def main() -> None:
    hyperparams = read_hyperparams()
    env = gym.make("Point4Rooms-v1")
    initial_state = env.reset()

    global_option = Option(budget=1, env=env, this_is_global_option=True, this_is_goal_option=False, parent_option=None, max_refine=hyperparams['max_refine'], N=hyperparams['N'], K=hyperparams['K'])
    goal_option = Option(budget=hyperparams['budget'], env=env, this_is_global_option=False, this_is_goal_option=True,
                         parent_option=None, max_refine=hyperparams['max_refine'], N=hyperparams['N'], K=hyperparams['K'])

    option_repertoire = [global_option]
    option_without_initiation_classifier = goal_option

    agent_over_options = DQNAgent(obs_size=env.observation_space.shape[0], option_repertoire=option_repertoire)

    obs = env.reset()
    done = False
    while not done:
        selected_option = agent_over_options.act(obs)
        next_obs, reward_list, done, successful_observations = selected_option.execute(obs)
        agent_over_options.step(obs, selected_option, reward_list, next_obs, done)

        if option_without_initiation_classifier.termination_classifier.check(next_obs) and not initial_state_covered(initial_state, option_repertoire):
            # no more things to do here, refining is a option's internal process
            option_without_initiation_classifier.create_initiation_classifier(successful_observations)
            if option_without_initiation_classifier.initiation_classifier_created:
                option_without_initiation_classifier.agent.load_global_weights(global_option.agent.actor_network, global_option.agent.critic_network)
                agent_over_options.add_option(option_without_initiation_classifier)
                option_repertoire.append(option_without_initiation_classifier)
                option_without_initiation_classifier = Option(hyperparams['budget'], env=env, parent_option=option_without_initiation_classifier,
                                                              max_refine=hyperparams['max_refine'], N=hyperparams['N'], K=hyperparams['K'])


if __name__ == "__main__":
    main()
