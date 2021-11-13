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


def main() -> None:
    hyperparams = read_hyperparams()
    env = gym.make("Point4Rooms-v1")

    global_option = Option(budget=1, env=env, parent=None)
    goal_option = Option(hyperparams['budget'], env=env, parent=None)

    option_repertoire = [global_option]
    untrained_option = goal_option

    agent_over_options = DDQNAgent(obs_size=env.observation_space.shape[0], option_repertoire=option_repertoire)

    obs = env.reset()
    done = False
    while not done:
        selected_option = agent_over_options.act(obs)
        rewards, obs = selected_option.execute(obs)


if __name__ == "__main__":
    main()
