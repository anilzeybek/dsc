import gym
import mujoco_maze
import json
from option import Option
from typing import Any, Dict


def read_hyperparams() -> Dict[str, Any]:
    with open('hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def main() -> None:
    hyperparams = read_hyperparams()
    env = gym.make("PointUmaze-v1")
    
    global_option = Option(1)
    goal_option = Option(hyperparams['budget'])

    option_repertoire = [global_option]
    untrained_option = goal_option

    # agent_over_options = DDQNAgent()

    obs = env.reset()
    done = False
    while not done:



if __name__ == "__main__":
    main()
