
import gym
import numpy as np
import sys
from collections import deque
from ddqn_agent import DDQNAgent

N_EPISODES = 10000
env = gym.make('LunarLander-v2')
agent = DDQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

scores = deque(maxlen=10)
for i in range(1, N_EPISODES+1):
    state = env.reset()
    score = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

    scores.append(score)
    mean_score = np.mean(scores)

    print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}', end="")
    if i % 10 == 0:
        print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}')