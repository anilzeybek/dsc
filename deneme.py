import gym
import numpy as np
from collections import deque
from ddpg.ddpg_agent import DDPGAgent

N_EPISODES = 5000
env = gym.make('LunarLanderContinuous-v2')
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high)

scores = deque(maxlen=10)
for i in range(1, N_EPISODES+1):
    obs = env.reset()
    score = 0
    done = False
    while not done:
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward

    scores.append(score)
    mean_score = np.mean(scores)

    print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}', end="")
    if i % 10 == 0:
        print(f'\rEpisode: {i}\tAverage Score: {mean_score:.2f}')


while True:
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        action = agent.act(obs, noise=0)
        next_obs, reward, done, _ = env.step(action)
        env.render()
        
        obs = next_obs
        score += reward

    print(score)