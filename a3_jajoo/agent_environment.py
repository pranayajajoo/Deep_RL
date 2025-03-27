import torch
import numpy as np

def agent_environment_episode_loop(agent, env, num_episodes, debug=False, track_q=False):
    episode_returns = []
    mean_q_predictions = []

    for episode in range(num_episodes): 
        if track_q:
            episode_q_values = []
        observation, info = env.reset()
        episode_return = 0
        terminated, truncated = False, False
        step = 0

        while not (terminated or truncated):
            step += 1
            action, qval = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.process_transition(next_observation, reward, terminated, truncated)
            
            episode_return += reward
            observation = next_observation

            if track_q:
                episode_q_values.append(qval)

        episode_returns.append(episode_return)
        if track_q:
            mean_q_predictions.append(np.mean(episode_q_values))

        if debug:
            print(f"episode: {episode}, episode_return: {episode_return}")

    if track_q:
        return episode_returns, mean_q_predictions
    else:
        return episode_returns, None
