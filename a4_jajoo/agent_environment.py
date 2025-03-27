import torch
import numpy as np

def agent_environment_episode_loop(agent, env, num_episodes, debug=False, track_q=False):
    episode_returns = []
    q_values = []
    steps_per_episode = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_return = 0
        steps_this_episode = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.select_action(observation)

            state_tensor = torch.FloatTensor(observation.reshape(1, -1)).to(agent.device)
            action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(agent.device)
            qval = agent.critic.q1(state_tensor, action_tensor).item()

            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.process_transition(observation, action, reward, next_observation, terminated, truncated)

            episode_return += reward
            steps_this_episode += 1
            observation = next_observation

            if track_q:
                q_values.append(qval)

        episode_returns.append(episode_return)
        steps_per_episode.append(steps_this_episode)

        if debug:
            print(f"episode: {episode}, return: {episode_return}, steps: {agent.steps}, policy noise: {agent.current_noise}")

    if track_q:
        return episode_returns, q_values, steps_per_episode
    else:
        return episode_returns, None, steps_per_episode


def agent_environment_step_loop(agent, env, total_steps_target, debug=False, track_q=False):
    episode_returns = []
    q_values = []
    steps_per_episode = []

    total_steps = 0
    episode = 0

    while total_steps < total_steps_target:
        observation, info = env.reset()
        episode_return = 0
        steps_this_episode = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.select_action(observation)

            state_tensor = torch.FloatTensor(observation.reshape(1, -1)).to(agent.device)
            action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(agent.device)
            qval = agent.critic.q1(state_tensor, action_tensor).item()

            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.process_transition(observation, action, reward, next_observation, terminated, truncated)

            episode_return += reward
            steps_this_episode += 1
            total_steps += 1
            observation = next_observation

            if track_q:
                q_values.append(qval)

            if total_steps >= total_steps_target:
                break

        episode_returns.append(episode_return)
        steps_per_episode.append(steps_this_episode)

        if debug:
            print(f"Episode: {episode}, Return: {episode_return}, Steps so far: {total_steps}, Policy noise: {agent.current_noise}")
        episode += 1

    if track_q:
        return episode_returns, q_values, steps_per_episode
    else:
        return episode_returns, None, steps_per_episode




