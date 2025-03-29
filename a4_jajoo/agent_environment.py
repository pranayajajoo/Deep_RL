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






def ppo_agent_environment_step_loop(agent, env, total_steps_target, debug=False, **kwargs):
    episode_returns = []
    steps_per_episode = []
    
    total_steps = 0
    episode = 0
    next_obs, info = env.reset()
    next_done = False
    num_iterations = total_steps_target // agent.batch_size

    for iteration in range(num_iterations + 1):
        if agent.anneal_lr:
            agent.annealing_lr(iteration)
        
        episode_return = 0
        steps_this_episode = 0
        
        for step in range(agent.num_steps):
            agent.global_step += 1
            total_steps += 1
            steps_this_episode += 1
            
            obs = next_obs
            done = next_done

            # 🔧 FIXED INDENTATION HERE
            with torch.no_grad():
                action, logprob, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 🔧 FIXED: Update next_done properly
            next_done = done

            agent.store_trajectories(step, obs, action, logprob, reward, done, value)
            episode_return += reward

            if done:
                if debug:
                    print(f"Episode: {episode}, Return: {episode_return}, Steps: {total_steps}")
                episode_returns.append(episode_return)
                steps_per_episode.append(steps_this_episode)
                next_obs, info = env.reset()
                next_done = False
                episode += 1
                break

        advantages, returns = agent.compute_gae_and_returns(
            next_obs, torch.tensor([next_done], dtype=torch.float).to(agent.device)
        )
        agent.process_transition(advantages, returns, iteration=total_steps // agent.batch_size)

    return episode_returns, steps_per_episode

