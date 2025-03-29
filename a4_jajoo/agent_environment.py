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
    
    # while total_steps < total_steps_target:
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
            
            with torch
            action, logprob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
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
        
        advantages, returns = agent.compute_gae_and_returns(next_obs, torch.tensor([next_done], dtype=torch.float).to(agent.device))
        agent.process_transition(advantages, returns, iteration=total_steps//agent.batch_size)
    
    return episode_returns, steps_per_episode

def pj_ppo_agent_environment_step_loop(agent, env, total_steps_target, debug=False, track_q=False, **kwargs):
    episode_returns = []
    q_values = [] if track_q else None
    steps_per_episode = []
    
    total_steps = 0
    episode = 0
    next_obs, info = env.reset()
    next_obs = torch.FloatTensor(next_obs).to(agent.device)
    next_done = torch.zeros(1).to(agent.device)
    
    while total_steps < total_steps_target:
        episode_return = 0
        steps_this_episode = 0
        episode_qs = []
        
        # Collect trajectories
        for step in range(agent.num_steps):
            agent.global_step += 1
            total_steps += 1
            steps_this_episode += 1
            
            obs = next_obs
            done = next_done
            
            with torch.no_grad():
                action, logprob,q_value = agent.select_action(obs, track_q)
                value = q_value.flatten()
                if track_q:
                    value_estimate = agent.critic(obs).item()
                    episode_qs.append(value_estimate)
            
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            episode_return += reward
            
            next_obs = torch.FloatTensor(next_obs).to(agent.device)
            next_done = torch.tensor([terminated or truncated], dtype=torch.float).to(agent.device)
            reward = torch.tensor([reward], dtype=torch.float).to(agent.device)
            
            agent.store_trajectories(step, obs, action, logprob, reward, done, value)
            # import ipdb; ipdb.set_trace()

            # if track_q:
            #     episode_q_values.append(q_value.item())
            
            if terminated or truncated:
                if debug:
                    print(f"Episode: {episode}, Return: {episode_return}, Steps: {total_steps}")
                episode_returns.append(episode_return)
                steps_per_episode.append(steps_this_episode)
                if track_q:
                    q_values.append(np.mean(episode_qs) if episode_qs else 0)
                next_obs, info = env.reset()
                next_obs = torch.FloatTensor(next_obs).to(agent.device)
                next_done = torch.zeros(1).to(agent.device)
                episode += 1
                episode_return = 0
                steps_this_episode = 0
        
        # Compute advantages and returns
        with torch.no_grad():
            advantages, returns = agent.compute_gae_and_returns(next_obs, next_done)
        
        # Update agent
        agent.process_transition(advantages, returns, iteration=total_steps//agent.batch_size)

    if track_q:
                q_values.append(q_value)
        
    return episode_returns, q_values, steps_per_episode