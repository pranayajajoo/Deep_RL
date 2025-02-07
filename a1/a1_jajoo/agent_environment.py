

def agent_environment_episode_loop(agent, env, num_episodes):
    episode_returns = []
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_return = 0
        done, truncated = False,False
        while not (truncated or done):
            action = agent.act(observation)
            next_observation, reward, done, truncated, info = env.step(action)
            episode_return += reward
            agent.process_transition(next_observation, reward, done, truncated)
            observation = next_observation
        episode_returns.append(episode_return)
        agent.prev_state = None
        agent.prev_action = None
    return episode_returns

def agent_environment_step_loop(agent, env, num_steps):
    observation, info = env.reset()
    episode_returns = []
    episode_return = 0
    for _ in range(num_steps):
        action = agent.act(observation)
        next_observation, reward, done, truncated, info = env.step(action)
        episode_return += reward
        if truncated or done:
            episode_returns.append(episode_return)
            episode_return = 0
            observation, info = env.reset()
            agent.prev_state = None
            agent.prev_action = None 
        else:
            observation = next_observation
            agent.process_transition(observation, reward, done, truncated)
    return episode_returns
