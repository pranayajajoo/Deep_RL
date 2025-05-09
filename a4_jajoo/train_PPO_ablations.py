import gymnasium as gym
import agent_environment
import numpy as np
import epsilon_greedy_explorers
import PPO
import TD3
# import double_dqn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import replay_buffer
import argparse

CCID="jajoo"

def interpolate_returns(all_returns, all_steps, num_points=1000):
    interp_returns = []

    # Use actual max steps instead of a hardcoded 1M
    x_max = max([np.sum(steps) for steps in all_steps])
    x_target = np.linspace(0, x_max, num_points)

    for returns, steps in zip(all_returns, all_steps):
        x_actual = np.cumsum(steps)
        y_actual = np.array(returns)

        if x_actual[-1] < x_target[-1]:
            x_actual = np.append(x_actual, x_target[-1])
            y_actual = np.append(y_actual, y_actual[-1])

        y_interp = np.interp(x_target, x_actual, y_actual)
        interp_returns.append(y_interp)

    return x_target, interp_returns





# Plot
def plot_interpolated(interpolated_returns, x_steps, file, label="PPO", ylabel="Return", window=10):
    array = np.array(interpolated_returns)
    mean = np.mean(array, axis=0)

    def running_mean(x, N):
        return np.convolve(x, np.ones(N)/N, mode='valid')

    smoothed_mean = running_mean(mean, window)

    plt.figure(figsize=(10, 6))

    for run in array:
        smoothed_run = running_mean(run, window)
        trimmed_x = x_steps[:len(smoothed_run)]
        plt.plot(trimmed_x, smoothed_run, color='lightcoral', alpha=0.3)

    plt.plot(x_steps[:len(smoothed_mean)], smoothed_mean, color='darkred', label=f"{label} (avg)", linewidth=2.5)

    plt.title(f"({CCID}) TD3 Episodic Return (Aligned by Steps)")
    plt.xlabel("Environment Steps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.xlim([0, x_steps[len(smoothed_mean)-1]])  # ✅ Fixes axis issue
    plt.savefig(file)



# Adapted from ChatGPT
def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return", title="Episodic Returns", window=10):
    array = np.array(episode_returns_list)
    mean = np.mean(array, axis=0)

    # Compute running mean
    def running_mean(x, N):
        return np.convolve(x, np.ones(N)/N, mode='valid')

    smoothed_mean = running_mean(mean, window)

    plt.figure(figsize=(10, 6))

    # Plot each run (optional)
    for run in array:
        smoothed_run = running_mean(run, window)
        plt.plot(smoothed_run, color='lightcoral', alpha=0.3, linewidth=1)

    # Plot smoothed average
    plt.plot(smoothed_mean, color='darkred', label=f"{label} (avg)", linewidth=2.5)

    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(file)



def plot_many_algs(lists, labels, colors, file, ylabel="Return", title="Stepwise Returns"):
    # Define a function to calculate the running average

    running_avgs = []
    for i in range(len(lists)):
        running_avgs.append(np.mean(np.array(lists[i]), axis=0))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original data
    # plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Original Data')

    for i in range(len(lists)):
        # Plot the running average
        plt.plot(
            range(0, len(running_avgs[i])),
            running_avgs[i],
            color=colors[i],
            label=labels[i],
        )

    # Adding labels and title
    plt.title(f"({CCID}){title}")
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


if __name__ == '__main__':

    print('Running Ant-v4')
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--env", default="Ant-v4")
    parser.add_argument("--algorithm", default="PPO")
    args = parser.parse_args()

    
    num_seeds = args.num_runs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = args.algorithm
    if agent == "TD3":
        actor_lr = 3e-4
        critic_lr = 3e-4
        discount = 0.99
        tau = 0.005
        decay_steps = 500000
        noise_clip = 0.5
        policy_update_frequency = 4
        batch_size = 256
        buffer_size = 1000000
        num_training_steps = 1_000_000
        noise = 0.2
        agent_class_to_text = {TD3.TD3: 'TD3'} 
    elif agent == "PPO":
        num_training_steps = 2_000_000
        agent_class_to_text = {PPO.PPO: 'PPO'} 


    n_steps = [1]
    n_step_colors = ['green']
    agent_classes = [PPO.PPO]

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])


    perf_dict = {}
    q_val_dict = {}
    step_dict = {}  # for tracking steps per episode
    for n_step in n_steps:
        perf_dict[n_step] = {}
        q_val_dict[n_step] = {}
        step_dict[n_step] = {}
        for agent_class in agent_classes:
            agent_text = agent_class_to_text[agent_class]
            alg_returns = []
            alg_q_values = []
            alg_steps = []
            for seed in range(num_seeds):
                
                num_actions = action_dim
                if agent == "TD3":
                    buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)
                    agent = agent_class(state_dim, action_dim, max_action, discount, tau, device, noise, decay_steps, noise_clip, policy_update_frequency, batch_size, buffer)
                    step_returns, step_q_values = agent_environment.agent_environment_episode_loop(agent, env, num_training_steps, args.debug, args.track_q)

                elif agent == "PPO":
                    # agent = agent_class(state_dim, action_dim, max_action, device, env)
                    agent = agent_class(state_dim, action_dim, max_action, device)
                    episode_returns, episode_q_values, steps_per_episode = agent_environment.ppo_agent_environment_step_loop(agent, env, num_training_steps, args.debug, args.track_q, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, clip_vloss=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, num_steps=2048, num_minibatches=32, update_epochs=10, norm_adv=True, anneal_lr=True, total_timesteps = num_training_steps)
                alg_returns.append(episode_returns)
                alg_q_values.append(episode_q_values)
                alg_steps.append(steps_per_episode)
                # import ipdb; ipdb.set_trace()


            perf_dict[n_step][agent_text] = alg_returns
            q_val_dict[n_step][agent_text] = alg_q_values
            step_dict[n_step][agent_text] = alg_steps
            # import ipdb; ipdb.set_trace()
            plot_alg_results(perf_dict[n_step][agent_text], f"{agent_text}_{n_step}_step_ant.png", label=agent_text)
            if args.track_q:
                plot_alg_results(q_val_dict[n_step][agent_text], f"{agent_text}_{n_step}_step_ant_q_vals.png",
                                 label=agent_text, ylabel="Q-values", title="Q-values")


        # === After training ===
    all_returns = perf_dict[1]['PPO']
    all_steps = step_dict[1]['PPO']  # New: step_dict to store steps

    x_common, interpolated = interpolate_returns(all_returns, all_steps)

    plot_interpolated(interpolated, x_common, "ppo_interpolated_1M_steps.png")


    for n_step in n_steps:
        plot_many_algs([perf_dict[n_step][agent_text] for agent_text in ['PPO']],
                       ['PPO'], ['green'], f"ppo_{n_step}_step_ant.png",)
        if args.track_q:
            plot_many_algs([q_val_dict[n_step][agent_text] for agent_text in ['PPO']],
                       ['PPO'], ['green'], f"ant_{n_step}_q_vals.png", ylabel="Q-values", title="Q-values")
    
    for agent_class in agent_classes:
        agent_text = agent_class_to_text[agent_class]
        plot_many_algs([perf_dict[n_step][agent_text] for n_step in n_steps],
                       [f"{n_step}-step {agent_text}"for n_step in n_steps], n_step_colors, f"{agent_text}_ant.png")
