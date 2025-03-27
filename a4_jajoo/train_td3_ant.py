import gymnasium as gym
import agent_environment
import numpy as np
import epsilon_greedy_explorers
import TD3
# import double_dqn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import replay_buffer
import argparse

CCID="jajoo"

# Adapted from ChatGPT
def plot_alg_results(step_returns_list, file, label="Algorithm", ylabel="Return", title="Stepwise Returns"):

    # Compute running average
    running_avg = np.mean(np.array(step_returns_list), axis=0)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original data
    # plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Original Data')

    # Plot the running average
    plt.plot(
        range(0, len(running_avg)),
        running_avg,
        color='r',
        label=label
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--env", default="Ant-v4")
    parser.add_argument("--decay-steps", default=100000)
    args = parser.parse_args()

    
    num_seeds = args.num_runs
    actor_lr = 3e-4
    critic_lr = 3e-4
    discount = 0.99
    tau = 0.005
    initial_policy_noise = 0.5
    decay_steps = args.decay_steps # 100000
    decay_steps = 500000
    noise_clip = 0.5
    policy_update_frequency = 4
    batch_size = 256
    buffer_size = 1000000
    # num_training_episodes = 2000
    # total_training_steps = 1_000_000
    total_training_steps = 100000


    agent_class_to_text = {TD3.TD3: 'TD3'} 

    n_steps = [1]
    n_step_colors = ['green']
    agent_classes = [TD3.TD3]

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    # import ipdb; ipdb.set_trace()



    perf_dict = {}
    q_val_dict = {}
    for n_step in n_steps:
        perf_dict[n_step] = {}
        q_val_dict[n_step] = {}
        for agent_class in agent_classes:
            agent_text = agent_class_to_text[agent_class]
            alg_returns = []
            alg_q_values = []
            for seed in range(num_seeds):
                
                num_actions = action_dim
                buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)
                agent = agent_class(state_dim, action_dim, max_action, discount, tau, initial_policy_noise, decay_steps, noise_clip, policy_update_frequency, batch_size, buffer)
                # step_returns, step_q_values = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
                step_returns, step_q_values = agent_environment.agent_environment_step_loop(agent, env, total_training_steps, args.debug, args.track_q)
                alg_returns.append(step_returns)
                alg_q_values.append(step_q_values)
                # import ipdb; ipdb.set_trace()


            perf_dict[n_step][agent_text] = alg_returns
            q_val_dict[n_step][agent_text] = alg_q_values
            # import ipdb; ipdb.set_trace()
            plot_alg_results(perf_dict[n_step][agent_text], f"{agent_text}_{n_step}_step_ant.png", label=agent_text)
            if args.track_q:
                plot_alg_results(q_val_dict[n_step][agent_text], f"{agent_text}_{n_step}_step_ant_q_vals.png",
                                 label=agent_text, ylabel="Q-values", title="Q-values")


    for n_step in n_steps:
        plot_many_algs([perf_dict[n_step][agent_text] for agent_text in ['TD3']],
                       ['TD3'], ['green'], f"td3_{n_step}_step_ant.png",)
        if args.track_q:
            plot_many_algs([q_val_dict[n_step][agent_text] for agent_text in ['TD3']],
                       ['TD3'], ['green'], f"ant_{n_step}_q_vals.png", ylabel="Q-values", title="Q-values")
    
    for agent_class in agent_classes:
        agent_text = agent_class_to_text[agent_class]
        plot_many_algs([perf_dict[n_step][agent_text] for n_step in n_steps],
                       [f"{n_step}-step {agent_text}"for n_step in n_steps], n_step_colors, f"{agent_text}_ant.png")
