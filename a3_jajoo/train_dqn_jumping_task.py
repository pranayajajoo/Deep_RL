import argparse
import gymnasium as gym
import agent_environment
import numpy as np
import matplotlib.pyplot as plt
import jumping_task
from jumping_task.envs import JumpTaskEnv
import epsilon_greedy_explorers
import torch
import torch.nn as nn
import replay_buffer
import dqn
import double_dqn
import pandas as pd
import os


CCID="jajoo"
class LinearDecayEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, start_epsilon, end_epsilon, decay_steps, num_actions):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        assert start_epsilon >= end_epsilon >= 0
        self.epsilon = start_epsilon
        self.decay_steps = decay_steps
        self.num_actions = num_actions
        self.steps = 0

    def select_action(self, action_values) -> int:
        epsilon_decay_step_size = (self.start_epsilon - self.end_epsilon) / self.decay_steps
        epsilon = max(self.start_epsilon - self.steps * epsilon_decay_step_size, self.end_epsilon)
        max_actions = np.where(action_values == np.max(action_values))[0]
        action_probs = epsilon_greedy_explorers.compute_epsilon_greedy_action_probs(action_values, epsilon)
        self.steps += 1
        return np.random.choice(len(action_probs), p=action_probs)


def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return"):

    # Compute running average
    running_avg = np.mean(np.array(episode_returns_list), axis=0)
    new_running_avg = running_avg.copy()
    for i in range(len(running_avg)):
        new_running_avg[i] = np.mean(running_avg[max(0, i-100):min(len(running_avg), i + 100)])
    running_avg = new_running_avg

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the running average
    plt.plot(
        range(0, len(running_avg)),
        running_avg,
        color='r',
        label=label
    )

    # Adding labels and title
    plt.title(f"({CCID})Episodic Returns")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)

def extract_config(filename_without_ext):
    configs = ['config_1', 'config_2', 'config_3']
    for configuration in configs:
        if configuration in filename_without_ext:
            return configuration
    return None

def produce_plots_for_all_configs():
    configs = ['config_1', 'config_2', 'config_3']
    data_dict = {}
    for configuration in configs:
        data_dict[configuration] = []
    files = os.listdir("data")
    for file in files:
        full_path = os.path.join("data", file)
        if os.path.isfile(full_path):
            if os.path.splitext(file)[-1] == '.csv':
                config = extract_config(os.path.splitext(file)[0])
                assert config is not None, f"{file} is not in the required format."
                df = pd.read_csv(full_path)
                data_dict[config].append(np.squeeze(df.values))

    for configuration in configs:
        if data_dict[configuration]:
            plot_alg_results(data_dict[configuration], f"dqn_jumping_{configuration}.png")

def get_env(config_num, render=False):
    if config_num == 1:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 25, 30],
                        agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                        obstacle_position=0, obstacle_size=(9,10),
                        rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                        max_number_of_steps=300, two_obstacles=False, finish_jump=False)
    elif config_num == 2:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[30, 40],
                        agent_w=7, agent_h=7, agent_init_pos=0, agent_speed=1,
                        obstacle_position=0, obstacle_size=(11,17),
                        rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                        max_number_of_steps=300, two_obstacles=False, finish_jump=False,
                        jump_height=24)
    else:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 30, 40],
                    agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                    obstacle_position=0, obstacle_size=(9,10),
                    rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                    max_number_of_steps=300, two_obstacles=True, finish_jump=False)
    return env

def input_preprocessor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    if len(x.shape) == 2:
        x = x.unsqueeze(0)

    elif len(x.shape) == 3:
        x = x.unsqueeze(1)  
    return x

def reward_phi(r):
    reward = r
    return reward

class JumpingQNetwork(nn.Module):
    
    def __init__(self, num_actions):
        super().__init__()
        self.body = torch.nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
            )
        self.head = torch.nn.Sequential(
            nn.Linear(512, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions))
        
        
    def forward(self, input):
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, 0)
        fa = torch.flatten(self.body(input), 1)
        q_vals = self.head(fa).squeeze(0)
        return q_vals
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Which environment", type=int, choices=[1,2,3], default=1)
    parser.add_argument("--num-training-episodes", help="How many episodes you want to train your agent", default=2000, type=int)
    parser.add_argument("--run-label", help="Akin to a random seed", default=1, type=int)
    parser.add_argument("--min-replay-size-before-updates", help="minimum size of replay buffer before gradient updates are performed", default=1000, type=int)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--track-q", action='store_true')


    parser.add_argument("--n-step", help="n-step return", default=20, type=int)
    parser.add_argument("--lr", help="optimizer lr", default=0.003, type=int)
    parser.add_argument("--buffer-size", help="buffer size", default=25000, type=int)
    parser.add_argument("--epsilon-decay-steps", help="steps to min epsilon", default=10000, type=int)
    parser.add_argument("--gradients-update-frequency", help="how frequently to do GD", default=4, type=int)
    parser.add_argument("--minibatch-size", help="sample batch size from replay buffer", default=128, type=int)
    parser.add_argument("--target-update-interval", help="how often target is updated", default=100, type=int)
    args = parser.parse_args()

    env = get_env(args.config, args.render)
    num_actions = env.action_space.n

    # exploration policy
    initial_epsilon = 1.0
    final_epsilon = 0.0001
    epsilon_decay_steps = args.epsilon_decay_steps

    # replay buffer
    buffer_size = args.buffer_size
    discount = 0.99
    n_step = args.n_step # 20
    min_replay_size_before_updates = args.min_replay_size_before_updates # 1000

    # agent params
    target_update_interval = args.target_update_interval
    gradient_update_frequency = args.gradients_update_frequency # 4
    minibatch_size = args.minibatch_size # 128

    # optimizer
    lr = args.lr
    optimizer_eps = 1e-2

    num_training_episodes = args.num_training_episodes

    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_network = JumpingQNetwork(num_actions)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
    buffer = replay_buffer.ReplayBuffer(buffer_size, discount, n_step)

    agent = dqn.DQN(q_network, optimizer, buffer, explorer, discount, target_update_interval,
                    gradient_update_frequency, input_preprocessor,
                    minibatch_size,
                    min_replay_size_before_updates, reward_phi)
    episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
    df = pd.DataFrame(episode_returns)
    df.to_csv(f'data/config_{args.config}_run_{args.run_label}.csv', index=False)
    produce_plots_for_all_configs()
