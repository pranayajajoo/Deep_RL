import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from actor import PPOActor as Actor
from critic import PPOCritic as Critic
from torch.distributions.normal import Normal

class PPO:
    def __init__(
        self,
        obs_dim,
        action_dim,
        max_action,
        device,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        num_steps=2048,
        num_minibatches=32,
        update_epochs=10,
        norm_adv=True,
        anneal_lr=True,
        total_timesteps=1_000_000,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
            eps=1e-5
        )

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.anneal_lr = anneal_lr
        self.total_timesteps = total_timesteps

        self.batch_size = num_steps  # For single environment
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_iterations = total_timesteps // self.batch_size
        self.global_step = 0

        # Rollout buffer
        self.obs_buf = torch.zeros((num_steps, obs_dim)).to(device)
        self.actions_buf = torch.zeros((num_steps, action_dim)).to(device)
        self.logprobs_buf = torch.zeros(num_steps).to(device)
        self.rewards_buf = torch.zeros(num_steps).to(device)
        self.dones_buf = torch.zeros(num_steps).to(device)
        self.values_buf = torch.zeros(num_steps).to(device)

    def select_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            dist = self.actor(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            value = self.critic(obs_tensor).squeeze()
            return action.squeeze(0).cpu().numpy(), log_prob, value

    def store_trajectories(self, step, obs, action, logprob, reward, done, value):
        self.obs_buf[step] = torch.FloatTensor(obs).to(self.device)
        self.actions_buf[step] = torch.FloatTensor(action).to(self.device)
        self.logprobs_buf[step] = logprob
        self.rewards_buf[step] = reward
        self.dones_buf[step] = done
        self.values_buf[step] = value

    def compute_gae_and_returns(self, last_obs, last_done):
        with torch.no_grad():
            next_value = self.critic(torch.FloatTensor(last_obs).to(self.device).unsqueeze(0))
            advantages = torch.zeros_like(self.rewards_buf).to(self.device)
            last_gae_lambda = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_nonterminal = 1.0 - last_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - self.dones_buf[t + 1]
                    next_values = self.values_buf[t + 1]
                delta = self.rewards_buf[t] + self.gamma * next_values * next_nonterminal - self.values_buf[t]
                advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae_lambda
            returns = advantages + self.values_buf
        return advantages, returns
    
    def annealing_lr(self, iteration):
        frac = 1.0 - (iteration - 1.0) / self.num_iterations
        new_lr = frac * self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = new_lr

    def normalizing_advantage(self, mb_advantages):
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        return mb_advantages
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor.log_std.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def process_transition(self, advantages, returns, iteration):
        b_obs = self.obs_buf.reshape((-1, self.obs_dim))
        b_actions = self.actions_buf.reshape((-1, self.action_dim))
        b_logprobs = self.logprobs_buf.reshape(-1)
        b_values = self.values_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # if self.anneal_lr:
        #     frac = 1.0 - (iteration - 1.0) / self.num_iterations
        #     new_lr = frac * self.optimizer.param_groups[0]['lr']
        #     self.optimizer.param_groups[0]['lr'] = new_lr

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                
                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # action_mean = self.actor(b_obs[mb_inds])
                # action_logstd = self.actor.log_std.expand_as(action_mean)
                # action_std = torch.exp(action_logstd)
                # probs = Normal(action_mean, action_std)
                # if action is None:
                #     action = probs.sample()
                # newlogprob = probs.log_prob(action).sum(1)
                # entropy = probs.entropy().sum(1)
                # newvalue = self.critic(b_obs[mb_inds])

                # dist = self.actor(b_obs[mb_inds])
                # new_logprob = dist.log_prob(b_actions[mb_inds]).sum(-1)
                # entropy = dist.entropy().sum(-1)
                # ratio = (new_logprob - b_logprobs[mb_inds]).exp()

                with torch.no_grad():
                    old_approx_kl = (-ratio.log()).mean()
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = self.normalizing_advantage(mb_advantages)

                # losses
                #policy
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                #value
                new_value = self.critic(b_obs[mb_inds]).view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(new_value - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()