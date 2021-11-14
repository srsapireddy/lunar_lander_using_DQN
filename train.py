import gym
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
import matplotlib.pyplot as plt

# Seeding the libs
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class QNet(nn.Module):
    def __init__(self, states, actions):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(states[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, s, a, r, next_s, d):
        self.buffer.append((s, a, r, next_s, d))

    def sample(self):
        exps = random.sample(self.buffer, k=self.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([_[0] for _ in exps if _])).float()
        actions = torch.from_numpy(np.vstack([_[1] for _ in exps if _])).long()
        rewards = torch.from_numpy(np.vstack([_[2] for _ in exps if _])).float()
        next_states = torch.from_numpy(np.vstack([_[3] for _ in exps if _])).float()
        dones = torch.from_numpy(np.vstack([_[4] for _ in exps if _]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class LearningAgent(object):
    def __init__(self):
        self.episodes = 1000
        self.replay_buffer_size = 100000
        self.batch_size = 64
        self.gamma = 0.99
        self.lr = 1e-3
        self.tau = 1e-3
        self.c_steps = 4
        self.env = gym.make("LunarLander-v2").env
        self.env.seed(seed)
        self.Q = QNet(self.env.observation_space.shape, self.env.action_space.n)
        self.FQ = QNet(self.env.observation_space.shape, self.env.action_space.n)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.batch_size)
        self.time_step = 0

    def step(self, s, a, r, next_s, d):
        self.replay_buffer.add(s, a, r, next_s, d)
        self.time_step += 1
        if self.time_step % self.c_steps == 0:
            if len(self.replay_buffer) > self.batch_size:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample()
                action_values = self.FQ(next_states).detach()
                max_action_values = action_values.max(1)[0].unsqueeze(1)
                qt = rewards + (self.gamma * max_action_values * (1 - dones))
                qe = self.Q(states).gather(1, actions)

                loss = F.mse_loss(qe, qt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.soft_update_fq(self.Q, self.FQ)

    def soft_update_fq(self, q, fq):
        for q_params, fq_params in zip(q.parameters(), fq.parameters()):
            fq_params.data.copy_(self.tau * q_params.data + (1.0 - self.tau) * fq_params.data)

    def epsilon_greedy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.env.action_space.n)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            # Setting network to eval mode
            self.Q.eval()
            with torch.no_grad():
                action_values = self.Q(state)
            # Back to training mode
            self.Q.train()
            return np.argmax(action_values.cpu().data.numpy())

    def save_agent(self, filename="agent.pth"):
        torch.save(self.Q.state_dict(), filename)


if __name__ == "__main__":
    agent = LearningAgent()

    eps, eps_decay = 1, 0.9999
    rewards, avg_rewards, eps_list = [], [], []
    episode_cap = 1000

    # Running learning algorithm
    for episode in range(agent.episodes):
        # Resetting env for every episode
        state = agent.env.reset()
        acc_reward = 0

        # Stepping through an episode
        for _ in range(episode_cap):
            action = agent.epsilon_greedy(state, eps)
            observation, reward, done, info = agent.env.step(action)
            agent.step(state, action, reward, observation, done)

            state = observation
            acc_reward += reward
            if done:
                break
            eps = max(eps * eps_decay, 0.01)
            if episode % 50 == 0:
                agent.env.render()
        rewards.append(acc_reward)
        avg_rew = np.average(rewards[-100:])
        eps_list.append(eps)
        avg_rewards.append(avg_rew)
        print(
            "Avg Reward:", avg_rew,
            "Current Rewards:", acc_reward,
            "EPS:", eps,
            "Buffer:", len(agent.replay_buffer)
        )
        # UnComment below if statement for stop early stopping
        # if avg_rewards[-1] >= 200:
        #     break
    agent.save_agent("agent.pth")

    plt.figure(0)
    plt.title('Training Rewards Plot')
    plt.xlabel('# Episodes')
    plt.ylabel('Reward')
    plt.plot(rewards, label='Rewards')
    plt.plot(avg_rewards, label='AvgRewards')
    plt.legend(loc="best")
    plt.savefig("graphs/training.png")

    # EPS Decay
    plt.figure(1)
    plt.title('Epsilon Decay Vs Rewards Plot')
    plt.xlabel('Epsilon')
    plt.ylabel('Rewards')
    axes = plt.axes()
    axes.invert_xaxis()
    eps_decays = [0.9, 0.99, 0.999, 0.9999, 0.99999]
    for eps_decay in eps_decays:
        agent, eps = LearningAgent(), 1
        rewards, avg_rewards, eps_list = [], [], []

        # Running learning algorithm
        for episode in range(agent.episodes):
            # Resetting env for every episode
            state = agent.env.reset()
            acc_reward = 0

            # Stepping through an episode
            for _ in range(episode_cap):
                action = agent.epsilon_greedy(state, eps)
                observation, reward, done, info = agent.env.step(action)
                agent.step(state, action, reward, observation, done)

                state = observation
                acc_reward += reward
                if done:
                    break
                eps = max(eps * eps_decay, 0.01)
            rewards.append(acc_reward)
            avg_rew = np.average(rewards[-100:])
            eps_list.append(eps)
            avg_rewards.append(avg_rew)
            print(
                "Avg Reward:", avg_rew,
                "Current Rewards:", acc_reward,
                "EPS:", eps,
                "Buffer:", len(agent.replay_buffer)
            )
        plt.plot(eps_list, avg_rewards, label='ε - %s' % eps_decay)
    plt.legend(loc="best")
    plt.savefig("graphs/eps_decay.png")

    # Buffer Sizes
    plt.figure(2)
    plt.title('Experience Replay Buffer Size Vs Rewards Plot')
    plt.xlabel('Experience')
    plt.ylabel('Rewards')
    buffer_sizes = [10, 100, 1000, 10000, 100000]
    for b_size in buffer_sizes:
        agent, eps = LearningAgent(), 1
        rewards, avg_rewards, eps_list = [], [], []
        agent.replay_buffer_size = b_size

        # Running learning algorithm
        for episode in range(agent.episodes):
            # Resetting env for every episode
            state = agent.env.reset()
            acc_reward = 0

            # Stepping through an episode
            for _ in range(episode_cap):
                action = agent.epsilon_greedy(state, eps)
                observation, reward, done, info = agent.env.step(action)
                agent.step(state, action, reward, observation, done)

                state = observation
                acc_reward += reward
                if done:
                    break
                eps = max(eps * eps_decay, 0.01)
            rewards.append(acc_reward)
            avg_rew = np.average(rewards[-100:])
            eps_list.append(eps)
            avg_rewards.append(avg_rew)
            print(
                "Avg Reward:", avg_rew,
                "Current Rewards:", acc_reward,
                "EPS:", eps,
                "Buffer:", len(agent.replay_buffer)
            )
        plt.plot(avg_rewards, label='Buffer Size - %s' % b_size)
    plt.legend(loc="best")
    plt.savefig("graphs/buffer_size.png")

    # Soft Updates
    plt.figure(3)
    plt.title('Soft Update Factor τ Vs Rewards Plot')
    plt.xlabel('τ')
    plt.ylabel('Rewards')
    taus = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    for tau in taus:
        agent, eps = LearningAgent(), 1
        rewards, avg_rewards, eps_list = [], [], []
        agent.tau = tau

        # Running learning algorithm
        for episode in range(agent.episodes):
            # Resetting env for every episode
            state = agent.env.reset()
            acc_reward = 0

            # Stepping through an episode
            for _ in range(episode_cap):
                action = agent.epsilon_greedy(state, eps)
                observation, reward, done, info = agent.env.step(action)
                agent.step(state, action, reward, observation, done)

                state = observation
                acc_reward += reward
                if done:
                    break
                eps = max(eps * eps_decay, 0.01)
            rewards.append(acc_reward)
            avg_rew = np.average(rewards[-100:])
            eps_list.append(eps)
            avg_rewards.append(avg_rew)
            print(
                "Avg Reward:", avg_rew,
                "Current Rewards:", acc_reward,
                "EPS:", eps,
                "Buffer:", len(agent.replay_buffer)
            )
        plt.plot(avg_rewards, label='τ value - %s' % tau)
    plt.legend(loc="best")
    plt.savefig("graphs/soft_update_factor.png")
