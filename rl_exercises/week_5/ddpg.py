import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self, size=100000):
        self.data = deque(maxlen=size)

    def push(self, s, a, r, ns, done):
        self.data.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, done = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(device)
        a = torch.tensor(np.array(a), dtype=torch.float32).to(device)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1).to(device)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).to(device)
        done = torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1).to(device)

        return s, a, r, ns, done

    def __len__(self):
        return len(self.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.max_action * x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    def __init__(
        self,
        env,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        noise_std=0.1,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = Buffer()

    def predict_action(self, state, evaluate=False):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]

        if not evaluate:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise

        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def update_agent(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        s, a, r, ns, done = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_a = self.actor_target(ns)
            next_q = self.critic_target(ns, next_a)
            y = r + self.gamma * (1 - done) * next_q

        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def evaluate(self, episodes=5):
        returns = []

        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total = 0.0

            while not done:
                action = self.predict_action(state, evaluate=True)
                ns, r, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total += r
                state = ns

            returns.append(total)

        return np.mean(returns), np.std(returns)

    def train(self, episodes=200, eval_interval=20):
        all_returns = []

        for ep in range(1, episodes + 1):
            state, _ = self.env.reset()
            done = False
            ep_return = 0.0

            while not done:
                action = self.predict_action(state)
                ns, r, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.push(state, action, r, ns, done)
                self.update_agent()

                ep_return += r
                state = ns

            all_returns.append(ep_return)

            if ep % 10 == 0:
                print(f"Episode {ep:3d} Return {ep_return:8.2f}")

            if ep % eval_interval == 0:
                mean_ret, std_ret = self.evaluate()
                print(
                    f"[Eval] Episode {ep:3d} AvgReturn {mean_ret:8.2f} ± {std_ret:6.2f}"
                )

        return all_returns


def set_seed(env, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def plot_returns(returns, filename="ddpg_pendulum.png"):
    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DDPG on Pendulum-v1")
    plt.savefig(filename)
    plt.close()


def main():
    env = gym.make("Pendulum-v1")
    set_seed(env, 0)

    agent = DDPGAgent(
        env=env,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        noise_std=0.1,
    )

    returns = agent.train(episodes=200, eval_interval=20)
    plot_returns(returns, "ddpg_pendulum.png")

    env.close()


if __name__ == "__main__":
    main()
