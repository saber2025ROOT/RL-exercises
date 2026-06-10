"""
Soft Actor-Critic (SAC)

"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Any, Deque, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


def set_seed(env: gym.Env, seed: int = 0) -> None:
    env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class ReplayBuffer:
    """
    Replay buffer for off-policy learning.

    SAC is off-policy, so it does not train only on the most recent trajectory.
    Instead, it stores transitions and repeatedly samples mini-batches from them.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = (
            deque(maxlen=capacity)
        )

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-network for SAC.

    The critic estimates Q(s, a), so both the state and action are used as input.
    SAC uses two independent Q-networks to reduce overestimation bias.
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if action.dim() == 1:
            action = action.unsqueeze(0)

        state = state.view(state.size(0), -1)
        action = action.view(action.size(0), -1)

        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy with tanh squashing.

    The network outputs mean and log standard deviation.
    An action is sampled using the reparameterization trick, then squashed with tanh
    and scaled to match the environment action bounds.
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, self.action_dim)
        self.log_std_layer = nn.Linear(hidden_size, self.action_dim)

        action_scale = (action_space.high - action_space.low) / 2.0
        action_bias = (action_space.high + action_space.low) / 2.0

        self.register_buffer(
            "action_scale",
            torch.tensor(action_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(action_bias, dtype=torch.float32),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        state = state.view(state.size(0), -1)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and return its corrected log-probability.

        The log-probability correction is necessary because tanh changes the
        probability density.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()

        tanh_action = torch.tanh(z)
        action = tanh_action * self.action_scale + self.action_bias

        log_prob = normal.log_prob(z)

        # Tanh-squash correction.
        log_prob -= torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation.

        We use tanh(mean) instead of sampling.
        """
        mean, _ = self.forward(state)
        tanh_action = torch.tanh(mean)
        action = tanh_action * self.action_scale + self.action_bias

        return action


class SACAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        replay_size: int = 1_000_000,
        batch_size: int = 256,
        start_steps: int = 10_000,
        update_after: int = 1_000,
        update_every: int = 1,
        seed: int = 0,
        hidden_size: int = 256,
    ) -> None:
        set_seed(env, seed)

        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                "SACAgent expects a continuous Box action space. "
                "For LunarLander-v3, use gym.make('LunarLander-v3', continuous=True)."
            )

        self.env = env
        self.seed = seed

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every

        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.policy = GaussianPolicy(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        self.q1 = QNetwork(env.observation_space, env.action_space, hidden_size)
        self.q2 = QNetwork(env.observation_space, env.action_space, hidden_size)

        self.q1_target = QNetwork(env.observation_space, env.action_space, hidden_size)
        self.q2_target = QNetwork(env.observation_space, env.action_space, hidden_size)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr_critic)

        if self.automatic_entropy_tuning:
            self.target_entropy = -float(np.prod(env.action_space.shape))
            self.log_alpha = torch.tensor(
                np.log(alpha),
                dtype=torch.float32,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha

        self.replay_buffer = ReplayBuffer(replay_size)

        self.eval_steps = []
        self.eval_returns = []
        self.eval_stds = []

    @property
    def current_alpha(self) -> torch.Tensor:
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()

        return torch.tensor(self.alpha, dtype=torch.float32)

    def predict(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action_t = self.policy.deterministic_action(state_t)
            else:
                action_t, _ = self.policy.sample(state_t)

        return action_t.squeeze(0).cpu().numpy()

    def soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def update(self) -> Tuple[float, float, float, float]:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        alpha = self.current_alpha

        # ------------------------------------------------------------
        # Critic target:
        # y = r + gamma * (1 - done) *
        #     (min(Q_target_1, Q_target_2) - alpha * log pi(a'|s'))
        # ------------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)

            next_q1 = self.q1_target(next_states, next_actions)
            next_q2 = self.q2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)

            target_q = rewards + self.gamma * (1.0 - dones) * (
                next_q - alpha.detach() * next_log_probs
            )

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ------------------------------------------------------------
        # Actor objective:
        # maximize Q(s, a) + entropy
        # implemented as minimizing alpha * log_pi - min(Q1, Q2)
        # ------------------------------------------------------------
        new_actions, log_probs = self.policy.sample(states)

        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (alpha.detach() * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ------------------------------------------------------------
        # Automatic entropy tuning:
        # alpha increases if entropy is too low, and decreases if entropy is too high.
        # ------------------------------------------------------------
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha_value = self.current_alpha.item()
            alpha_loss_value = alpha_loss.item()
        else:
            alpha_value = float(self.alpha)
            alpha_loss_value = 0.0

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        return (
            policy_loss.item(),
            q1_loss.item() + q2_loss.item(),
            alpha_loss_value,
            alpha_value,
        )

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10_000,
        eval_episodes: int = 5,
    ) -> None:
        eval_env = gym.make(self.env.spec.id, continuous=True)
        set_seed(eval_env, self.seed)

        state, _ = self.env.reset(seed=self.seed)
        episode_return = 0.0
        episode_step = 0

        last_policy_loss = 0.0
        last_q_loss = 0.0
        last_alpha_loss = 0.0
        last_alpha = float(self.current_alpha.item())

        for step in range(1, total_steps + 1):
            # Initial random exploration improves replay-buffer diversity.
            if step <= self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.predict(state, deterministic=False)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.push(
                state,
                action,
                float(reward),
                next_state,
                float(done),
            )

            state = next_state
            episode_return += float(reward)
            episode_step += 1

            if done:
                print(
                    f"[Train] Step {step:6d} "
                    f"EpisodeReturn {episode_return:8.1f} "
                    f"EpisodeLength {episode_step:4d}"
                )

                state, _ = self.env.reset()
                episode_return = 0.0
                episode_step = 0

            if (
                step >= self.update_after
                and len(self.replay_buffer) >= self.batch_size
                and step % self.update_every == 0
            ):
                (
                    last_policy_loss,
                    last_q_loss,
                    last_alpha_loss,
                    last_alpha,
                ) = self.update()

            if step % eval_interval == 0:
                mean_r, std_r = self.evaluate(eval_env, num_episodes=eval_episodes)

                self.eval_steps.append(step)
                self.eval_returns.append(mean_r)
                self.eval_stds.append(std_r)

                print(
                    f"[Eval ] Step {step:6d} "
                    f"AvgReturn {mean_r:8.1f} ± {std_r:6.1f} "
                    f"PolicyLoss {last_policy_loss:.3f} "
                    f"QLoss {last_q_loss:.3f} "
                    f"AlphaLoss {last_alpha_loss:.3f} "
                    f"Alpha {last_alpha:.3f}"
                )

        np.savez(
            f"sac_{self.env.spec.id}_{self.seed}_results.npz",
            steps=np.array(self.eval_steps),
            returns=np.array(self.eval_returns),
            stds=np.array(self.eval_stds),
        )

        print("Training complete.")

    def evaluate(
        self,
        eval_env: gym.Env,
        num_episodes: int = 10,
    ) -> Tuple[float, float]:
        returns = []

        for episode in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed + episode)
            done = False
            total_r = 0.0

            while not done:
                action = self.predict(state, deterministic=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_r += float(reward)

            returns.append(total_r)

        return float(np.mean(returns)), float(np.std(returns))


@hydra.main(config_path="../configs/agent/", config_name="sac", version_base="1.1")
def main(cfg: DictConfig) -> None:
    if cfg.env.get("continuous", False):
        env = gym.make(cfg.env.name, continuous=True)
    else:
        env = gym.make(cfg.env.name)

    set_seed(env, cfg.seed)

    agent = SACAgent(
        env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        lr_alpha=cfg.agent.lr_alpha,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        alpha=cfg.agent.alpha,
        automatic_entropy_tuning=cfg.agent.automatic_entropy_tuning,
        replay_size=cfg.agent.replay_size,
        batch_size=cfg.agent.batch_size,
        start_steps=cfg.agent.start_steps,
        update_after=cfg.agent.update_after,
        update_every=cfg.agent.update_every,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
    )

    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()