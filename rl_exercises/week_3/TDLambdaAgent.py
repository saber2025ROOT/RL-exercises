from __future__ import annotations

from typing import Any, DefaultDict

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent

State = Any


class TDLambdaAgent(AbstractAgent):
    """TD(lambda) prediction agent .

    Learns V(s).
    """

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.8,
    ) -> None:
        assert alpha > 0, "alpha must be > 0"
        assert 0 <= gamma <= 1, "gamma must be in [0, 1]"
        assert 0 <= lam <= 1, "lambda must be in [0, 1]"

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

        self.V: DefaultDict[State, float] = defaultdict(float)
        self.episode_states: list[State] = []

    def predict_action(
        self,
        state: np.ndarray,
        info: dict = {},
        evaluate: bool = False,
    ) -> Any:  # type: ignore # noqa
        return self.env.action_space.sample(), info

    def reset_episode(self) -> None:
        self.episode_states = []

    def update_agent(self, batch) -> float:  # type: ignore
        state, _, reward, next_state, done, _ = batch[0]
        return self.TD_lambda(state, reward, next_state, done)

    def TD_lambda(
        self,
        state: State,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        self.episode_states.append(state)

        old_value = self.V[state]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.V[next_state]

        delta = target - old_value

        t = len(self.episode_states) - 1

        for k, past_state in enumerate(self.episode_states):
            distance = t - k
            weight = (self.gamma * self.lam) ** distance
            self.V[past_state] += self.alpha * delta * weight

        if done:
            self.reset_episode()

        return self.V[state]

    def save(self, path: str) -> Any:
        np.save(path, dict(self.V))

    def load(self, path: str) -> Any:
        loaded_v = np.load(path, allow_pickle=True).item()
        self.V = defaultdict(float, loaded_v)
