from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class RandomWalkENV(gym.Env):
    """


    States
    ------
    Discrete(7):
    - 0: A, left terminal
    - 1: B
    - 2: C
    - 3: D, start state
    - 4: E
    - 5: F
    - 6: G, right terminal

    Actions
    -------
    Discrete(2):
    - 0: go left
    - 1: go right



    Reward / Outcome
    ----------------
    - reaching A gives reward/outcome 0
    - reaching G gives reward/outcome 1
    - all intermediate transitions give reward 0

    Start State
    -----------
    D
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

        self.left_terminal = 0
        self.right_terminal = 6
        self.start_state = 3
        self.position = self.start_state

        self.observation_space = gym.spaces.Discrete(7)
        self.action_space = gym.spaces.Discrete(2)

        self.states = np.arange(self.observation_space.n)
        self.actions = np.arange(self.action_space.n)

        self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.position = self.start_state
        return self.position, {}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:

        if not self.action_space.contains(int(action)):
            raise RuntimeError(f"{action} is not a valid action")

        move = int(self.rng.choice([-1, 1]))
        self.position = self.get_next_state(self.position, move)

        terminated = self.position in [self.left_terminal, self.right_terminal]
        truncated = False

        reward = 1.0 if self.position == self.right_terminal else 0.0

        return self.position, reward, terminated, truncated, {}

    def get_next_state(self, state: int, move: int) -> int:

        if move == -1:
            return max(self.left_terminal, state - 1)

        if move == 1:
            return min(self.right_terminal, state + 1)

        raise RuntimeError(f"{move} is not a valid move")

    def get_transition_matrix(self) -> np.ndarray:

        nS = self.observation_space.n
        nA = self.action_space.n

        T = np.zeros((nS, nA, nS), dtype=float)

        for s in range(nS):
            for a in range(nA):
                if s in [self.left_terminal, self.right_terminal]:
                    T[s, a, s] = 1.0
                    continue

                left = self.get_next_state(s, -1)
                right = self.get_next_state(s, 1)

                T[s, a, left] += 0.5
                T[s, a, right] += 0.5

        return T

    def get_reward_per_action(self) -> np.ndarray:

        nS = self.observation_space.n
        nA = self.action_space.n

        R = np.zeros((nS, nA), dtype=float)
        T = self.get_transition_matrix()

        rewards = np.zeros(nS, dtype=float)
        rewards[self.right_terminal] = 1.0

        for s in range(nS):
            for a in range(nA):
                for next_s in range(nS):
                    R[s, a] += T[s, a, next_s] * rewards[next_s]

        return R

    @staticmethod
    def true_values() -> np.ndarray:
        """
        True values for non-terminal states B, C, D, E, F.

        These are probabilities of reaching G before A.
        """
        return np.array(
            [
                1 / 6,
                2 / 6,
                3 / 6,
                4 / 6,
                5 / 6,
            ],
            dtype=float,
        )

    def render(self, mode: str = "human"):
        labels = ["A", "B", "C", "D", "E", "F", "G"]
        print(f"[RandomWalkPrediction] state={labels[self.position]}")
