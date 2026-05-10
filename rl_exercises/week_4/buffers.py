from typing import Any, Dict, List, Tuple, Union

import numpy as np
from rl_exercises.agent import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    """
    Simple FIFO replay buffer.

    Stores tuples of (state, action, reward, next_state, done, info),
    and evicts the oldest when capacity is exceeded.
    """

    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        """
        super().__init__()
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.infos: List[Dict] = []

    def add(
        self,
        state: np.ndarray,
        action: Union[int, float],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """
        Add a single transition to the buffer.

        If the buffer is full, the oldest transition is removed.

        Parameters
        ----------
        state : np.ndarray
            Observation before action.
        action : int or float
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Observation after action.
        done : bool
            Whether episode terminated/truncated.
        info : dict
            Gym info dict (can store extras).
        """
        if len(self.states) >= self.capacity:
            # TODO: pop the oldest element off each list (states, actions, …, infos)
            # pop oldest
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)

        # TODO: append state, action, reward, next_state, done, info to their respective lists
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(next_state)
        self.dones.append(bool(done))
        self.infos.append(info)
    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict]]:
        """
        Uniformly sample a batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        List of transitions as (state, action, reward, next_state, done, info).
        """
        # TODO: randomly choose `batch_size` unique indices from [0, len(self.states))
        idxs = np.random.choice(len(self.states), size=batch_size, replace=False)

        return [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
            )
            for i in idxs
        ]

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.states)



class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.

    Instead of sampling transitions uniformly, transitions with larger
    TD-errors are sampled more frequently.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(capacity)

        # alpha controls how strongly priorities affect sampling
        # alpha = 0 -> uniform replay
        self.alpha = alpha

        # beta controls importance-sampling correction
        self.beta = beta

        # small constant to avoid zero priorities
        self.eps = eps

        # stores the priority value for each transition
        self.priorities: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: Union[int, float],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        # remove oldest transition if buffer is full
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)
            self.priorities.pop(0)

        # new transitions receive maximum priority
        # so they are replayed at least once
        max_priority = max(self.priorities) if len(self.priorities) > 0 else 1.0

        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(next_state)
        self.dones.append(bool(done))
        self.infos.append(info)

        self.priorities.append(max_priority)

    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict, int, float]]:
        # convert priorities to numpy array
        priorities = np.array(self.priorities, dtype=np.float32)

        # compute sampling probabilities
        scaled_priorities = priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()

        # sample transitions according to priority distribution
        idxs = np.random.choice(
            len(self.states),
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        # importance-sampling weights reduce replay bias
        weights = (len(self.states) * probabilities[idxs]) ** (-self.beta)

        # normalize weights for stability
        weights = weights / weights.max()

        return [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
                int(i),
                float(weights[j]),
            )
            for j, i in enumerate(idxs)
        ]

    def update_priorities(self, idxs: List[int], td_errors: np.ndarray) -> None:
        """
        Update transition priorities using the latest TD-errors.
        """

        for idx, td_error in zip(idxs, td_errors):
            self.priorities[idx] = float(abs(td_error) + self.eps)