from __future__ import annotations

import numpy as np
from rl_exercises.week_3.randomwalkenv import RandomWalkENV
from rl_exercises.week_3.TDLambdaAgent import TDLambdaAgent


def run_td_lambda(
    lam: float = 0.8,
    alpha: float = 0.1,
    episodes: int = 100,
    seed: int = 0,
) -> None:
    env = RandomWalkENV(seed=seed)
    agent = TDLambdaAgent(env=env, alpha=alpha, gamma=1.0, lam=lam)

    for s in [1, 2, 3, 4, 5]:
        agent.V[s] = 0.5

    for _ in range(episodes):
        state, _ = env.reset()
        agent.reset_episode()

        done = False

        while not done:
            action, _ = agent.predict_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            batch = [(state, action, reward, next_state, done, info)]
            agent.update_agent(batch)

            state = next_state

    learned = np.array([agent.V[s] for s in [1, 2, 3, 4, 5]])
    true = env.true_values()
    rmse = np.sqrt(np.mean((learned - true) ** 2))

    print(f"lambda={lam}, alpha={alpha}, episodes={episodes}")
    print("learned:", learned)
    print("true:   ", true)
    print("rmse:   ", rmse)


if __name__ == "__main__":
    for lam in [0.0, 0.3, 0.8, 1.0]:
        run_td_lambda(lam=lam, alpha=0.1, episodes=100, seed=0)
