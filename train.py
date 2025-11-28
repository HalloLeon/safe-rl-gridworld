import os

import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent
from common.constants import ACTIONS
from common.types import Action
from common.types import Label
from gridworld import GridWorld
from gridworld import GridConfigFactory
from shield import SafetyShield
from shield_synthesis.automaton.dfa import build_simple_dfa
from shield_synthesis.safety_game import SafetyGameSolver


def run_episode(
    env: GridWorld, agent: QLearningAgent, max_steps: int = 100
) -> tuple[float, dict, int]:
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, done, info = env.next_step(action)
        agent.update_q_value(state, action, reward, next_state)
        total_reward += reward
        state = next_state

        if done:
            break

    agent.decay_exploration()

    return total_reward, info, step + 1


def rolling_sum(x, window=25):
    x = np.array(x)

    if x.size < window:
        return x

    cumsum = np.cumsum(np.insert(x, 0, 0))
    avg = (cumsum[window:] - cumsum[:-window]) / float(window)

    return avg


def train_baseline():
    config = GridConfigFactory.random_config(n_rows=8, n_cols=8)
    env = GridWorld(config)

    n_states = config.n_rows * config.n_cols
    n_actions = len(GridWorld.ACTIONS)

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        seed=0,
    )

    n_episodes = 1000
    rewards = []
    steps = []
    unsafe_flags = []

    for _ in range(n_episodes):
        total_reward, info, total_steps = run_episode(env, agent)
        rewards.append(total_reward)
        unsafe_flags.append(info["hazard"])
        steps.append(total_steps)

    rolling_window = 25

    plt.figure()
    plt.plot(rolling_sum(rewards, rolling_window))
    plt.title(f"Average Reward (rolling window={rolling_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("plots/reward.png", dpi=150)

    plt.figure()
    plt.plot(rolling_sum(unsafe_flags, rolling_window))
    plt.title(f"Unsafe Episodes Fraction (rolling window={rolling_window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.savefig("plots/unsafe.png", dpi=150)

    plt.figure()
    plt.plot(rolling_sum(steps, rolling_window))
    plt.title(f"Average Steps (rolling window={rolling_window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.savefig("plots/steps.png", dpi=150)


if __name__ == "__main__":
    train_baseline()
