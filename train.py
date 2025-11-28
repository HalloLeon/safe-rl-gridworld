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

    # step is 0-based, so total steps = step + 1
    return total_reward, info, step + 1


def build_shielded_env() -> tuple[GridWorld, GridConfigFactory]:
    # 1. Random grid with walls and guards
    config = GridConfigFactory.random_config(n_rows=8, n_cols=8)
    env = GridWorld(config)

    # 2. Build DFA for the safety property:
    #    - Label 0 = SAFE
    #    - Labels 1,2 = UNSAFE (wall / visible)
    actions: list[Action] = list(range(len(ACTIONS)))
    safe_labels: list[Label] = [0]
    unsafe_labels: list[Label] = [1, 2]

    dfa = build_simple_dfa(
        safe_labels=safe_labels,
        unsafe_labels=unsafe_labels,
        actions=actions,
    )

    # 3. Build safety game and compute winning region
    solver = SafetyGameSolver(
        dfa=dfa,
        actions=actions,
        mdp_next=env.peek_step,
        compute_label=env.compute_label,
    )

    # Use the environment's initial MDP state
    initial_mdp_state = env.cur_state
    winning_region = solver.compute_winning_region(initial_mdp_state)

    # 4. Build shield and attach to env
    shield = SafetyShield(dfa, winning_region, env.peek_step, env.compute_label)
    env.shield = shield

    return env, config


def rolling_avg(x: list[int], window: int = 25):
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
    plt.plot(rolling_avg(rewards, rolling_window))
    plt.title(f"Average Reward without Shield (rolling window={rolling_window})")
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
