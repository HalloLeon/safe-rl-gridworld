import os
import time

import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent
from common.constants import ACTIONS
from common.constants import VERBOSE
from gridworld import GridConfig
from gridworld import GridWorld
from gridworld import GridConfigFactory
from shield import SafetyShield
from shield_synthesis.automaton.dfa import build_simple_dfa
from shield_synthesis.safety_game import SafetyGameSolver


def train(
    config: GridConfig,
    shielded: bool = True,
    n_episodes: int = 500,
    n_steps: int = 100,
) -> tuple[list[float], list[int], list[int]]:
    if shielded:
        env, config = build_shielded_env(config)
    else:
        env = GridWorld(config)

    n_states = config.n_rows * config.n_cols
    n_actions = len(ACTIONS)

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        rng=env.rng,
    )

    rewards = []
    steps = []
    unsafe_flags = []  # 1 if caught in this episode, else 0

    for _ in range(n_episodes):
        total_reward, info, total_steps = run_episode(env, agent, n_steps)
        rewards.append(total_reward)
        steps.append(total_steps)
        unsafe_flags.append(1 if info.get("caught", False) else 0)

    return rewards, steps, unsafe_flags


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

    # Step is 0-based, so total steps = step + 1
    return total_reward, info, step + 1


def plot_results(
    rewards: list[float],
    steps: list[int],
    unsafe_flags: list[int],
    file_prefix: str,
    rolling_window: int = 25,
) -> None:
    os.makedirs("plots", exist_ok=True)

    # Plot average reward
    plt.figure()
    plt.plot(rolling_avg(rewards, rolling_window))
    plt.title(f"Reward (rolling window={rolling_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}reward.png", dpi=150)

    # Plot fraction of episodes where the agent was caught
    plt.figure()
    plt.plot(rolling_avg(unsafe_flags, rolling_window))
    plt.title(
        f"Fraction of Episodes Caught without Shield (rolling window={rolling_window})"
    )
    plt.xlabel("Episode")
    plt.ylabel("Fraction caught")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}unsafe.png", dpi=150)

    # Plot average number of steps per episode
    plt.figure()
    plt.plot(rolling_avg(steps, rolling_window))
    plt.title(f"Steps (rolling window={rolling_window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.tight_layout()
    plt.savefig(f"plots/{file_prefix}steps.png", dpi=150)


def rolling_avg(x: list[int], window: int = 10) -> np.ndarray:
    x = np.array(x)

    if x.size < window:
        return x

    cumsum = np.cumsum(np.insert(x, 0, 0))
    avg = (cumsum[window:] - cumsum[:-window]) / float(window)

    return avg


def build_shielded_env(config: GridConfig) -> tuple[GridWorld, GridConfigFactory]:
    # 1. Random grid with walls and guards
    env = GridWorld(config)

    # 2. Build DFA for the safety property:
    #    - Label 0 = SAFE
    #    - Labels 1,2 = UNSAFE (wall / visible)
    actions = list(range(len(ACTIONS)))
    safe_labels = [0]
    unsafe_labels = [1, 2]

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

    if (env.initial_state, dfa.initial) not in winning_region:
        raise RuntimeError("Error: Initial state is not in the winning region!")

    # 4. Build shield and attach to env
    shield = SafetyShield(dfa, winning_region, env.peek_step, env.compute_label)
    env.shield = shield

    return env, config


def print_grid_config(config: GridConfig) -> None:
    # Pretty-print a static GridConfig as an ASCII map.
    #
    # Legend:
    #     S  = start
    #     G  = goal
    #     #  = wall
    #     ^v<> = guard (facing up/down/left/right)
    #     .  = empty cell
    #
    # Map guard facing index -> char, assuming:
    # DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

    facing_chars = {
        0: "^",  # UP
        1: "v",  # DOWN
        2: "<",  # LEFT
        3: ">",  # RIGHT
    }

    # Build a 2D grid of '.' first
    grid = [["." for _ in range(config.n_cols)] for _ in range(config.n_rows)]

    # Walls
    for r, c in config.walls:
        if 0 <= r < config.n_rows and 0 <= c < config.n_cols:
            grid[r][c] = "#"

    # Goals
    for r, c in config.goals:
        if 0 <= r < config.n_rows and 0 <= c < config.n_cols:
            grid[r][c] = "G"

    # Guards
    for g_pos, facing_direction in config.guards:
        r, c = g_pos

        if 0 <= r < config.n_rows and 0 <= c < config.n_cols:
            ch = facing_chars.get(
                facing_direction, "X"
            )  # Fallback if facing is unexpected
            grid[r][c] = ch

    # Start (draw last so it's visible even if overlapping something else)
    sr, sc = config.start

    if 0 <= sr < config.n_rows and 0 <= sc < config.n_cols:
        grid[sr][sc] = "S"

    for r in range(config.n_rows):
        row_str = " ".join(f"{cell:2s}" for cell in grid[r])
        print(f"{row_str}")


if __name__ == "__main__":
    config = GridConfigFactory.build_random_config(
        n_rows=7, n_cols=7, walls_fraction=0.2, n_guards=2, seed=100
    )

    print_grid_config(config)

    try:
        if VERBOSE:
            print(
                f"Train config:\n"
                f"  Rows:       {config.n_rows}\n"
                f"  Cols:       {config.n_cols}\n"
                f"  Guards:     {len(config.guards)}\n"
            )

            start = time.perf_counter()

        unshielded_rewards, unshielded_steps, unshielded_unsafe_flags = train(
            config, shielded=False
        )

        if VERBOSE:
            end = time.perf_counter()
            print(f"Unshielded training time: {end - start:.2f} seconds\n")
            start = time.perf_counter()

        shielded_rewards, shielded_steps, shielded_unsafe_flags = train(
            config, shielded=True
        )

        if VERBOSE:
            end = time.perf_counter()
            print(f"Shielded training time: {end - start:.2f} seconds\n")
    except RuntimeError as e:
        print(e)
        exit(1)

    plot_results(
        unshielded_rewards,
        unshielded_steps,
        unshielded_unsafe_flags,
        file_prefix="unshielded_",
    )

    plot_results(
        shielded_rewards,
        shielded_steps,
        shielded_unsafe_flags,
        file_prefix="shielded_",
    )
