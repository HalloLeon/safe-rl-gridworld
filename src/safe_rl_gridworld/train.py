import os
import time
from typing import Optional
from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from safe_rl_gridworld.core.agent import QLearningAgent
from safe_rl_gridworld.common.constants import ACTIONS
from safe_rl_gridworld.common.constants import VERBOSE
from safe_rl_gridworld.core.gridworld import GridConfig
from safe_rl_gridworld.core.gridworld import GridWorld
from safe_rl_gridworld.core.gridworld import GridConfigFactory
from safe_rl_gridworld.core.shield import SafetyShield
from safe_rl_gridworld.core.shield_synthesis.automaton.dfa import build_simple_dfa
from safe_rl_gridworld.core.shield_synthesis.safety_game import SafetyGameSolver


def evaluate_shield_effectiveness(
    config: GridConfig,
    verbose: bool = False,
):
    """
    Run a comparison experiment between shielded and unshielded training.

    This function:
      - Prints the grid configuration for visual inspection,
      - Trains an unshielded Q-learning agent,
      - Trains a shielded agent on the same environment,
      - Measures rewards, steps, and safety violations for both,
      - Produces a combined 3-panel plot comparing the two training curves
        and saves it to plots/shielded_vs_unshielded.png".

    The plot includes:
      * Smoothed episode reward,
      * Fraction of episodes in which the agent was caught,
      * Episode length (steps).

    Args:
        config:
            A `GridConfig` instance describing the environment layout
            (walls, guards, starting position, goals, etc.).
        verbose:
            If True, prints debug information and training time stats.

    Raises:
        RuntimeError:
            If the shielded environment cannot be constructed because its
            initial state is not part of the safety game's winning region.
    """

    VERBOSE = verbose

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
            print("Starting unshielded training...")

        unshielded_rewards, unshielded_steps, unshielded_unsafe_flags = train(
            config, shielded=False
        )

        if VERBOSE:
            end = time.perf_counter()
            print(f"Unshielded training time: {end - start:.2f} seconds\n")
            start = time.perf_counter()
            print("Starting shielded training...")

        shielded_rewards, shielded_steps, shielded_unsafe_flags = train(
            config, shielded=True
        )

        if VERBOSE:
            end = time.perf_counter()
            print(f"Shielded training time: {end - start:.2f} seconds\n")
    except RuntimeError as e:
        print(e)
        exit(1)

    fig, (ax_reward, ax_unsafe, ax_steps) = plot_results(
        config,
        unshielded_rewards,
        unshielded_steps,
        unshielded_unsafe_flags,
        label="unshielded",
    )

    plot_results(
        config,
        shielded_rewards,
        shielded_steps,
        shielded_unsafe_flags,
        label="shielded",
        fig=fig,
        axes=(ax_reward, ax_unsafe, ax_steps),
    )

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/shielded_vs_unshielded.png", dpi=150)


def train(
    config: GridConfig,
    shielded: bool = True,
    n_episodes: int = 500,
    n_steps: int = 100,
) -> tuple[list[float], list[int], list[int]]:
    """
    Train a Q-learning agent in a (possibly shielded) gridworld environment.

    This function sets up a `GridWorld` based on the given configuration,
    optionally augments it with a safety shield, and then trains a single
    Q-learning agent for a fixed number of episodes.

    Args:
        config:
            Static grid configuration describing size, walls, goals, and guards.
        shielded:
            If True, wraps the environment with a safety shield constructed
            from a safety game winning region. If False, uses the bare
            `GridWorld` without shielding.
        n_episodes:
            Number of training episodes.
        n_steps:
            Maximum number of environment steps per episode.

    Returns:
        (rewards, steps, unsafe_flags):
            - rewards: Total reward obtained in each episode.
            - steps: Number of steps taken in each episode.
            - unsafe_flags: 1 if the agent was caught in the episode,
              0 otherwise.
    """

    if shielded:
        env, config = _build_shielded_env(config)
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
        total_reward, info, total_steps = _run_episode(env, agent, n_steps)
        rewards.append(total_reward)
        steps.append(total_steps)
        unsafe_flags.append(1 if info.get("caught", False) else 0)

    return rewards, steps, unsafe_flags


def _build_shielded_env(config: GridConfig) -> tuple[GridWorld, GridConfigFactory]:
    """
    Construct a shielded `GridWorld` from a given configuration.

    This builds the environment, creates a DFA encoding the safety property
    (SAFE = label 0, UNSAFE = labels 1 and 2), solves the corresponding safety
    game to obtain its winning region, verifies that the initial state is
    winning, and attaches a `SafetyShield` to the environment.

    Args:
        config: Static grid configuration used to instantiate the environment.

    Returns:
        (env, config): The shielded environment and the original configuration.

    Raises:
        RuntimeError:
            If the initial product state (initial MDP state + DFA initial state)
            is not included in the computed winning region. This indicates that
            the configuration cannot satisfy the safety specification.
    """

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


def _run_episode(
    env: GridWorld, agent: QLearningAgent, max_steps: int = 100
) -> tuple[float, dict, int]:
    """
    Run a single training episode in the given environment.

    The episode starts from a fresh environment reset and proceeds for at most
    `max_steps` steps, or until the environment signals termination.

    At each step:
      - the agent chooses an action using its exploration–exploitation policy,
      - the environment transitions to a new state and returns reward,
      - the agent updates its Q-value for the observed transition.

    At the end of the episode, the agent's exploration rate is decayed once.

    Args:
        env:
            The `GridWorld` instance (shielded or unshielded) to interact with.
        agent:
            The `QLearningAgent` to train.
        max_steps:
            Maximum number of time steps for this episode.

    Returns:
        (total_reward, info, total_steps):
            - total_reward: Sum of rewards over the episode,
            - info: Dictionary returned by the environment
            - total_steps: Number of steps actually executed
              (between 1 and `max_steps`).
    """

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
    config: GridConfig,
    rewards: list[float],
    steps: list[int],
    unsafe_flags: list[int],
    label: str,
    fig: Optional[Figure] = None,
    axes: Optional[Sequence[Axes]] = None,
    rolling_window: int = 25,
) -> tuple[Figure, Sequence[Axes]]:
    """
    Plot training statistics for a single run, optionally overlaying
    multiple runs on the same axes.

    This function draws three smoothed curves (using a rolling average):
      1. Reward per episode,
      2. Fraction of episodes in which the agent was caught,
      3. Number of steps per episode.

    If `axes` is None, a new figure with three subplots is created. If
    `axes` is provided, the curves are drawn into those existing axes,
    which makes it easy to compare multiple runs (e.g. shielded vs
    unshielded) by calling this function multiple times with the same
    axes.

    No files are saved here; the caller is responsible for saving
    the figure (e.g. via `fig.savefig(...)`).

    Args:
        config:
            Grid configuration used for the experiment.
        rewards:
            List of total episode rewards, one per episode.
        steps:
            List of episode lengths (in time steps), one per episode.
        unsafe_flags:
            List of episode-level flags, where 1 indicates the agent was
            caught in that episode, and 0 otherwise.
        label:
            Label for this run (e.g. "unshielded", "shielded"), used in
            the legend for each subplot.
        fig:
            Optional figure to draw into. If provided, `axes` must also
            be provided and correspond to this figure. If None, a new
            figure is created.
        axes:
            Optional sequence of three `Axes` objects,
            `(reward_axis, unsafe_axis, steps_axis)`. If provided, plots
            are drawn into these axes (and `fig` should be the figure
            they belong to). If None, a new set of axes is created along
            with a new figure.
        rolling_window:
            Window size for computing the rolling average that is plotted.

    Returns:
        (fig, axes):
            The figure and axes containing the
            plots. If `fig` and `axes` were provided, the same objects are
            returned; otherwise, newly created ones are returned.
    """

    # Create axes if not provided
    if fig is None or axes is None:
        fig, (ax_reward, ax_unsafe, ax_steps) = plt.subplots(
            3, 1, sharex=True, figsize=(7, 9)
        )

        # Add configuration context as a figure-level title
        n_guards = len(config.guards)
        n_walls = len(config.walls)
        n_goals = len(config.goals)

        fig.suptitle(
            (
                f"Grid {config.n_rows}×{config.n_cols} "
                f"(guards={n_guards}, walls={n_walls}, goals={n_goals})"
            ),
            fontsize=12,
            y=0.98,
        )
    else:
        ax_reward, ax_unsafe, ax_steps = axes

    # Smooth data
    reward_smoothed = _rolling_avg(rewards, rolling_window)
    unsafe_smoothed = _rolling_avg(unsafe_flags, rolling_window)
    steps_smoothed = _rolling_avg(steps, rolling_window)

    # X-axis indices
    episodes_reward = np.arange(len(reward_smoothed))
    episodes_unsafe = np.arange(len(unsafe_smoothed))
    episodes_steps = np.arange(len(steps_smoothed))

    # Reward plot
    ax_reward.plot(episodes_reward, reward_smoothed, label=label)
    ax_reward.set_title(f"Reward (rolling window={rolling_window})")
    ax_reward.set_ylabel("Reward")

    # Fraction caught plot
    ax_unsafe.plot(episodes_unsafe, unsafe_smoothed, label=label)
    ax_unsafe.set_title(
        f"Fraction of Episodes Caught (rolling window={rolling_window})"
    )
    ax_unsafe.set_ylabel("Fraction caught")

    # Steps plot
    ax_steps.plot(episodes_steps, steps_smoothed, label=label)
    ax_steps.set_title(f"Steps per Episode (rolling window={rolling_window})")
    ax_steps.set_xlabel("Episode")
    ax_steps.set_ylabel("Steps")

    return fig, (ax_reward, ax_unsafe, ax_steps)


def _rolling_avg(x: list[float] | list[int], window: int = 10) -> np.ndarray:
    """
    Compute a simple rolling average over a 1D sequence.

    If the sequence length is at least `window`, the result contains the mean of
    each consecutive window of size `window`, i.e. it has length
    `len(x) - window + 1`.

    If the input length is smaller than `window`, the raw data is returned
    unchanged (as a NumPy array).

    Args:
        x:
            Input sequence of numeric values (e.g. rewards or steps).
        window:
            Size of the rolling window.

    Returns:
        A NumPy array of rolling averages, or the original data converted
        to a NumPy array if `len(x) < window`.
    """

    x = np.array(x)

    if x.size < window:
        return x

    cumsum = np.cumsum(np.insert(x, 0, 0))
    avg = (cumsum[window:] - cumsum[:-window]) / float(window)

    return avg


def print_grid_config(config: GridConfig) -> None:
    """
    Pretty-print a static `GridConfig` as an ASCII map.

    The map is printed row by row to stdout using a small legend:
        S           start position
        G           goal cell
        #           wall
        ^ v < >     guard, oriented up / down / left / right
        .           empty cell

    Args:
        config: Static grid configuration whose layout should be printed.
    """

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

    print("")  # Blank line after the grid


if __name__ == "__main__":
    config = GridConfigFactory.build_random_config(7, 7, n_guards=1, seed=42)
    evaluate_shield_effectiveness(config, verbose=True)
