import random
from typing import Optional

import numpy as np

from safe_rl_gridworld.common.types import Action


class QLearningAgent:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.

    Args:
        n_states: Number of discrete environment states.
        n_actions: Number of discrete actions.
        learning_rate: Step size α in the Q-learning update.
        discount_factor: Discount factor γ for future rewards.
        exploration_rate: Current ε for ε-greedy action selection.
        exploration_decay: Multiplicative decay factor applied to ε after each episode.
        min_exploration_rate: Lower bound on ε.
        rng: Random number generator used for exploration.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        rng: Optional[random.Random] = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # ε-greedy exploration parameters
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        self.rng = rng or random.Random()

        # Q-values with shape (n_states, n_actions)
        self.q_table = np.zeros((n_states, n_actions), dtype=float)

    def get_action(self, state_index: int) -> Action:
        """
        Choose an action given the state index using ε-greedy strategy.

        With probability ε: choose a random action.
        With probability 1 - ε: choose the greedy action argmax_a Q(state, a).
        """

        if self.rng.random() < self.exploration_rate:
            return self.rng.randrange(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state_index]))

    def update_q_value(
        self, state_index: int, action: Action, reward: float, next_state_index: int
    ) -> None:
        """
        Perform the Q-learning update:
            Q(s, a) = Q(s, a) + α [ r + γ max_a' Q(s', a') - Q(s, a) ]

        Args:
            state_index: Current state index s.
            action: Action a taken in state s.
            reward: Immediate reward r received.
            next_state_index: Next state index s' observed.
        """

        best_next = max(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[state_index, action]
        new_q_value = self.q_table[state_index, action] + self.learning_rate * td_error
        self.q_table[state_index, action] = new_q_value

    def decay_exploration(self) -> None:
        """
        Decay ε after an episode, but never go below min_exploration_rate.
        """

        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )
