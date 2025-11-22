import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        seed=0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=float)

    def get_action(self, state):
        if self.rng.random() < self.exploration_rate:
            return self.rng.integers(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state]))

    def update_q_value(self, state, action, reward, next_state):
        best_next = max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[state, action]
        new_q_value = self.q_table[state, action] + self.learning_rate * td_error
        self.q_table[state, action] = new_q_value

    def decay_exploration(self):
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )
