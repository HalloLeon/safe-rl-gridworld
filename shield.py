import random
from typing import Callable
from typing import Optional

from common.constants import ACTIONS
from common.types import DFAState
from common.types import MDPState
from shield_synthesis.automaton.dfa import DFA


class SafetyShield:
    def __init__(
        self,
        dfa: DFA,
        winning_region: set[tuple[MDPState, DFAState]],
        peek_mdp_step: Callable[[MDPState, int], MDPState],
        compute_mdp_label: Callable[[MDPState], int],
        rng: Optional[random.Random] = None,
    ):
        self.dfa = dfa
        self.winning_region = winning_region
        self.peek_mdp_step = peek_mdp_step
        self.compute_mdp_label = compute_mdp_label
        self.rng = rng if rng is not None else random.Random()

    def filter_action(self, mdp_state: MDPState, action: int) -> int:
        if self.is_action_safe(mdp_state, action):
            self._update_dfa_state(mdp_state, action)
            return action

        # Try all other actions, collect safe ones
        safe_actions = [
            a
            for a in range(len(ACTIONS))
            if self.is_action_safe(mdp_state, a)
        ]

        if safe_actions:
            selected_action = self.rng.choice(safe_actions)
            self._update_dfa_state(mdp_state, selected_action)

            return selected_action

        # No safe action found (should not happen if winning region is correct),
        # but keep DFA in sync with original action
        self._update_dfa_state(mdp_state, action)
        return action

    def is_action_safe(self, mdp_state: MDPState, action: int) -> bool:
        next_mdp_state = self.env.peek_step(mdp_state, action)
        next_label = self.env.compute_label(next_mdp_state)
        next_dfa_state = self.dfa.peek_next((next_label, action))

        # Long-term safety via winning region
        # (avoid states from which safety cannot be guaranteed)
        return (next_mdp_state, next_dfa_state) in self.winning_region

    def _update_dfa_state(self, mdp_state, action):
        next_mdp_state = self.peek_mdp_step(mdp_state, action)
        next_label = self.compute_mdp_label(next_mdp_state)
        self.dfa.next((next_label, action))
