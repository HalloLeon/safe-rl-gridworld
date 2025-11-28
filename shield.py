import random
from typing import Optional

from gridworld import GridWorld
from shield_synthesis.automaton.dfa import DFA
from shield_synthesis.automaton.dfa import DFAState
from shield_synthesis.safety_game import MDPState


class SafetyShield:
    def __init__(
        self,
        env: GridWorld,
        dfa: DFA,
        winning_region: set[tuple[MDPState, DFAState]],
        rng: Optional[random.Random] = None,
    ):
        self.env = env
        self.dfa = dfa
        self.winning_region = winning_region
        self.rng = rng if rng is not None else random.Random()

    def filter_action(self, mdp_state: MDPState, action: int) -> int:
        if self.is_action_safe(mdp_state, action):
            self._update_dfa_state(mdp_state, action)
            return action

        # Try all other actions, collect safe ones
        safe_actions = [
            a
            for a in range(len(GridWorld.ACTIONS))
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
        next_mdp_state = self.env.peek_step(mdp_state, action)
        next_label = self.env.compute_label(next_mdp_state)
        self.dfa.next((next_label, action))
