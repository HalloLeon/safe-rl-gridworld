import random
from typing import Callable
from typing import Optional

from common.constants import ACTIONS
from common.types import Action
from common.types import DFAState
from common.types import Label
from common.types import MDPState
from shield_synthesis.automaton.dfa import DFA


class SafetyShield:
    def __init__(
        self,
        dfa: DFA,
        winning_region: set[tuple[MDPState, DFAState]],
        peek_mdp_step: Callable[[MDPState, Action], set[MDPState]],
        compute_mdp_label: Callable[[MDPState], Label],
        rng: Optional[random.Random] = None,
    ):
        self.dfa = dfa
        self.winning_region = winning_region
        self.peek_mdp_step = peek_mdp_step
        self.compute_mdp_label = compute_mdp_label
        self.rng = rng or random.Random()

        # Cache: (dfa_state, mdp_state, action) -> bool
        self._safe_cache: dict[tuple[DFAState, MDPState, Action], bool] = {}

    def filter_action(self, mdp_state: MDPState, action: int) -> int:
        # Filter the agent's proposed action using the winning region.
        #
        # An action is considered safe iff for all possible environment
        # successors (due to potential guard nondeterminism) the product state
        # (next_mdp_state, next_dfa_state) lies in the winning region.

        if self._is_action_safe(mdp_state, action):
            return action

        # Try all other actions, collect safe ones
        safe_actions = [
            a for a in range(len(ACTIONS)) if self._is_action_safe(mdp_state, a)
        ]

        if safe_actions:
            selected_action = self.rng.choice(safe_actions)
            return selected_action

        # No safe action found (should not happen if winning region is correct)
        return action

    def _is_action_safe(self, mdp_state: MDPState, action: int) -> bool:
        # Check if an action is safe from (mdp_state, current_dfa_state)
        # under all possible environment successors.

        cur_state = self.dfa.cur_state
        cache_key = (cur_state, mdp_state, action)

        cached = self._safe_cache.get(cache_key)
        if cached is not None:
            return cached

        successors = self.peek_mdp_step(mdp_state, action)
        if not successors:
            self._safe_cache[cache_key] = (
                False  # No defined successors -> treat as unsafe
            )
            return False

        for next_mdp_state in successors:
            next_label = self.compute_mdp_label(next_mdp_state)
            next_dfa_state = self.dfa.peek_next(
                cur_state, (next_label, action)
            )  # Compute DFA successor from the current DFA state

            if (next_mdp_state, next_dfa_state) not in self.winning_region:
                # At least one possible successor leaves the winning region
                self._safe_cache[cache_key] = False
                return False

        # All possible successors stay inside the winning region
        self._safe_cache[cache_key] = True

        return True

    def update(self, next_mdp_state, action):
        # Must be called by the environment after it has taken the chosen action
        # and observed the actual next MDP state.
        #
        # This keeps the DFA's internal cur_state in sync with the true run.

        next_label = self.compute_mdp_label(next_mdp_state)
        return self.dfa.next((next_label, action))
    
    def reset(self) -> None:
        self.dfa.reset()
