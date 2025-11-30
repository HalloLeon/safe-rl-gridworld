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
    """
    Safety shield for an MDP + DFA product game.

    The shield filters proposed actions so that the actual run stays within
    the winning region as long as the environment behaves according to
    `peek_mdp_step`.

    Args:
        dfa:
            Deterministic finite automaton encoding the safety specification
            over (label, action) pairs.
        winning_region:
            Set of product states `(mdp_state, dfa_state)` from which the
            agent can enforce safety in the safety game.
        peek_mdp_step:
            Function implementing the symbolic MDP transition,
            giving *all* possible successor MDP states for a given
            (mdp_state, action) pair (capturing environment nondeterminism),
        compute_mdp_label:
            Labelling function, mapping an MDP state to the label used
            by the DFA.
        rng:
            Optional random number generator used to break ties between
            multiple safe actions. If `None`, a fresh `random.Random`
            instance is created.
    """

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
        """
        Given the current MDP state and a proposed action, return a safe action.

        An action is considered safe iff for all possible environment successors
        the product state (next_mdp_state, next_dfa_state) lies in the winning region.

        Logic:
          1. If the proposed action is safe, keep it.
          2. Otherwise, search over all actions and pick a safe one at random.
          3. If no safe action exists (should not happen if the winning region
             is correct and the agent is in it), fall back to the original action.

        Args:
            mdp_state: Current MDP state.
            action: Proposed action.

        Returns:
            The action to execute (possibly different from the proposed one).
        """

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
        """
        Check if an action is safe from the current DFA state and given MDP state.

        An action is safe iff for *all* nondeterministic environment successors
        s' ∈ peek_mdp_step(mdp_state, action), the product (s', q') is in the
        winning region, where q' = dfa.peek_next(cur_dfa_state, (label(s'), action)).
        """

        cur_state = self.dfa.cur_state
        cache_key = (cur_state, mdp_state, action)

        cached = self._safe_cache.get(cache_key)
        if cached is not None:
            return cached

        successors = self.peek_mdp_step(mdp_state, action)
        if not successors:
            # No defined successors -> treat as unsafe
            self._safe_cache[cache_key] = False
            return False

        for next_mdp_state in successors:
            next_label = self.compute_mdp_label(next_mdp_state)
            next_dfa_state = self.dfa.peek_next(
                cur_state, (next_label, action)
            )  # DFA successor from the current DFA state

            if (next_mdp_state, next_dfa_state) not in self.winning_region:
                # At least one possible successor leaves the winning region
                self._safe_cache[cache_key] = False
                return False

        # All possible successors stay inside the winning region
        self._safe_cache[cache_key] = True

        return True

    def update(self, next_mdp_state: MDPState, action: Action) -> DFAState:
        """
        Update the internal DFA state after the environment has actually
        executed an action and transitioned to a new MDP state.

        This *must* be called by the environment after each real step in order
        for the shield's internal DFA state to stay in sync with the actual run.
        """

        next_label = self.compute_mdp_label(next_mdp_state)
        return self.dfa.next((next_label, action))

    def reset(self) -> None:
        """
        Reset the internal DFA state.

        This *must* be called whenever the environment is reset to an initial MDP
        state consistent with the DFA's initial state.
        """

        self.dfa.reset()
