from collections import deque
from typing import Callable


from common.constants import VERBOSE
from common.types import Action
from common.types import Label
from common.types import MDPState
from shield_synthesis.automaton.dfa import DFA
from shield_synthesis.automaton.dfa import DFAState


class SafetyGameSolver:
    """
    Solves a safety game on the product of an MDP and a DFA.

    The winning region is the set of product states (s, q) from which the
    agent can enforce that the DFA remains in a safe state,
    regardless of the environment's moves.

    Args:
        dfa:
            Deterministic finite automaton encoding the safety specification.
        actions:
            List of all available agent actions in the MDP. These are the
            actions over which the safety game quantifies.
        mdp_next:
            Symbolic transition function of the MDP.
            Must enumerate *all* possible successor MDP states for the
            given (state, action), including environment nondeterminism.
        compute_label:
            Labelling function used as input to the DFA transitions.
    """

    def __init__(
        self,
        dfa: DFA,
        actions: list[Action],
        mdp_next: Callable[[MDPState, Action], set[MDPState]],
        compute_label: Callable[[MDPState], Label],
    ):
        self.dfa = dfa
        self.actions = actions
        self.mdp_next = mdp_next
        self.compute_label = compute_label

        # Cache for performance
        self._label_cache: dict[MDPState, Label] = {}

    def compute_winning_region(
        self, initial_mdp_state: MDPState
    ) -> set[tuple[MDPState, DFAState]]:
        """
        Compute the winning region of the safety game, starting from the
        product initial state (initial_mdp_state, dfa.initial).

        Returns:
            A set of product states (mdp_state, dfa_state) that are winning.
        """

        reachable = self._compute_reachable_states(initial_mdp_state)

        if VERBOSE:
            print(f"Reachable states: {len(reachable)}")

        winning = self._compute_winning_states(reachable)

        if VERBOSE:
            print(f"Winning states: {len(winning)}")

        return winning

    def _compute_reachable_states(
        self, initial_mdp_state: MDPState
    ) -> set[tuple[MDPState, DFAState]]:
        """
        Forward BFS in the product MDP × DFA.

        Starts from (initial_mdp_state, dfa.initial) and explores all
        states reachable under all actions and all nondeterministic MDP
        successors.
        """

        initial = (initial_mdp_state, self.dfa.initial)
        reachable = {initial}

        queue = deque([initial])

        while queue:
            mdp_state, dfa_state = queue.popleft()

            for a in self.actions:
                successors = self.mdp_next(mdp_state, a)

                for next_mdp_state in successors:
                    next_label = self._get_label(next_mdp_state)
                    next_dfa_state = self.dfa.peek_next(dfa_state, (next_label, a))
                    next_state = (next_mdp_state, next_dfa_state)

                    if next_state not in reachable:
                        reachable.add(next_state)
                        queue.append(next_state)

        return reachable

    def _compute_winning_states(
        self, reachable: set[tuple[MDPState, DFAState]]
    ) -> set[tuple[MDPState, DFAState]]:
        """
        Standard fixpoint computation for safety games.

        Start with all reachable states whose DFA component is safe,
        then iteratively remove those from which the controller cannot
        enforce safety under all environment moves.
        """

        winning = {state for state in reachable if self.dfa.is_safe_state(state[1])}

        changed = True
        while changed:
            changed = False
            new_winning = set()

            for mdp_state, dfa_state in reachable:
                if not self.dfa.is_safe_state(dfa_state):
                    continue  # Cannot be winning

                if self._has_safe_action(mdp_state, dfa_state, winning):
                    new_winning.add((mdp_state, dfa_state))

            if new_winning != winning:
                winning = new_winning
                changed = True

        return winning

    def _has_safe_action(
        self,
        mdp_state: MDPState,
        dfa_state: DFAState,
        winning: set[tuple[MDPState, DFAState]],
    ) -> bool:
        """
        Check if there exists at least one controller action 'a' such that
        for *all* environment successors s' ∈ mdp_next(mdp_state, a),
        the product state (s', q') is winning, where
        q' = dfa.peek_next(dfa_state, (label(s'), a)).
        """

        for a in self.actions:
            successors = self.mdp_next(mdp_state, a)

            if not successors:
                continue  # No transitions for this action

            all_good = True

            for next_mdp_state in successors:
                next_label = self._get_label(next_mdp_state)
                next_dfa_state = self.dfa.peek_next(dfa_state, (next_label, a))

                if (next_mdp_state, next_dfa_state) not in winning:
                    all_good = False
                    break

            if all_good:
                # Found an action that keeps the agent in the winning region
                return True

        # No action satisfies the "for all successors" condition
        return False

    def _get_label(self, mdp_state: MDPState) -> Label:
        """
        Cached lookup of compute_label(mdp_state).
        """

        if mdp_state in self._label_cache:
            return self._label_cache[mdp_state]

        lab = self.compute_label(mdp_state)
        self._label_cache[mdp_state] = lab

        return lab
