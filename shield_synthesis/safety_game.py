from collections import deque
from typing import Callable


from common.constants import VERBOSE
from shield_synthesis.automaton.dfa import DFA
from shield_synthesis.automaton.dfa import DFAState


MDPState = tuple
Label = int
Action = int


class SafetyGameSolver:
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
        winning = {state for state in reachable if self.dfa.is_safe_state(state[1])}

        changed = True
        while changed:
            changed = False
            new_winning = set()

            for mdp_state, dfa_state in reachable:
                if not self.dfa.is_safe_state(dfa_state):
                    continue  # cannot be winning

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
        # There exists an action a such that for all environment successors
        # s' in mdp_successors(mdp_state, a), the product (s', q') is winning
        # for q' = dfa.peek_next(dfa_state, (compute_label(s'), a))

        for a in self.actions:
            successors = self.mdp_next(mdp_state, a)

            if not successors:
                continue  # no transitions for this action

            all_good = True

            for next_mdp_state in successors:
                next_label = self._get_label(next_mdp_state)
                next_dfa_state = self.dfa.peek_next(dfa_state, (next_label, a))

                if (next_mdp_state, next_dfa_state) not in winning:
                    all_good = False
                    break

            if all_good:
                return True

        return False

    def _get_label(self, mdp_state: MDPState) -> Label:
        if mdp_state in self._label_cache:
            return self._label_cache[mdp_state]

        lab = self.compute_label(mdp_state)
        self._label_cache[mdp_state] = lab

        return lab
