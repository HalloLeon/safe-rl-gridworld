from collections import deque
from typing import Callable

from shield_synthesis.automaton.dfa import DFA
from shield_synthesis.automaton.dfa import DFA_State


MDP_State = tuple
Label = int
Action = int


class SafetyGameSolver:
    def __init__(
        self,
        dfa: DFA,
        actions: list[Action],
        mdp_next: Callable[[MDP_State, Action], MDP_State],
        compute_label: Callable[[MDP_State], Label],
    ):
        self.dfa = dfa
        self.actions = actions
        self.mdp_next = mdp_next
        self.compute_label = compute_label

    def compute_winning_region(
        self, initial_mdp_state: MDP_State
    ) -> set[tuple[MDP_State, DFA_State]]:
        reachable = self._compute_reachable_states(initial_mdp_state)
        winning = self._compute_winning_states(reachable)

        return winning

    def _compute_reachable_states(
        self, initial_mdp_state: MDP_State
    ) -> set[tuple[MDP_State, DFA_State]]:
        initial = (initial_mdp_state, self.dfa.initial)
        reachable = {initial}

        queue = deque([initial])

        while queue:
            mdp_state, dfa_state = queue.popleft()

            for a in self.actions:
                next_mdp_state = self.mdp_next(mdp_state, a)
                next_label = self.compute_label(next_mdp_state)
                next_dfa_state = self.dfa.peek_next(dfa_state, (next_label, a))
                next_state = (next_mdp_state, next_dfa_state)

                if next_state not in reachable:
                    reachable.add(next_state)
                    queue.append(next_state)

        return reachable

    def _compute_winning_states(
        self, reachable: set[tuple[MDP_State, DFA_State]]
    ) -> set[tuple[MDP_State, DFA_State]]:
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
        mdp_state: MDP_State,
        dfa_state: DFA_State,
        winning: set[tuple[MDP_State, DFA_State]],
    ) -> bool:
        for a in self.actions:
            next_mdp_state = self.mdp_next(mdp_state, a)
            next_label = self.compute_label(next_mdp_state)
            next_dfa_state = self.dfa.peek_next(dfa_state, (next_label, a))

            if (next_mdp_state, next_dfa_state) in winning:
                return True

        return False
