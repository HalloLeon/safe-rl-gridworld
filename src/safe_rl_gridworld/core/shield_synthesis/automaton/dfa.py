from safe_rl_gridworld.common.types import Action
from safe_rl_gridworld.common.types import DFAState
from safe_rl_gridworld.common.types import Label


class DFA:
    """
    Deterministic finite automaton (DFA) over an alphabet of (label, action) pairs.

    Args:
        states: Finite set of DFA states.
        initial: Initial DFA state.
        safe_states: Subset of states considered "safe" for safety games.
        alphabet: All allowed input symbols (label, action).
    """

    def __init__(
        self,
        states: list[DFAState],
        initial: DFAState,
        safe_states: list[DFAState],
        alphabet: list[tuple[Label, Action]],
    ):
        self.states = states
        self.initial = initial
        self.cur_state = initial
        self.safe_states = set(safe_states)
        self.alphabet = alphabet

        # transitions[q][letter] = q'
        self.transitions: dict[DFAState, dict[tuple[Label, Action], DFAState]] = {
            q: {} for q in states
        }

    def add_transition(
        self, src: DFAState, letter: tuple[Label, Action], dst: DFAState
    ):
        """
        Add a transition to the DFA.

        Args:
            src: Source DFA state.
            letter: Input symbol `(label, action)`.
            dst: Destination DFA state.

        Raises:
            AssertionError: If `src` or `dst` is not a valid state, or
                `letter` is not in the DFA's alphabet.
        """

        assert src in self.states
        assert dst in self.states
        assert letter in self.alphabet

        self.transitions[src][letter] = dst

    def reset(self) -> None:
        """Reset the DFA to its initial state."""
        self.cur_state = self.initial

    def next(self, letter: tuple[Label, Action]) -> DFAState:
        """
        Consume an input symbol and update the current state.

        Args:
            letter: Input symbol `(label, action)` to process.

        Returns:
            The new current DFA state after applying the transition.

        Raises:
            KeyError: If no transition is defined for `(cur_state, letter)`.
        """

        self.cur_state = self.transitions[self.cur_state][letter]
        return self.cur_state

    def peek_next(self, state: DFAState, letter: tuple[Label, Action]) -> DFAState:
        """
        Return the successor state from a given state without mutating the DFA.

        Args:
            state: The DFA state from which to apply the transition.
            letter: Input symbol `(label, action)` to process.

        Returns:
            The successor DFA state.

        Raises:
            KeyError: If no transition is defined for `(state, letter)`.
        """

        return self.transitions[state][letter]

    def is_safe_state(self, state: DFAState) -> bool:
        """
        Check whether a DFA state is designated as safe.

        Args:
            state: DFA state to be checked.

        Returns:
            True if `state` is in `safe_states`, False otherwise.
        """

        return state in self.safe_states


def build_simple_dfa(
    safe_labels: list[Label],
    unsafe_labels: list[Label],
    actions: list[Action],
) -> DFA:
    """
    Build a 2-state DFA capturing a simple safety property.

    Implementation details:
        * The DFA has two states:
            - SAFE_STATE (0): the system is safe so far.
            - SINK_STATE (1): an unsafe label has been seen (irrecoverable).
        * Alphabet symbols are `(label, action)` pairs.
        * From SAFE_STATE:
            - If `label` is in `safe_labels`, stay in SAFE_STATE.
            - If `label` is in `unsafe_labels`, transition to SINK_STATE.
        * From SINK_STATE:
            - Stay in SINK_STATE for all `(label, action)`.

    Args:
        safe_labels: List of labels that are considered safe.
        unsafe_labels: List of labels that are considered unsafe.
        actions: List of possible actions.

    Returns:
        A DFA object encoding the described safety property.

    Note:
        The caller is responsible for ensuring that `safe_labels` and
        `unsafe_labels` cover all labels that may appear in the MDP.
    """

    # States
    SAFE_STATE = 0
    SINK_STATE = 1

    # Alphabet: all (label, action) pairs
    alphabet: list[tuple[Label, Action]] = [
        (label, action)
        for label in (*safe_labels, *unsafe_labels)
        for action in actions
    ]

    dfa = DFA(
        states=[SAFE_STATE, SINK_STATE],
        initial=SAFE_STATE,
        safe_states=[SAFE_STATE],
        alphabet=alphabet,
    )

    # From SAFE_STATE:
    #  - safe labels -> stay in SAFE_STATE
    #  - unsafe labels -> go to SINK_STATE
    for label in safe_labels:
        for action in actions:
            dfa.add_transition(SAFE_STATE, (label, action), SAFE_STATE)

    for label in unsafe_labels:
        for action in actions:
            dfa.add_transition(SAFE_STATE, (label, action), SINK_STATE)

    # From SINK_STATE: stay in SINK_STATE for all labels/actions
    for label in (*safe_labels, *unsafe_labels):
        for action in actions:
            dfa.add_transition(SINK_STATE, (label, action), SINK_STATE)

    return dfa
