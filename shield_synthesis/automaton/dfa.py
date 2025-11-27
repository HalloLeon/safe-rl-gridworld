from shield_synthesis.safety_game import Action
from shield_synthesis.safety_game import DFA
from shield_synthesis.safety_game import Label


DFA_State = int


class DFA:
    def __init__(
        self,
        states: list[DFA_State],
        initial: DFA_State,
        safe_states: list[DFA_State],
        alphabet: list[tuple[Label, Action]],
    ):
        self.states = states
        self.initial = initial
        self.cur_state = initial
        self.safe_states = set(safe_states)
        self.alphabet = alphabet
        self.transitions: dict[DFA_State, dict[tuple[Label, Action], DFA_State]] = {
            q: {} for q in states
        }

    def add_transition(
        self, src: DFA_State, letter: tuple[Label, Action], dst: DFA_State
    ):
        assert src in self.states
        assert dst in self.states
        assert letter in self.alphabet

        self.transitions[src][letter] = dst

    def reset(self) -> None:
        self.cur_state = self.initial

    def next(self, letter: tuple[Label, Action]) -> DFA_State:
        self.cur_state = self.transitions[self.cur_state][letter]
        return self.cur_state

    def peek_next(self, state: DFA_State, letter: tuple[Label, Action]) -> DFA_State:
        return self.transitions[state][letter]

    def is_safe_state(self, state: DFA_State) -> bool:
        return state in self.safe_states


def make_simple_dfa(
    safe_labels: list[Label],
    unsafe_labels: list[Label],
    actions: list[Action],
) -> DFA:
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
