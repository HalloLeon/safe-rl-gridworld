State = int
Label = int
Action = int


class DFA:
    def __init__(
        self,
        states: list[State],
        initial: State,
        safe_states: list[State],
        alphabet: list[tuple[Label, Action]],
    ):
        self.states = states
        self.initial = initial
        self.cur_state = initial
        self.safe_states = set(safe_states)
        self.alphabet = alphabet
        self.transitions: dict[State, dict[tuple[Label, Action], State]] = {
            q: {} for q in states
        }

    def add_transition(self, src: State, letter: tuple[Label, Action], dst: State):
        assert src in self.states
        assert dst in self.states
        assert letter in self.alphabet

        self.transitions[src][letter] = dst

    def reset(self) -> None:
        self.cur_state = self.initial

    def next_state(self, letter: tuple[Label, Action]) -> None:
        self.cur_state = self.transitions[self.cur_state][letter]

    def is_safe_state(self, state: State) -> bool:
        return state in self.safe_states


def make_default_dfa() -> DFA:
    # States
    SAFE_STATE = 0
    SINK_STATE = 1

    # Labels
    SAFE = 0
    VISIBLE = 1
    CRASH = 2

    # Actions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    default_dfa = DFA(
        states=[SAFE_STATE, SINK_STATE],
        initial=SAFE_STATE,
        safe_states=[SAFE_STATE],
        alphabet=[
            (SAFE, UP),
            (SAFE, RIGHT),
            (SAFE, DOWN),
            (SAFE, LEFT),
            (VISIBLE, UP),
            (VISIBLE, RIGHT),
            (VISIBLE, DOWN),
            (VISIBLE, LEFT),
            (CRASH, UP),
            (CRASH, RIGHT),
            (CRASH, DOWN),
            (CRASH, LEFT),
        ],
    )

    default_dfa.add_transition(SAFE_STATE, (SAFE, UP), SAFE_STATE)
    default_dfa.add_transition(SAFE_STATE, (SAFE, RIGHT), SAFE_STATE)
    default_dfa.add_transition(SAFE_STATE, (SAFE, DOWN), SAFE_STATE)
    default_dfa.add_transition(SAFE_STATE, (SAFE, LEFT), SAFE_STATE)
    default_dfa.add_transition(SAFE_STATE, (VISIBLE, UP), SINK_STATE)
    default_dfa.add_transition(SAFE_STATE, (VISIBLE, RIGHT), SINK_STATE)
    default_dfa.add_transition(SAFE_STATE, (VISIBLE, DOWN), SINK_STATE)
    default_dfa.add_transition(SAFE_STATE, (VISIBLE, LEFT), SINK_STATE)
    default_dfa.add_transition(SAFE_STATE, (CRASH, UP), SINK_STATE)
    default_dfa.add_transition(SAFE_STATE, (CRASH, RIGHT), SINK_STATE)
    default_dfa.add_transition(SAFE_STATE, (CRASH, DOWN), SINK_STATE)
    default_dfa.add_transition(SAFE_STATE, (CRASH, LEFT), SINK_STATE)

    for action in [UP, RIGHT, DOWN, LEFT]:
        for label in [SAFE, VISIBLE, CRASH]:
            default_dfa.add_transition(SINK_STATE, (label, action), SINK_STATE)

    return default_dfa
