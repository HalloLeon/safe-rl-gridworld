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
