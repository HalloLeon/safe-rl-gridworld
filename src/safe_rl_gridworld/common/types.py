Action = int  # index into ACTIONS
Label = int

Pos = tuple[int, int]  # (row, col)
FacingDirection = int  # index into DIRECTIONS
GuardState = tuple[Pos, FacingDirection]

MDPState = tuple[
    Pos, tuple[GuardState, ...]
]  # MDPState: (agent_pos, tuple of (guard_pos, facing_direction) for each guard)
DFAState = int
