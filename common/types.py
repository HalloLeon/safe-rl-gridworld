Action = int
Label = int

Pos = tuple[int, int]  # (row, col)
FacingDirection = int  # 0: up, 1: down, 2: left, 3: right
GuardState = tuple[Pos, FacingDirection]

MDPState = tuple[
    Pos, tuple[GuardState, ...]
]  # MDPState: (agent_pos, tuple of (guard_pos, facing) for each guard)
DFAState = int
