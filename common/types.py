Action = int
Label = int

AgentPos = tuple[int, int]
GuardPos = tuple[int, int]
FacingDirection = int  # 0: up, 1: down, 2: left, 3: right
GuardState = tuple[GuardPos, FacingDirection]

MDPState = tuple[
    AgentPos, tuple[GuardState, ...]
]  # MDPState: (agent_pos, tuple of (guard_pos, facing) for each guard)
DFAState = int
