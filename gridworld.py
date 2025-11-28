import random

from dataclasses import dataclass
from typing import NoReturn
from typing import Optional

from shield import SafetyShield


AgentPos = tuple[int, int]
GuardPos = tuple[int, int]
FacingDirection = int  # 0: up, 1: down, 2: left, 3: right
GuardState = tuple[GuardPos, FacingDirection]
# MDPState: (agent_pos, tuple of (guard_pos, facing) for each guard)
MDPState = tuple[AgentPos, tuple[GuardState, ...]]

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


class Guard:
    VISION_RANGE = 3

    def __init__(
        self, pos: GuardPos, facing_direction: FacingDirection, env: "GridWorld"
    ):
        self.pos = pos
        self.facing_direction = facing_direction
        self.env = env

    def next_step(self):
        dr, dc = ACTIONS[self.facing_direction]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)

        if (
            self.env.in_bounds(next_pos)
            and not self.env.is_wall(next_pos)
            and not self.env.is_agent(next_pos)
        ):
            self.pos = next_pos
            return

        # Try all other directions
        for action in range(len(ACTIONS)):
            adr, adc = ACTIONS[action]
            potential_pos = (self.pos[0] + adr, self.pos[1] + adc)

            if (
                not self.env.in_bounds(potential_pos)
                or self.env.is_wall(potential_pos)
                or self.env.is_agent(potential_pos)
            ):
                continue

            self.pos = potential_pos
            self.facing_direction = action

            return


@dataclass
class GridConfig:
    n_rows: int = 5
    n_cols: int = 5
    start: AgentPos = (0, 0)
    goals: tuple[AgentPos, ...] = ((4, 4),)
    walls: tuple[AgentPos, ...] = ((1, 1), (2, 2), (3, 3))
    guards: tuple[GuardState, ...] = ()
    hazards: tuple[AgentPos, ...] = ()
    reward_goal: float = 10.0
    reward_trap: float = -10.0
    reward_step: float = -0.1
    terminate_on_goal: bool = True
    terminate_on_trap: bool = True


class GridConfigFactory:
    @dataclass
    class GridContext:
        n_rows: int
        n_cols: int
        start: tuple
        goal: tuple
        path: list
        obstacles: list
        hazards: list
        rng: random.Random

    def __new__(cls: type, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("GridConfigFactory is not instantiable.")

    @classmethod
    def default_config(cls: type) -> GridConfig:
        return GridConfig()

    @classmethod
    def random_config(cls: type, n_rows: int, n_cols: int, seed: int = 0) -> GridConfig:
        context = cls.GridContext(
            n_rows=n_rows,
            n_cols=n_cols,
            start=(),
            goal=(),
            path=[],
            obstacles=[],
            hazards=[],
            rng=random.Random(seed),
        )

        cls._select_start_goal(context)
        cls._generate_simple_path(context)
        cls._generate_random_obstacles(context)
        cls._generate_hazards(context)

        return GridConfig(
            n_rows=n_rows,
            n_cols=n_cols,
            start=context.start,
            goal=context.goal,
            walls=context.obstacles,
            hazards=context.hazards,
            reward_goal=10.0,
            reward_trap=-10.0,
            reward_step=-0.1,
        )

    @classmethod
    def _select_start_goal(cls: type, context: GridContext) -> None:
        n_rows = context.n_rows
        n_cols = context.n_cols
        rng = context.rng

        start = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))
        goal = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))

        while goal == start:
            goal = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))

        context.start = start
        context.goal = goal

    @classmethod
    def _generate_simple_path(cls: type, context: GridContext) -> None:
        start = context.start
        goal = context.goal

        path = [start]
        current = start

        while current != goal:
            current = cls._next_step(current, goal, context.rng)
            path.append(current)

        context.path = path

    @staticmethod
    def _next_step(current: tuple, goal: tuple, rng: random.Random) -> tuple:
        row_step = 0
        col_step = 0

        if current[0] < goal[0]:
            row_step = 1
        elif current[0] > goal[0]:
            row_step = -1

        if current[1] < goal[1]:
            col_step = 1
        elif current[1] > goal[1]:
            col_step = -1

        if row_step != 0 and col_step != 0:
            if rng.random() < 0.5:
                return (current[0] + row_step, current[1])
            else:
                return (current[0], current[1] + col_step)
        elif row_step != 0:
            return (current[0] + row_step, current[1])
        elif col_step != 0:
            return (current[0], current[1] + col_step)

        return current

    @classmethod
    def _generate_hazards(cls: type, context: GridContext) -> None:
        n_rows = context.n_rows
        n_cols = context.n_cols
        start = context.start
        goal = context.goal
        path = context.path
        obstacles = context.obstacles
        hazards = context.hazards

        for _ in range(context.rng.randint(0, (n_rows * n_cols) // 10)):
            hazard = (
                context.rng.randint(0, n_rows - 1),
                context.rng.randint(0, n_cols - 1),
            )

            if (
                hazard != start
                and hazard != goal
                and hazard not in obstacles
                and hazard not in path
                and hazard not in hazards
            ):
                hazards.append(hazard)

    @classmethod
    def _generate_random_obstacles(cls: type, context: GridContext) -> None:
        n_rows = context.n_rows
        n_cols = context.n_cols
        start = context.start
        goal = context.goal
        path = context.path
        obstacles = context.obstacles
        hazards = context.hazards

        for _ in range(context.rng.randint(0, (n_rows * n_cols) // 5)):
            obstacle = (
                context.rng.randint(0, n_rows - 1),
                context.rng.randint(0, n_cols - 1),
            )

            if (
                obstacle != start
                and obstacle != goal
                and obstacle not in hazards
                and obstacle not in path
                and obstacle not in obstacles
            ):
                obstacles.append(obstacle)


class GridWorld:
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def __init__(self, config: GridConfig):
        self.config = config
        self.state = config.start
        self.done = False

    def in_bounds(self, state: tuple) -> bool:
        return 0 <= state[0] < self.config.n_rows and 0 <= state[1] < self.config.n_cols

    def is_goal(self, state: tuple) -> bool:
        return state == self.config.goal

    def is_obstacle(self, state: tuple) -> bool:
        return self.config.obstacles and state in self.config.obstacles

    def is_hazard(self, state: tuple) -> bool:
        return self.config.hazards and state in self.config.hazards

    def index_to_state(self, index: int) -> tuple:
        row = index // self.config.n_cols
        col = index % self.config.n_cols

        return (row, col)

    def state_to_index(self, state: tuple) -> int:
        return state[0] * self.config.n_cols + state[1]
    
    def compute_label(self, state: tuple) -> int:
        pass

    def reset(self) -> tuple:
        self.state = self.config.start
        self.done = False

        return self.state_to_index(self.state)

    def next_step(self, action: int) -> tuple:
        if self.done:
            raise RuntimeError("Episode has terminated. Please reset the environment.")

        row, col = self.state
        d_row, d_col = self.ACTIONS[action]

        new_row = row + d_row
        new_column = col + d_col

        new_state = (new_row, new_column)

        if not self.in_bounds(new_state) or self.is_obstacle(new_state):
            new_state = self.state

        reward = self.config.reward_step

        if self.is_goal(new_state):
            reward += self.config.reward_goal

            if self.config.terminate_on_goal:
                self.done = True
        elif self.is_hazard(new_state):
            reward += self.config.reward_trap

            if self.config.terminate_on_trap:
                self.done = True

        self.state = new_state

        return (
            self.state_to_index(new_state),
            reward,
            self.done,
            {"goal": self.is_goal(new_state), "hazard": self.is_hazard(new_state)},
        )
    
    def peek_step(self, action: int) -> tuple:
        pass
