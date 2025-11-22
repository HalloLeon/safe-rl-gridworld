import random

from dataclasses import dataclass
from typing import NoReturn


@dataclass
class GridWorldConfig:
    n_rows: int = 5
    n_cols: int = 5
    start: tuple = (0, 0)
    goal: tuple = (4, 4)
    obstacles: tuple = ((1, 1), (2, 2), (3, 3))
    traps: tuple = ((1, 3), (3, 1))
    reward_goal: float = 10.0
    reward_trap: float = -10.0
    reward_step: float = -0.1
    terminate_on_goal: bool = True
    terminate_on_trap: bool = True


class GridWorldGenerator:
    def __new__(cls: type, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("GridWorldGenerator is not instantiable.")

    @staticmethod
    def default_config() -> GridWorldConfig:
        return GridWorldConfig()

    @staticmethod
    def random_config(seed: int) -> GridWorldConfig:
        random.seed(seed)

        n_rows = random.randint(5, 10)
        n_cols = random.randint(5, 10)
        start = (random.randint(0, n_rows - 1), random.randint(0, n_cols - 1))
        goal = (random.randint(0, n_rows - 1), random.randint(0, n_cols - 1))

        while goal == start:
            goal = (random.randint(0, n_rows - 1), random.randint(0, n_cols - 1))

        obstacles = []

        for _ in range(random.randint(0, (n_rows * n_cols) // 5)):
            obstacle = (random.randint(0, n_rows - 1), random.randint(0, n_cols - 1))

            if obstacle != start and obstacle != goal:
                obstacles.append(obstacle)

        traps = []

        for _ in range(random.randint(0, (n_rows * n_cols) // 10)):
            trap = (random.randint(0, n_rows - 1), random.randint(0, n_cols - 1))

            if trap != start and trap != goal and trap not in obstacles:
                traps.append(trap)

        return GridWorldConfig(
            n_rows=n_rows,
            n_cols=n_cols,
            start=start,
            goal=goal,
            obstacles=obstacles,
            traps=traps,
            reward_goal=10.0,
            reward_trap=-10.0,
            reward_step=-0.1,
        )


class GridWorld:
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def __init__(self, config: GridWorldConfig):
        self.config = config
        self.state = config.start
        self.done = False

    def in_bounds(self, state: tuple) -> bool:
        return 0 <= state[0] < self.config.n_rows and 0 <= state[1] < self.config.n_cols

    def is_goal(self, state: tuple) -> bool:
        return state == self.config.goal

    def is_obstacle(self, state: tuple) -> bool:
        return self.config.obstacles and state in self.config.obstacles

    def is_trap(self, state: tuple) -> bool:
        return self.config.traps and state in self.config.traps

    def reset(self) -> tuple:
        self.state = self.config.start
        self.done = False

        return self.state

    def step(self, action: int) -> tuple:
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
        elif self.is_trap(new_state):
            reward += self.config.reward_trap

            if self.config.terminate_on_trap:
                self.done = True

        self.state = new_state

        return new_state, reward, self.done
