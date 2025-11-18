import random

from dataclasses import dataclass


@dataclass
class GridWorldConfig:
    n_rows: int = 5
    n_cols: int = 5
    start: tuple = (0, 0)
    goal: tuple = (4, 4)
    obstacles: list = None
    traps: list = None
    reward_goal: float = 10.0
    reward_trap: float = -10.0
    reward_step: float = -0.1


class GridWorldGenerator:
    def __new__(cls):
        raise TypeError("GridWorldGenerator is not instantiable.")

    @staticmethod
    def default_config() -> GridWorldConfig:
        return GridWorldConfig()

    @staticmethod
    def random_config(seed: int) -> GridWorldConfig:
        random.seed(seed)

        n_rows = random.randint(3, 10)
        n_cols = random.randint(3, 10)
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
