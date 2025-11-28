from dataclasses import dataclass
import random
from typing import NoReturn
from typing import Optional

from shield import SafetyShield


AgentPos = tuple[int, int]
GuardPos = tuple[int, int]
FacingDirection = int  # 0: up, 1: down, 2: left, 3: right
GuardState = tuple[GuardPos, FacingDirection]
MDPState = tuple[
    AgentPos, tuple[GuardState, ...]
]  # MDPState: (agent_pos, tuple of (guard_pos, facing) for each guard)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


@dataclass
class GridConfig:
    n_rows: int = 5
    n_cols: int = 5
    start: AgentPos = (0, 0)
    goals: tuple[AgentPos, ...] = ((4, 4),)
    walls: tuple[AgentPos, ...] = ((1, 1), (2, 2), (3, 3))
    guards: tuple[GuardState, ...] = ()
    reward_goal: float = 10.0
    penalty_if_caught: float = -10.0
    penalty_step: float = -0.1
    terminate_on_completion: bool = True
    terminate_if_caught: bool = True


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
                facing = rng.randint(0, len(ACTIONS) - 1)
                guards.append((guard_pos, facing))


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


class GridWorld:
    ACTIONS = ACTIONS  # up, down, left, right

    def __init__(
        self,
        config: GridConfig,
        shield: Optional[SafetyShield] = None,
        rng: Optional[random.Random] = None,
    ):
        self.config = config
        self.shield = shield
        self.rng = rng if rng is not None else random.Random()

        # Initialize guards as objects
        self.guards = self._init_guards()

        # Current agent position
        self.agent_pos = self.config.start

        # Track remaining goals as a list
        self.goals = list(self.config.goals)

        # Build the initial MDP state
        self.cur_state = self._build_mdp_state(self.agent_pos)
        self.initial_state = self.cur_state

        self.done: bool = False

    def _init_guards(self) -> list[Guard]:
        guards = []

        for guard_pos, facing_direction in self.config.guards:
            guards.append(Guard(guard_pos, facing_direction, self))

        return guards

    def _build_mdp_state(self, agent_pos: AgentPos) -> MDPState:
        guard_states = tuple((g.pos, g.facing_direction) for g in self.guards)
        return (agent_pos, guard_states)

    def in_bounds(self, pos: AgentPos) -> bool:
        return 0 <= pos[0] < self.config.n_rows and 0 <= pos[1] < self.config.n_cols

    def is_wall(self, pos: AgentPos) -> bool:
        return pos in self.config.walls

    def is_goal_pos(self, pos: AgentPos) -> bool:
        return pos in self.config.goals

    def is_agent(self, pos: AgentPos) -> bool:
        return self.agent_pos == pos

    def is_caught(self, mdp_state: MDPState) -> bool:
        agent_pos, guard_states = mdp_state

        for g_pos, facing in guard_states:
            if g_pos == agent_pos:
                return True

            if self._is_visible_from_guard(agent_pos, g_pos, facing):
                return True

        return False

    def _is_visible_from_guard(
        self, agent_pos: AgentPos, guard_pos: GuardPos, facing: FacingDirection
    ) -> bool:
        # Simple FOV: straight ray in facing direction up to VISION_RANGE,
        # blocked by walls.

        dr, dc = ACTIONS[facing]
        r, c = guard_pos

        for step in range(1, Guard.VISION_RANGE + 1):
            cell = (r + step * dr, c + step * dc)

            if not self.in_bounds(cell) or self.is_wall(cell):
                break

            if cell == agent_pos:
                return True

        return False

    def mdp_state_to_index(self, mdp_state: MDPState) -> int:
        row, col = mdp_state[0]
        return row * self.config.n_cols + col

    def index_to_mdp_state(self, index: int) -> MDPState:
        row = index // self.config.n_cols
        col = index % self.config.n_cols

        # Use current guard object states
        return self._build_mdp_state((row, col))

    def compute_label(self, mdp_state: MDPState) -> int:
        # Label encoding for DFA / shield:

        #    0 = SAFE
        #    1 = WALL      (on a wall cell)
        #    2 = VISIBLE   (seen by guard or sharing cell with guard)

        agent_pos, _ = mdp_state

        if self.is_wall(agent_pos):
            return 1
        if self.is_caught(mdp_state):
            return 2

        return 0

    def reset(self) -> int:
        # Reset agent and guards first, then rebuild state

        self.agent_pos = self.config.start
        self.guards = self._init_guards()
        self.goals = list(self.config.goals)
        self.done = False

        self.cur_state = self._build_mdp_state(self.agent_pos)
        self.initial_state = self.cur_state

        return self.mdp_state_to_index(self.cur_state)

    def next_step(self, action: int):
        # Standard step function:

        # - optional shield to filter the proposed action
        # - update agent position
        # - move guards
        # - compute reward / termination

        if self.done:
            raise RuntimeError("Episode has terminated. Please reset the environment.")

        # 1. Apply shield to current MDP state (if possible) to filter action
        #    or select action after simple safety checks
        mdp_before = self.cur_state

        if self.shield is not None:
            chosen_action = self.shield.filter_action(mdp_before, action)
        else:
            chosen_action = self._select_action(action)

        # 2. Move agent
        new_agent_pos = self._next_agent_pos(self.agent_pos, chosen_action)

        # Temporarily store new agent position so guards avoid it
        self.agent_pos = new_agent_pos

        # 3. Move guards
        for guard in self.guards:
            guard.next_step()

        # 4. Rebuild current MDP state
        self.cur_state = self._build_mdp_state(self.agent_pos)

        # 5. Compute reward and termination
        reward = self.config.penalty_step
        info = {
            "goal": False,
            "caught": False,
        }

        # Goal
        if self.agent_pos in self.goals:
            reward += self.config.reward_goal
            info["goal"] = True
            self.goals.remove(self.agent_pos)  # Remove this goal

            if self.config.terminate_on_completion and len(self.goals) == 0:
                self.done = True

        # Caught by guard
        if self.is_caught(self.cur_state):
            reward += self.config.penalty_if_caught
            info["caught"] = True

            if self.config.terminate_if_caught:
                self.done = True

        return self.mdp_state_to_index(self.cur_state), reward, self.done, info

    def _select_action(self, action: int) -> int:
        # Simple fallback action selection when no shield is provided:
        # - if proposed action leads to a valid cell, keep it;
        # - otherwise try to find some valid action;
        # - if none exists, return the original action.

        dr, dc = ACTIONS[action]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        if self.in_bounds(new_pos) and not self.is_wall(new_pos):
            return action

        # Try to find an alternative safe action
        for a in range(len(ACTIONS)):
            adr, adc = ACTIONS[a]
            potential_pos = (self.agent_pos[0] + adr, self.agent_pos[1] + adc)

            if self.in_bounds(potential_pos) and not self.is_wall(potential_pos):
                return a

        # No safe alternative found -> Fall back to original
        return action

    def _next_agent_pos(self, agent_pos: AgentPos, action: int) -> AgentPos:
        dr, dc = ACTIONS[action]
        new_pos = (agent_pos[0] + dr, agent_pos[1] + dc)

        if not self.in_bounds(new_pos) or self.is_wall(new_pos):
            return agent_pos  # Bump into wall / out-of-bounds -> stay

        return new_pos

    def peek_step(self, mdp_state: MDPState, action: int) -> MDPState:
        # Pure simulation: given an arbitrary MDP state and action,
        # return the next MDP state (agent + guards) without side effects.

        agent_pos, guard_states = mdp_state

        # Simulate agent
        new_agent_pos = self._next_agent_pos(agent_pos, action)

        # Simulate guards based on new agent position
        new_guard_states = []

        for g_pos, facing in guard_states:
            ng_pos, ng_facing = self._simulate_guard_step(g_pos, facing, new_agent_pos)
            new_guard_states.append((ng_pos, ng_facing))

        return (new_agent_pos, tuple(new_guard_states))

    def _simulate_guard_step(
        self, guard_pos: GuardPos, facing: FacingDirection, agent_pos: AgentPos
    ) -> GuardState:
        dr, dc = ACTIONS[facing]
        next_pos = (guard_pos[0] + dr, guard_pos[1] + dc)
        if (
            self.in_bounds(next_pos)
            and not self.is_wall(next_pos)
            and next_pos != agent_pos
        ):
            return next_pos, facing

        # Try all other directions
        for a in range(len(ACTIONS)):
            adr, adc = ACTIONS[a]
            potential_pos = (guard_pos[0] + adr, guard_pos[1] + adc)

            if (
                not self.in_bounds(potential_pos)
                or self.is_wall(potential_pos)
                or potential_pos == agent_pos
            ):
                continue

            return potential_pos, a

        # No move possible
        return guard_pos, facing
