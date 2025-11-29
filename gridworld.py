from dataclasses import dataclass
import random
from typing import NoReturn
from typing import Optional

from common.constants import ACTIONS
from common.constants import DIRECTIONS
from common.types import Pos
from common.types import FacingDirection
from common.types import GuardState
from common.types import MDPState
from shield import SafetyShield


@dataclass
class GridConfig:
    n_rows: int = 5
    n_cols: int = 5
    start: Pos = (0, 0)
    goals: tuple[Pos, ...] = ((4, 4),)
    walls: tuple[Pos, ...] = ((1, 1), (2, 2), (3, 3))
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
        n_guards: int
        start: Pos
        goal: Pos
        path: list[Pos]
        walls: list[Pos]
        guards: list[GuardState]
        rng: random.Random

    def __new__(
        cls: type["GridConfigFactory"], *args: object, **kwargs: object
    ) -> NoReturn:
        raise TypeError("GridConfigFactory is not instantiable.")

    @classmethod
    def default_config(cls: type["GridConfigFactory"]) -> GridConfig:
        return GridConfig()

    @classmethod
    def random_config(
        cls: type["GridConfigFactory"],
        n_rows: int,
        n_cols: int,
        n_guards: int = 1,
        seed: int = 0,
    ) -> GridConfig:
        context = cls.GridContext(
            n_rows=n_rows,
            n_cols=n_cols,
            n_guards=n_guards,
            start=(0, 0),
            goal=(0, 0),
            path=[],
            walls=[],
            guards=[],
            rng=random.Random(seed),
        )

        cls._select_start_goal(context)
        cls._generate_simple_path(context)
        cls._generate_random_walls(context)
        cls._generate_random_guards(context)

        return GridConfig(
            n_rows=n_rows,
            n_cols=n_cols,
            start=context.start,
            goals=(context.goal,),
            walls=tuple(context.walls),
            guards=tuple(context.guards),
            reward_goal=10.0,
            penalty_if_caught=-10.0,
            penalty_step=-0.1,
            terminate_on_completion=True,
            terminate_if_caught=True,
        )

    @staticmethod
    def _select_start_goal(context: GridContext) -> None:
        n_rows = context.n_rows
        n_cols = context.n_cols
        rng = context.rng

        start = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))
        goal = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))

        while goal == start:
            goal = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))

        context.start = start
        context.goal = goal

    @staticmethod
    def _generate_simple_path(context: GridContext) -> None:
        start = context.start
        goal = context.goal

        path = [start]
        current = start

        while current != goal:
            current = GridConfigFactory._next_step(current, goal, context.rng)
            path.append(current)

        context.path = path

    @staticmethod
    def _next_step(current: Pos, goal: Pos, rng: random.Random) -> Pos:
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
            # Randomly choose whether to move row-wise or column-wise
            if rng.random() < 0.5:
                return (current[0] + row_step, current[1])
            else:
                return (current[0], current[1] + col_step)
        elif row_step != 0:
            return (current[0] + row_step, current[1])
        elif col_step != 0:
            return (current[0], current[1] + col_step)

        return current

    @staticmethod
    def _generate_random_walls(context: GridContext) -> None:
        n_rows = context.n_rows
        n_cols = context.n_cols
        start = context.start
        goal = context.goal
        path = context.path
        walls = context.walls
        rng = context.rng

        # Up to 20% of the grid as walls
        max_walls = (n_rows * n_cols) // 5
        n_walls = rng.randint(0, max_walls)

        for _ in range(n_walls):
            wall = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))

            if (
                wall != start
                and wall != goal
                and wall not in path
                and wall not in walls
            ):
                walls.append(wall)

    @staticmethod
    def _generate_random_guards(context: GridContext) -> None:
        n_rows = context.n_rows
        n_cols = context.n_cols
        start = context.start
        goal = context.goal
        path = context.path
        walls = context.walls
        guards = context.guards
        rng = context.rng

        n_guards = context.n_guards

        for _ in range(n_guards):
            guard_pos = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))

            if (
                guard_pos != start
                and guard_pos != goal
                and guard_pos not in path
                and guard_pos not in walls
                and guard_pos not in [g[0] for g in guards]
            ):
                facing_direction = rng.randint(0, len(DIRECTIONS) - 1)
                guards.append((guard_pos, facing_direction))


class Guard:
    VISION_RANGE = 3

    def __init__(self, env: "GridWorld", pos: Pos, facing_direction: FacingDirection):
        self.env = env
        self.pos = pos
        self.facing_direction = facing_direction

    @staticmethod
    def peek_step(
        env: "GridWorld",
        cur_pos: Pos,
        facing: FacingDirection,
        agent_pos: AgentPos,
        other_guards: set[GuardPos],
    ) -> GuardState:
        # Pure guard movement logic that avoids:
        #   - walls
        #   - the agent
        #   - other guards' positions

        # Try to move forward in current facing direction
        dr, dc = ACTIONS[facing]
        next_pos = (cur_pos[0] + dr, cur_pos[1] + dc)

        if (
            env.in_bounds(next_pos)
            and not env.is_wall(next_pos)
            and next_pos != agent_pos
            and next_pos not in other_guards
        ):
            return next_pos, facing

        # Try all other actions
        for action in range(len(ACTIONS)):
            adr, adc = ACTIONS[action]
            potential_pos = (cur_pos[0] + adr, cur_pos[1] + adc)

            if (
                not env.in_bounds(potential_pos)
                or env.is_wall(potential_pos)
                or potential_pos == agent_pos
                or potential_pos in other_guards
            ):
                continue

            # If this is a real direction (UP/DOWN/LEFT/RIGHT), update facing
            if action < len(DIRECTIONS):
                return potential_pos, action
            else:
                # e.g. STAY action: keep facing
                return potential_pos, facing

        # No move possible -> stay in place
        return cur_pos, facing

    def next_step(self) -> None:
        # Use peek_step with the current env agent position,
        # then mutate self.pos and self.facing_direction.

        other_guards = {g.pos for g in self.env.guards if g.pos != self.pos}
        new_pos, new_facing_direction = Guard.peek_step(
            self.env, self.pos, self.facing_direction, self.env.agent_pos, other_guards
        )
        self.pos = new_pos
        self.facing_direction = new_facing_direction


class GridWorld:
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
            guards.append(Guard(self, guard_pos, facing_direction))

        return guards

    def _build_mdp_state(self, agent_pos: Pos) -> MDPState:
        guard_states = tuple((g.pos, g.facing_direction) for g in self.guards)
        return (agent_pos, guard_states)

    def in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos[0] < self.config.n_rows and 0 <= pos[1] < self.config.n_cols

    def is_wall(self, pos: Pos) -> bool:
        return pos in self.config.walls

    def is_goal_pos(self, pos: Pos) -> bool:
        return pos in self.config.goals

    def is_agent(self, pos: Pos) -> bool:
        return self.agent_pos == pos

    def is_caught(self, mdp_state: MDPState) -> bool:
        agent_pos, guard_states = mdp_state

        for g_pos, facing_direction in guard_states:
            if g_pos == agent_pos:
                return True

            if self._is_visible_from_guard(agent_pos, g_pos, facing_direction):
                return True

        return False

    def _is_visible_from_guard(
        self, agent_pos: Pos, guard_pos: Pos, facing: FacingDirection
    ) -> bool:
        # Simple FOV: Straight ray in facing direction up to VISION_RANGE,
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

    def peek_step(self, mdp_state: MDPState, action: int) -> MDPState:
        # Pure simulation: Given an arbitrary MDP state and action,
        # return the next MDP state (agent + guards) without side effects.
        # Guards also avoid colliding with each other.

        agent_pos, guard_states = mdp_state

        # Simulate agent
        new_agent_pos = self._next_agent_pos(agent_pos, action)

        # Simulate guards based on new agent position
        new_guard_states = []
        original_positions = [g_pos for (g_pos, _) in guard_states]

        for idx, (g_pos, facing_direction) in enumerate(guard_states):
            # Other guards' positions are:
            #   - all original positions except this guard's
            #   - all new positions already assigned earlier in this loop
            other_guards = set(original_positions[:idx] + original_positions[idx + 1 :])
            other_guards.update(pos for (pos, _) in new_guard_states)

            ng_pos, ng_facing_direction = Guard.peek_step(
                self,
                g_pos,
                facing_direction,
                new_agent_pos,
                other_guards,
            )
            new_guard_states.append((ng_pos, ng_facing_direction))

        return (new_agent_pos, tuple(new_guard_states))

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

        # No safe alternative found -> fall back to original
        return action

    def _next_agent_pos(self, agent_pos: Pos, action: int) -> Pos:
        # Compute the agent's next position given an action.

        # Only grid bounds are treated as hard constraints here.
        # Wall cells are not blocked at the dynamics level – they are
        # handled as 'unsafe' by the safety spec / shield (via labels/DFA),
        # or avoided by the fallback _select_action policy in the
        # unshielded baseline.

        dr, dc = ACTIONS[action]
        new_pos = (agent_pos[0] + dr, agent_pos[1] + dc)

        # Only enforce grid bounds here; walls are modeled via labels / DFA
        if not self.in_bounds(new_pos):
            return agent_pos  # bump into border -> stay

        return new_pos
