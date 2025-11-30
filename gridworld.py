from dataclasses import dataclass
from itertools import product
import random
from typing import NoReturn
from typing import Optional

from common.constants import ACTIONS
from common.constants import DIRECTIONS
from common.constants import GUARD_VISION_RANGE
from common.types import Action
from common.types import FacingDirection
from common.types import GuardState
from common.types import MDPState
from common.types import Pos
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
    rng: Optional[random.Random] = None


class GridConfigFactory:
    """
    Factory for constructing grid configurations.

    This class is a utility collection and must not be instantiated.
    """

    @dataclass
    class GridContext:
        n_rows: int
        n_cols: int
        start: Pos
        goals: list[Pos]
        path: list[Pos]
        walls: list[Pos]
        guards: list[GuardState]
        rng: random.Random

    def __new__(
        cls: type["GridConfigFactory"], *args: object, **kwargs: object
    ) -> NoReturn:
        raise TypeError("GridConfigFactory is not instantiable.")

    @classmethod
    def build_random_config(
        cls: type["GridConfigFactory"],
        n_rows: int,
        n_cols: int,
        walls_fraction: float = 0.2,
        n_guards: int = 1,
        seed: int = 0,
    ) -> GridConfig:
        """
        Build a random grid configuration.

        The procedure:
            1. Randomly choose a start and goal.
            2. Build a simple path from start to goal.
            3. Place walls randomly (`walls_fraction` of the grid),
               avoiding start, goals, and the path.
            4. Place `n_guards` guards in free cells.

        Args:
            n_rows: Number of grid rows.
            n_cols: Number of grid columns.
            walls_fraction: Fraction of cells to be turned into walls.
            n_guards: Number of guards to place.
            seed: RNG seed for reproducibility.

        Returns:
            A `GridConfig` instance with randomized layout and guards.
        """

        context = cls.GridContext(
            n_rows=n_rows,
            n_cols=n_cols,
            start=(0, 0),
            goals=[],
            path=[],
            walls=[],
            guards=[],
            rng=random.Random(seed),
        )

        cls._select_start_goal(context)
        cls._generate_simple_path(context)
        cls._generate_random_walls(context, walls_fraction)
        cls._generate_random_guards(context, n_guards)

        return GridConfig(
            n_rows=n_rows,
            n_cols=n_cols,
            start=context.start,
            goals=tuple(context.goals),
            walls=tuple(context.walls),
            guards=tuple(context.guards),
            reward_goal=10.0,
            penalty_if_caught=-10.0,
            penalty_step=-0.1,
            terminate_on_completion=True,
            terminate_if_caught=True,
            rng=context.rng,
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
        context.goals.append(goal)

    @staticmethod
    def _generate_simple_path(context: GridContext) -> None:
        start = context.start
        goals = context.goals

        path = [start]
        current = start

        while current not in goals:
            current = GridConfigFactory._next_step(current, goals, context.rng)
            path.append(current)

        context.path = path

    @staticmethod
    def _next_step(current: Pos, goals: list[Pos], rng: random.Random) -> Pos:
        row_step = 0
        col_step = 0
        goal = goals[0]

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
    def _generate_random_walls(context: GridContext, walls_fraction: float) -> None:
        n_rows = context.n_rows
        n_cols = context.n_cols
        start = context.start
        goals = context.goals
        path = context.path
        walls = context.walls
        rng = context.rng

        n_walls = int((n_rows * n_cols) * walls_fraction)

        while n_walls > 0:
            wall = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))

            if (
                wall != start
                and wall not in goals
                and wall not in path
                and wall not in walls
            ):
                walls.append(wall)
                n_walls -= 1

    @staticmethod
    def _generate_random_guards(context: GridContext, n_guards: int) -> None:
        n_rows = context.n_rows
        n_cols = context.n_cols
        start = context.start
        goals = context.goals
        path = context.path
        walls = context.walls
        guards = context.guards
        rng = context.rng

        while n_guards > 0:
            guard_pos = (rng.randint(0, n_rows - 1), rng.randint(0, n_cols - 1))

            if (
                guard_pos != start
                and guard_pos not in goals
                and guard_pos not in path
                and guard_pos not in walls
                and guard_pos not in [g[0] for g in guards]
            ):
                facing_direction = rng.randint(0, len(DIRECTIONS) - 1)
                guards.append((guard_pos, facing_direction))
                n_guards -= 1


class Guard:
    """
    Guard entity with simple local movement rules.

    Args:
        env: Reference to the `GridWorld` environment.
        pos: Initial position of the guard.
        facing_direction: Initial facing direction (index into `DIRECTIONS`/`ACTIONS`).
    """

    def __init__(self, env: "GridWorld", pos: Pos, facing_direction: FacingDirection):
        self.env = env
        self.pos = pos
        self.facing_direction = facing_direction

    @staticmethod
    def peek_step(
        env: "GridWorld",
        cur_pos: Pos,
        facing: FacingDirection,
        agent_pos: Pos,
        other_guards: set[Pos],
    ) -> list[GuardState]:
        """
        Compute all valid next guard states from a given state.

        The guard chooses among neighbors that:
          - are inside the grid,
          - are not walls,
          - are not occupied by the agent,
          - are not occupied by other guards.

        At intersections, the guard avoids immediately backtracking
        (moving opposite to its current facing) if there are other valid options.

        If no movement is possible at all, the guard stays in place.

        Args:
            env: `GridWorld` instance used for queries (bounds/walls).
            cur_pos: Current guard position.
            facing: Current facing direction index.
            agent_pos: Current agent position.
            other_guards: Positions of other guards to avoid.

        Returns:
            A list of `(pos, facing_direction)` pairs representing all valid
            next states for this guard.
        """

        valid_facing_directions = []

        # 1. Collect all valid movement directions (UP/DOWN/LEFT/RIGHT indices 0..len(DIRECTIONS)-1)
        for d in range(len(DIRECTIONS)):
            dr, dc = ACTIONS[d]
            candidate = (cur_pos[0] + dr, cur_pos[1] + dc)

            if (
                not env.in_bounds(candidate)
                or env.is_wall(candidate)
                or candidate == agent_pos
                or candidate in other_guards
            ):
                continue

            valid_facing_directions.append(d)

        # 2. If no directions are valid, staying in place is the only option
        if not valid_facing_directions:
            return [(cur_pos, facing)]

        # 3. Compute the "back" direction (opposite of current facing)
        back_direction = None
        bdr, bdc = ACTIONS[facing]
        back_vec = (-bdr, -bdc)

        for i, (dr, dc) in enumerate(DIRECTIONS):
            if (dr, dc) == back_vec:
                back_direction = i
                break

        # 4. At intersections, avoid the back direction if possible
        if (
            back_direction is not None
            and back_direction in valid_facing_directions
            and len(valid_facing_directions) > 1
        ):
            filtered = [d for d in valid_facing_directions if d != back_direction]

            if filtered:
                valid_facing_directions = filtered

        # 5. Build all possible next guard states
        possible_states = []

        for d in valid_facing_directions:
            dr, dc = ACTIONS[d]
            next_pos = (cur_pos[0] + dr, cur_pos[1] + dc)
            possible_states.append((next_pos, d))

        return possible_states

    def next_step(self) -> None:
        """
        Sample and apply a single guard move.
        """

        other_guards = {g.pos for g in self.env.guards if g.pos != self.pos}
        candidates = Guard.peek_step(
            self.env, self.pos, self.facing_direction, self.env.agent_pos, other_guards
        )

        # Candidates is guaranteed to be non-empty (at least stay)
        new_pos, new_facing_direction = self.env.rng.choice(candidates)
        self.pos = new_pos
        self.facing_direction = new_facing_direction


class GridWorld:
    """
    Gridworld MDP with guards and an optional safety shield.

    Args:
        config: `GridConfig` specifying layout, rewards, and guards.
        shield: Optional `SafetyShield` that filters agent actions.
    """

    def __init__(self, config: GridConfig, shield: Optional[SafetyShield] = None):
        self.config = config
        self.shield = shield
        self.rng = config.rng or random.Random()

        # Initialize guards as objects
        self.guards = self._init_guards()

        # Current agent position
        self.agent_pos = self.config.start

        # Track remaining goals as a list
        self.goals = list(self.config.goals)

        # Build the initial MDP state
        self.cur_state = self._build_mdp_state(self.agent_pos)
        self.initial_state = self.cur_state

        self.done = False

        # Cache for peek_step
        self._succ_cache: dict[tuple[MDPState, Action], set[MDPState]] = {}

    def _init_guards(self) -> list[Guard]:
        guards = []

        for guard_pos, facing_direction in self.config.guards:
            guards.append(Guard(self, guard_pos, facing_direction))

        return guards

    def _build_mdp_state(self, agent_pos: Pos) -> MDPState:
        guard_states = tuple((g.pos, g.facing_direction) for g in self.guards)
        return (agent_pos, guard_states)

    def in_bounds(self, pos: Pos) -> bool:
        """
        Check if a position lies within the grid bounds.
        """

        return 0 <= pos[0] < self.config.n_rows and 0 <= pos[1] < self.config.n_cols

    def is_wall(self, pos: Pos) -> bool:
        """
        Check if a position is a wall cell.
        """

        return pos in self.config.walls

    def is_goal_pos(self, pos: Pos) -> bool:
        """
        Check if a position is one of the goal cells.
        """

        return pos in self.config.goals

    def is_agent(self, pos: Pos) -> bool:
        """
        Check if a position is occupied by the agent.
        """

        return self.agent_pos == pos

    def is_guard(self, pos: Pos) -> bool:
        """
        Check if a position is occupied by any guard.
        """

        for guard in self.guards:
            if guard.pos == pos:
                return True

        return False

    def is_caught(self, mdp_state: MDPState) -> bool:
        """
        Check if the agent is caught by any guard in the given state.

        The agent is considered caught if:
          - any guard shares the same cell as the agent, or
          - the agent lies in a guard's field-of-view ray.
        """

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

        for step in range(1, GUARD_VISION_RANGE + 1):
            cell = (r + step * dr, c + step * dc)

            if not self.in_bounds(cell) or self.is_wall(cell):
                break

            if cell == agent_pos:
                return True

        return False

    def mdp_state_to_index(self, mdp_state: MDPState) -> int:
        """
        Map an MDP state to a discrete index (using only agent position).

        The index is `row * n_cols + col`, ignoring guard positions.
        This is intended for use as the state index in tabular RL.

        Args:
            mdp_state: MDP state `(agent_pos, guard_states)`

        Returns:
            Corresponding index in `[0, n_rows * n_cols)`
        """

        row, col = mdp_state[0]
        return row * self.config.n_cols + col

    def index_to_mdp_state(self, index: int) -> MDPState:
        """
        Map an index back to an MDP state, reusing current guard states.

        The agent position is reconstructed from the index; the guard states
        are taken from `self.guards`.

        Args:
            index: State index in `[0, n_rows * n_cols)`.

        Returns:
            Corresponding MDPState `(agent_pos, guard_states)`.
        """

        row = index // self.config.n_cols
        col = index % self.config.n_cols

        # Use current guard object states
        return self._build_mdp_state((row, col))

    def compute_label(self, mdp_state: MDPState) -> int:
        """
        Compute a label for the given MDP state for use by the DFA/shield.

        Label encoding:
            0 = SAFE
            1 = WALL      (agent on a wall cell)
            2 = VISIBLE   (agent seen by guard or sharing cell with guard)

        Args:
            mdp_state: The full MDP state.

        Returns:
            Integer label representing safety category.
        """

        agent_pos, _ = mdp_state

        if self.is_wall(agent_pos):
            return 1
        if self.is_caught(mdp_state):
            return 2

        return 0

    def next_step(self, action: Action) -> tuple[int, float, bool, dict]:
        """
        Perform one environment step given an agent action.

        Pipeline:
            1. Optionally filter the proposed action via the `SafetyShield`.
            2. Sample a successor MDP state via `peek_step`.
            3. Apply the sampled transition to internal agent/guard objects.
            4. Compute reward and termination flags.

        Args:
            action: Proposed agent action (index into `ACTIONS`).

        Returns:
            out: A tuple `(, reward, done, info)` where:
                    - next_state_idx: integer state index for the next agent position,
                    - reward: scalar reward,
                    - done: True if the episode terminated,
                    - info: dict with flags:
                        * "goal": True if a goal was reached this step,
                        * "caught": True if the agent was caught this step.

        Raises:
            RuntimeError: If the episode has already terminated and reset
                has not been called.
        """

        if self.done:
            raise RuntimeError("Episode has terminated. Please reset the environment.")

        # 1. Apply shield to current MDP state (if possible) to filter action
        #    or select action after simple safety checks
        mdp_before = self.cur_state

        if self.shield is not None:
            chosen_action = self.shield.filter_action(mdp_before, action)
        else:
            chosen_action = self._select_action(action)

        # 2. Sample from the same model used for shield/safety game
        successors = self.peek_step(mdp_before, chosen_action)
        next_mdp_state = self.rng.choice(tuple(successors))

        new_agent_pos, new_guard_states = next_mdp_state

        # 3. Apply to env objects
        self.agent_pos = new_agent_pos

        for guard, (g_pos, g_face) in zip(self.guards, new_guard_states):
            guard.pos = g_pos
            guard.facing_direction = g_face

        self.cur_state = next_mdp_state

        if self.shield is not None:
            self.shield.update(self.cur_state, chosen_action)

        # 4. Compute reward and termination
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

    def peek_step(self, mdp_state: MDPState, action: Action) -> set[MDPState]:
        """
        Return all possible successor MDP states for a given state–action pair.

        This method defines the environment’s transition relation in a
        side-effect-free way.

        The transition semantics are:

        *Agent move*
        - The agent attempts to move according to `action` (an index into
          `ACTIONS`).
        - If the target cell is out of bounds, a wall, or currently occupied by
          a guard, the agent stays in place.
        - Otherwise, the agent moves to the target cell.

        *Guard moves*
        - Each guard considers locally valid moves produced by `Guard.peek_step`,
          which only proposes moves that:
            - remain in bounds,
            - avoid walls,
            - avoid the (already updated) agent position, and
            - avoid other guards’ current positions.
        - The joint successor set is formed from all combinations of local guard
          moves in which no two guards occupy the same cell.

        If every joint guard move would result in a guard–guard collision, the
        guards are kept in their original positions and only the agent’s move
        (or stay) is applied.

        Args:
            mdp_state:
                Current MDP state `(agent_pos, guard_states)`, where
                `agent_pos` is a `(row, col)` tuple and `guard_states` is a
                tuple of `(guard_pos, facing_direction)` pairs.
            action:
                Agent action index (index into `ACTIONS`).

        Returns:
            A set of successor MDP states. Each successor has the same
            structure as `mdp_state`: `(next_agent_pos, next_guard_states)`.
        """

        key = (mdp_state, action)
        if key in self._succ_cache:
            return self._succ_cache[key]

        agent_pos, guard_states = mdp_state

        # 1. Compute the agent's new position using only (agent_pos, guard_states)
        #    so that the simulated transition depends purely on the given MDP state
        #    and not on the environment's internal guard objects (env.guards).
        dr, dc = ACTIONS[action]
        new_pos = (agent_pos[0] + dr, agent_pos[1] + dc)

        # Build a set of guard positions from guard_states
        guard_positions = {g_pos for (g_pos, _) in guard_states}

        if (
            not self.in_bounds(new_pos)
            or self.is_wall(new_pos)
            or new_pos in guard_positions
        ):
            new_agent_pos = agent_pos  # Stay
        else:
            new_agent_pos = new_pos

        # 2. For each guard, compute possible next states
        original_positions = [g_pos for (g_pos, _) in guard_states]
        guard_options_per_guard = []

        for idx, (g_pos, facing_direction) in enumerate(guard_states):
            # Other guards' positions for this guard:
            #   - all original positions except this guard's
            #   - all newly chosen positions of earlier guards
            other_guards_original_positions = set(
                original_positions[:idx] + original_positions[idx + 1 :]
            )

            # First, get candidates relative to original layout
            base_candidates = Guard.peek_step(
                self,
                g_pos,
                facing_direction,
                new_agent_pos,
                other_guards_original_positions,
            )

            guard_options_per_guard.append(base_candidates)

        # 3. Combine guards' moves via Cartesian product,
        #    then filter out combinations where guards collide
        next_states = set()

        for combo in product(*guard_options_per_guard):
            # Combo is a tuple of GuardState for each guard
            new_positions = [pos for (pos, _) in combo]

            if len(new_positions) != len(set(new_positions)):
                # Collision: two guards in the same cell
                continue

            next_states.add((new_agent_pos, combo))

        # If all combinations collided somehow (very unlikely), at least keep
        # the original guard layout (i.e., no guard move).
        if not next_states:
            next_states.add((new_agent_pos, guard_states))

        self._succ_cache[key] = next_states

        return next_states

    def _select_action(self, action: Action) -> int:
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

    def _next_agent_pos(self, agent_pos: Pos, action: Action) -> Pos:
        # Compute the agent's next position given an action

        dr, dc = ACTIONS[action]
        new_pos = (agent_pos[0] + dr, agent_pos[1] + dc)

        if (
            not self.in_bounds(new_pos)
            or self.is_wall(new_pos)
            or self.is_guard(new_pos)
        ):
            return agent_pos  # Stay

        return new_pos

    def reset(self) -> int:
        """
        Reset the environment to its initial configuration.

        Resets:
          - agent position to `config.start`,
          - guards to the initial guard states from `config.guards`,
          - remaining goals,
          - `done` flag,
          - current MDP state,
          - shield's DFA state (if present).

        Returns:
            Integer state index corresponding to the reset MDP state.
        """

        self.agent_pos = self.config.start
        self.guards = self._init_guards()
        self.goals = list(self.config.goals)
        self.done = False

        self.cur_state = self._build_mdp_state(self.agent_pos)
        self.initial_state = self.cur_state

        if self.shield is not None:
            self.shield.reset()

        return self.mdp_state_to_index(self.cur_state)
