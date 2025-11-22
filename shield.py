from gridworld import GridWorld


class SafetyShield:
    def __init__(self, env: GridWorld):
        self.env = env

    def is_action_safe(self, state_index: int, action: int) -> bool:
        state = self.env.index_to_state(state_index)

        next_state = (
            state[0] + GridWorld.ACTIONS[action][0],
            state[1] + GridWorld.ACTIONS[action][1],
        )

        if (
            not self.env.in_bounds(next_state)
            or self.env.is_obstacle(next_state)
            or self.env.is_hazard(next_state)
        ):
            return False
        else:
            return True

    def filter_action(self, state_index: int, action: int) -> int:
        if self.is_action_safe(state_index, action):
            return action
        else:
            # Find a safe action
            for a in range(len(GridWorld.ACTIONS)):
                if self.is_action_safe(state_index, a):
                    return a

            # If no safe action is found, return the original action
            return action
