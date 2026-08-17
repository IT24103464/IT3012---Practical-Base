# agent.py
import random
from collections import deque
import heapq


class SearchAgent:
    def bfs_search(self, start, goal, walls, grid_size):
        frontier = deque()
        reached = set()
        frontier.append((start, []))
        reached.add(start)
        directions = {
                'Up': (0, 1),
                'Right': (1, 0),
                'Down': (0, -1),
                'Left': (-1, 0),
            }
        while frontier:
            current_position, path = frontier.popleft()
            if current_position == goal:
                return path
            for action, (dx, dy) in directions.items():
                next_position = (
                    current_position[0] + dx,
                    current_position[1] + dy
                )
                if (
                    next_position[0] < 0
                    or next_position[0] >= grid_size[0]
                    or next_position[1] < 0
                    or next_position[1] >= grid_size[1]
                ):
                    continue
                if next_position in walls:
                    continue
                if next_position in reached:
                    continue
                reached.add(next_position)
                new_path = path + [action]
                frontier.append((next_position, new_path))
        return None

    def dfs_search(self, start, goal, walls, grid_size):
        frontier = []
        reached = set()
        frontier.append((start, []))
        reached.add(start)
        directions = {
                'Up': (0, 1),
                'Right': (1, 0),
                'Down': (0, -1),
                'Left': (-1, 0),
            }
        while frontier:
            current_position, path = frontier.pop()
            if current_position == goal:
                return path
            for action, (dx, dy) in directions.items():
                next_position = (
                    current_position[0] + dx,
                    current_position[1] + dy
                )
                if (
                    next_position[0] < 0
                    or next_position[0] >= grid_size[0]
                    or next_position[1] < 0
                    or next_position[1] >= grid_size[1]
                ):
                    continue
                if next_position in walls:
                    continue
                if next_position in reached:
                    continue
                reached.add(next_position)
                new_path = path + [action]
                frontier.append((next_position, new_path))
        return None

    def ucs_search(self):
        pass

class SimpleReflexAgent:
    """Uses only the current percept and stores no history."""

    def sense_and_act(self, percept: dict) -> str:
        # Condition-action rules based only on the current percept.
        if percept["food_here"]:
            return "Suck"

        if percept["wall_ahead"]:
            return random.choice(["TurnLeft", "TurnRight"])

        # Keep the original random exploration style while usually moving forward.
        return random.choice([
            "MoveForward",
            "MoveForward",
            "TurnLeft",
            "TurnRight",
        ])


class ModelBasedAgent:
    """Uses an internal state to avoid repeatedly trying the same failed routes."""

    DIRECTIONS = ["Up", "Right", "Down", "Left"]

    DIRECTION_VECTORS = {
        "Up": (0, 1),
        "Right": (1, 0),
        "Down": (0, -1),
        "Left": (-1, 0),
    }

    def __init__(self):
        # The real coordinates are hidden, so the agent tracks a relative position.
        self.relative_position = (0, 0)
        self.facing_index = 0

        # Internal memory state required by the model-based architecture.
        self.visited_cells = {(0, 0)}
        self.blocked_directions = {(0, 0): set()}
        self.parent_direction = {}
        self.target_direction = None

        self.last_action = None
        self.previous_percept = None
        self.percept_history = []

    def _cell_in_direction(self, direction_index: int) -> tuple[int, int]:
        direction = self.DIRECTIONS[direction_index]
        dx, dy = self.DIRECTION_VECTORS[direction]

        return (
            self.relative_position[0] + dx,
            self.relative_position[1] + dy,
        )

    def _turn_towards(self, target_direction: int) -> str:
        difference = (target_direction - self.facing_index) % 4

        if difference == 1:
            return "TurnRight"

        if difference == 3:
            return "TurnLeft"

        # Reversing direction requires two turns. The target is remembered,
        # so the next decision will complete the turn instead of oscillating.
        return random.choice(["TurnLeft", "TurnRight"])

    def sense_and_act(self, percept: dict) -> str:
        # -------------------------------------------------------------
        # Transition model: update the internal state from the last action.
        # -------------------------------------------------------------
        if self.last_action == "TurnLeft":
            self.facing_index = (self.facing_index - 1) % 4

        elif self.last_action == "TurnRight":
            self.facing_index = (self.facing_index + 1) % 4

        elif (
            self.last_action == "MoveForward"
            and self.previous_percept is not None
            and not self.previous_percept["wall_ahead"]
        ):
            new_position = self._cell_in_direction(self.facing_index)

            if new_position not in self.visited_cells:
                # Remember how to return from the newly discovered cell.
                self.parent_direction[new_position] = (self.facing_index + 2) % 4

            self.relative_position = new_position
            self.visited_cells.add(new_position)
            self.blocked_directions.setdefault(new_position, set())
            self.target_direction = None

        # -------------------------------------------------------------
        # Sensor model: record what the current percept tells the agent.
        # -------------------------------------------------------------
        self.percept_history.append({
            "percept": percept.copy(),
            "last_action": self.last_action,
        })

        blocked_here = self.blocked_directions.setdefault(
            self.relative_position,
            set(),
        )

        if percept["wall_ahead"]:
            blocked_here.add(self.facing_index)

            if self.target_direction == self.facing_index:
                self.target_direction = None

        # -------------------------------------------------------------
        # Condition-action rules that query the internal memory.
        # -------------------------------------------------------------
        if percept["food_here"]:
            action = "Suck"

        else:
            # Continue a direction already selected during a previous turn.
            if self.target_direction is not None:
                if self.facing_index != self.target_direction:
                    action = self._turn_towards(self.target_direction)
                elif percept["wall_ahead"]:
                    self.target_direction = None
                    action = random.choice(["TurnLeft", "TurnRight"])
                else:
                    action = "MoveForward"

            else:
                # Randomly choose an unexplored neighbouring cell.
                unexplored_directions = []

                for direction_index in range(4):
                    next_cell = self._cell_in_direction(direction_index)

                    if (
                        direction_index not in blocked_here
                        and next_cell not in self.visited_cells
                    ):
                        unexplored_directions.append(direction_index)

                if unexplored_directions:
                    self.target_direction = random.choice(unexplored_directions)

                    if self.target_direction == self.facing_index:
                        action = "MoveForward"
                    else:
                        action = self._turn_towards(self.target_direction)

                else:
                    # No new path is available here. Backtrack to the parent cell.
                    back_direction = self.parent_direction.get(
                        self.relative_position
                    )

                    if back_direction is not None:
                        self.target_direction = back_direction

                        if self.target_direction == self.facing_index:
                            if percept["wall_ahead"]:
                                blocked_here.add(self.facing_index)
                                self.target_direction = None
                                action = random.choice(["TurnLeft", "TurnRight"])
                            else:
                                action = "MoveForward"
                        else:
                            action = self._turn_towards(self.target_direction)
                    else:
                        # At the starting cell, choose any direction not known blocked.
                        available_directions = [
                            direction_index
                            for direction_index in range(4)
                            if direction_index not in blocked_here
                        ]

                        if available_directions:
                            self.target_direction = random.choice(
                                available_directions
                            )

                            if self.target_direction == self.facing_index:
                                action = "MoveForward"
                            else:
                                action = self._turn_towards(
                                    self.target_direction
                                )
                        else:
                            action = random.choice(["TurnLeft", "TurnRight"])

        self.last_action = action
        self.previous_percept = percept.copy()

        return action
