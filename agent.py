# agent.py
import random
from collections import deque
import heapq
import math


class SearchAgent:
    def __init__(self):
        self.plan = []
        self.active_algo = 'AStar'
    
        # self.active_algo = 'DFS'
    
        # self.active_algo = 'UCS'

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

    def ucs_search(self, start, goal, walls, grid_size):
        frontier = []
        reached = set()
        heapq.heappush(frontier, (0, start, []))
        reached.add(start)
        directions = {
                'Up': (0, 1),
                'Right': (1, 0),
                'Down': (0, -1),
                'Left': (-1, 0),
            }
        while frontier:
            current_cost, current_position, path = heapq.heappop(frontier)
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
                new_cost = current_cost + 1
                new_path = path + [action]
                heapq.heappush(frontier, (new_cost, next_position, new_path)) 
        return None

    def manhattan_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        x = abs(x1 - x2) + abs(y1 - y2)
        return x

    def euclidean_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        x_diff = x1 - x2
        y_diff = y1 - y2
        x = math.sqrt((x_diff ** 2) + (y_diff ** 2))
        return x

    def sense_and_act(self, percept):
        if percept['food_here']:
            return 'Suck'

        if not self.plan:
            start = percept['agent_pos']
            walls = percept['walls']
            grid_size = percept['grid_size']
            all_food = percept['all_food']

            if not all_food:
                return None

            goal = min(
                all_food,
                key=lambda food: abs(food[0] - start[0]) + abs(food[1] - start[1])
            )

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, goal, walls, grid_size)

            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, goal, walls, grid_size)

            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, goal, walls, grid_size)
            
            elif self.active_algo == 'AStar':
                self.plan = self.astar_search(start, goal, walls, grid_size)

        if self.plan:
            return self.plan.pop(0)

        return None

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        reached = set()
        frontier = []
        directions = {
                'Up': (0, 1),
                'Right': (1, 0),
                'Down': (0, -1),
                'Left': (-1, 0),
            }
        if heuristic_type == 'manhattan':
            h_cost = self.manhattan_distance(start_pos, goal_pos)
        else:
            h_cost = self.euclidean_distance(start_pos, goal_pos)
        g_cost = 0
        f_cost = g_cost + h_cost
        heapq.heappush(frontier, (f_cost, g_cost, start_pos, []))

        while frontier:
            f_cost, g_cost, current_position, path = heapq.heappop(frontier)
            if current_position == goal_pos:
                return path
            reached.add(current_position)
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
                new_path = path + [action]
                if heuristic_type == 'manhattan':
                    h_cost = self.manhattan_distance(next_position, goal_pos)
                else:
                    h_cost = self.euclidean_distance(next_position, goal_pos)
                new_g_cost = g_cost + 1
                new_f_cost = new_g_cost + h_cost
                heapq.heappush(frontier, (new_f_cost, new_g_cost, next_position, new_path))
        return None



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
