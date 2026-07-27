from __future__ import annotations

import random
import time
from collections import deque
from enum import IntEnum


Cell = int | str
Coordinate = tuple[int, int]


class GameState(IntEnum):
    IN_PROGRESS = 0
    WON = 1
    LOST = 2
    STOPPED = 3


class MinesweeperGame:
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        mine_count: int = 15,
        mine_locations: list[Coordinate] | None = None,
        seed: int | None = None,
        first_safe: bool = False,
    ):
        self.width = width
        self.height = height
        self.mine_count = mine_count
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than zero.")
        total_cells = width * height
        if mine_count < 0:
            raise ValueError("mine_count must be zero or greater.")
        if mine_count > total_cells:
            raise ValueError("mine_count cannot exceed the number of board cells.")
        if first_safe and mine_count >= width * height:
            raise ValueError("first_safe requires at least one non-mine cell.")
        self.mine_locations = (
            list(mine_locations)
            if mine_locations is not None
            else self.random_mine_locations(
                int(time.time() * 1000000) if seed is None else seed,
                avoid_first_cell=first_safe,
            )
        )
        self.mine_map = self._create_mine_map()
        self.visible_map: list[list[Cell]] = [
            ["-" for _ in range(self.width)] for _ in range(self.height)
        ]
        self.remaining_cells = self.width * self.height - len(self.mine_locations)
        self.state = GameState.IN_PROGRESS

    def random_mine_locations(
        self,
        seed: int | None = None,
        avoid_first_cell: bool = False,
    ) -> list[Coordinate]:
        rng = random.Random(seed)
        mine_locations: set[Coordinate] = set()

        while len(mine_locations) < self.mine_count:
            mine_index = rng.randint(0, self.width * self.height - 1)
            x = mine_index % self.width
            y = mine_index // self.width

            if avoid_first_cell and (x, y) == (0, 0):
                continue

            mine_locations.add((x, y))

        return list(mine_locations)

    def _create_mine_map(self) -> list[list[Cell]]:
        mine_map: list[list[Cell]] = [[0 for _ in range(self.width)] for _ in range(self.height)]

        for mine_x, mine_y in self.mine_locations:
            mine_map[mine_y][mine_x] = "M"
            for neighbor_x, neighbor_y in self.neighbors(mine_x, mine_y):
                if mine_map[neighbor_y][neighbor_x] != "M":
                    mine_map[neighbor_y][neighbor_x] += 1

        return mine_map

    def neighbors(self, x: int, y: int) -> list[Coordinate]:
        neighbors: list[Coordinate] = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def open_cell(self, x: int, y: int) -> bool:
        if self.visible_map[y][x] == "F":
            return True

        if self.mine_map[y][x] == "M":
            self.visible_map[y][x] = "M"
            self.state = GameState.LOST
            return False

        if self.visible_map[y][x] != "-":
            return True

        queue = deque([(x, y)])
        visited = {(x, y)}

        while queue:
            current_x, current_y = queue.popleft()

            if self.visible_map[current_y][current_x] != "-":
                continue

            cell_value = self.mine_map[current_y][current_x]
            self.visible_map[current_y][current_x] = cell_value
            self.remaining_cells -= 1

            if cell_value == 0:
                for neighbor_x, neighbor_y in self.neighbors(current_x, current_y):
                    if (neighbor_x, neighbor_y) in visited:
                        continue
                    if self.mine_map[neighbor_y][neighbor_x] == "M":
                        continue

                    visited.add((neighbor_x, neighbor_y))
                    queue.append((neighbor_x, neighbor_y))

        if self.remaining_cells == 0:
            self.state = GameState.WON

        return True

    def toggle_flag(self, x: int, y: int) -> bool:
        if self.visible_map[y][x] == "-":
            self.visible_map[y][x] = "F"
            return True
        if self.visible_map[y][x] == "F":
            self.visible_map[y][x] = "-"
            return False
        return False

    def clone(self) -> MinesweeperGame:
        cloned = MinesweeperGame(
            width=self.width,
            height=self.height,
            mine_count=self.mine_count,
            mine_locations=self.mine_locations,
        )
        cloned.visible_map = [row[:] for row in self.visible_map]
        cloned.remaining_cells = self.remaining_cells
        cloned.state = self.state
        return cloned

    def display_mine_map(self) -> None:
        print("current mine location", self.mine_locations)
        print("===" * 30)
        self._display_map(self.mine_map)

    def display_visible_map(self) -> None:
        self._display_map(self.visible_map)

    def _display_map(self, board: list[list[Cell]]) -> None:
        print("  " + " ".join(str(i % 10) for i in range(self.width)))
        print("  " + "---" * self.width)
        for row_index, row in enumerate(board):
            print(f"{row_index % 10}: " + " ".join(map(str, row)))
