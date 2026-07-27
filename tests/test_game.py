import unittest

from minesweeper_ml.game import GameState, MinesweeperGame


class MinesweeperGameTest(unittest.TestCase):
    def test_rejects_non_positive_board_dimensions(self):
        for width, height in ((0, 2), (2, 0), (-1, 2), (2, -1)):
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    MinesweeperGame(width=width, height=height, mine_count=0)

    def test_rejects_negative_mine_count(self):
        with self.assertRaises(ValueError):
            MinesweeperGame(width=2, height=2, mine_count=-1)

    def test_first_safe_option_keeps_zero_zero_from_being_a_mine(self):
        for seed in range(20):
            game = MinesweeperGame(
                width=2,
                height=2,
                mine_count=3,
                seed=seed,
                first_safe=True,
            )

            self.assertNotIn((0, 0), game.mine_locations)

    def test_first_safe_option_protects_only_zero_zero(self):
        game = MinesweeperGame(
            width=2,
            height=1,
            mine_count=1,
            seed=7,
            first_safe=True,
        )

        self.assertEqual({(1, 0)}, set(game.mine_locations))

    def test_open_empty_cell_reveals_connected_empty_area_and_border_numbers(self):
        game = MinesweeperGame(width=3, height=3, mine_count=1, mine_locations=[(2, 2)])

        self.assertIs(game.open_cell(0, 0), True)

        self.assertEqual(GameState.WON, game.state)
        self.assertEqual(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, "-"],
            ],
            game.visible_map,
        )
        self.assertEqual(0, game.remaining_cells)

    def test_open_mine_reveals_mine_and_loses_game(self):
        game = MinesweeperGame(width=2, height=2, mine_count=1, mine_locations=[(1, 1)])

        self.assertIs(game.open_cell(1, 1), False)

        self.assertEqual(GameState.LOST, game.state)
        self.assertEqual("M", game.visible_map[1][1])

    def test_flagged_cell_cannot_be_opened_until_unflagged(self):
        game = MinesweeperGame(width=2, height=2, mine_count=1, mine_locations=[(1, 1)])

        self.assertIs(game.toggle_flag(1, 1), True)
        self.assertIs(game.open_cell(1, 1), True)
        self.assertEqual(GameState.IN_PROGRESS, game.state)
        self.assertEqual("F", game.visible_map[1][1])

        self.assertIs(game.toggle_flag(1, 1), False)
        self.assertEqual("-", game.visible_map[1][1])

    def test_clone_preserves_state_without_sharing_visible_rows(self):
        game = MinesweeperGame(
            width=3,
            height=2,
            mine_count=1,
            mine_locations=[(2, 1)],
        )
        game.open_cell(0, 0)

        cloned = game.clone()
        cloned.visible_map[0][0] = "-"

        self.assertEqual(game.mine_locations, cloned.mine_locations)
        self.assertEqual(game.remaining_cells, cloned.remaining_cells)
        self.assertEqual(game.state, cloned.state)
        self.assertNotEqual(game.visible_map, cloned.visible_map)
