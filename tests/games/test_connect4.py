import pytest

import numpy as np

from games.connect4 import core


@pytest.mark.parametrize('arr, are_connected', [
    (np.array([0]), False),
    (np.array([0, 0, 1]), False),
    (np.array([1, 1, 1]), False),
    (np.array([1, 1, 1, 1]), True),
    (np.array([0, 1, 1, 1]), False),
    (np.array([0, 1, 1, 1, 1]), True),
    (np.array([0, 1, 1, 1, 1, 0]), True),
    (np.array([0, 0, 1, 1, 1, 0, 1]), False),
])
def test_check_4(arr, are_connected):
    assert core._check_4(arr) == are_connected


@pytest.mark.parametrize('state, heights', [
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 2, 1, 0, 0],
            [2, 1, 2, 1, 1, 0, 0],
        ]),
        np.array([4, 4, 4, 1, 3, 5, 5]),
    ),
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [2, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 2, 1, 0, 0],
            [2, 0, 2, 1, 1, 1, 2],
        ]),
        np.array([2, 5, 4, 1, 3, 4, 4]),
    )
])
def test_next_moves_heights(state, heights):
    p1, p2 = _test_state_to_valid_matrices(state)
    board = core.Connect4Board(p1, p2)
    assert (board._next_moves_heights() == heights).all()


def test_action_internal_consistency():
    state = core.Connect4Board.init()
    for i in range(6):
        possible_actions = state.possible_actions()
        new_action = possible_actions[i % len(possible_actions)]
        state.take_action(new_action)


@pytest.mark.parametrize('state, board_str', [
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [2, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 2, 1, 0, 0],
            [2, 0, 2, 1, 1, 1, 2],
        ]),
        '\n'.join([
            '+---------------+',
            '|               |',
            '|               |',
            '|       +       |',
            '| o     o       |',
            '| +     o +     |',
            '| o   o + + + o |',
            '+---------------+',
        ])
    )
])
def test_board_print(state, board_str):
    p1, p2 = _test_state_to_valid_matrices(state)
    board = core.Connect4Board(p1, p2)
    assert str(board) == board_str


@pytest.mark.parametrize('state, actions', [
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [2, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 2, 1, 0, 0],
            [2, 0, 2, 1, 1, 1, 2],
        ]),
        np.array([True]*7),
    ),
    (
        np.array([
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 2, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1],
            [2, 0, 0, 2, 0, 0, 2],
            [1, 0, 0, 2, 1, 0, 2],
            [2, 0, 2, 1, 1, 1, 2],
        ]),
        np.array([True, True, True, False, True, True, False]),
    ),
])
def test_possible_actions_mask(state, actions):
    p1, p2 = _test_state_to_valid_matrices(state)
    board = core.Connect4Board(p1, p2)
    assert (board.possible_actions_mask() == actions).all()


def test_init():
    init_state = core.Connect4Board.init()
    assert (init_state.player1 == init_state.player2).all()
    assert init_state.player1.shape == (6, 7)
    assert (init_state.player1 == 0).all()
    assert init_state.player1.dtype == int


@pytest.mark.parametrize('state, last_move, outcome', [
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [2, 0, 0, 2, 1, 0, 0],
            [1, 0, 2, 2, 1, 2, 1],
        ]),
        3,
        None
    ),
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 1, 1, 1],
        ]),
        3,
        core.GameOutcomes.LOSS
    ),
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 1, 1, 1],
        ]),
        1,
        core.GameOutcomes.LOSS
    ),
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 2, 1, 0, 0, 0],
            [0, 2, 1, 1, 0, 0, 0],
            [2, 1, 1, 1, 0, 0, 0],
        ]),
        0,
        core.GameOutcomes.LOSS
    ),
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 2, 1, 0],
            [0, 0, 0, 2, 1, 1, 0],
            [0, 0, 2, 1, 1, 1, 0],
        ]),
        4,
        core.GameOutcomes.LOSS
    ),
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 1, 2, 0, 0],
            [0, 0, 0, 2, 1, 2, 0],
            [0, 0, 0, 1, 1, 1, 2],
        ]),
        5,
        core.GameOutcomes.LOSS
    ),
    (
        np.array([
            [1, 2, 2, 1, 1, 2, 2],
            [2, 1, 1, 2, 2, 1, 1],
            [1, 2, 2, 1, 1, 2, 2],
            [2, 1, 1, 2, 2, 1, 1],
            [1, 2, 2, 1, 1, 2, 2],
            [2, 1, 1, 2, 2, 1, 1],
        ]),
        5,
        core.GameOutcomes.DRAW
    ),
])
def test_game_outcome(state, last_move, outcome):
    p1, p2 = _test_state_to_valid_matrices(state)
    board = core.Connect4Board(p1, p2)
    assert board.game_outcome(last_move=last_move) == outcome


@pytest.mark.parametrize('state, action, new_state', [
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 2, 1, 0, 0],
            [2, 1, 2, 1, 1, 0, 0],
        ]),
        2,
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [2, 1, 2, 1, 1, 0, 0],
        ]),
    ),
    (
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 2, 0],
            [2, 0, 0, 2, 0, 1, 0],
            [1, 0, 0, 2, 1, 2, 0],
            [2, 0, 2, 1, 1, 1, 2],
        ]),
        5,
        np.array([
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 2, 0],
            [2, 0, 0, 2, 0, 1, 0],
            [1, 0, 0, 2, 1, 2, 0],
            [2, 0, 2, 1, 1, 1, 2],
        ]),
    ),
])
def test_take_action(state, action, new_state):
    p1, p2 = _test_state_to_valid_matrices(state)
    new_p2, new_p1 = _test_state_to_valid_matrices(new_state)

    board = core.Connect4Board(p1, p2)
    new_board = board.take_action(action)

    updated_p1 = new_board.player1
    updated_p2 = new_board.player2

    assert (updated_p1 == new_p1).all()
    assert (updated_p2 == new_p2).all()


def _test_state_to_valid_matrices(state):
    return 1*(state == 1), 1*(state == 2)
