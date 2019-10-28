import numpy as np

from games import Connect4Board
import neural_network as nn


def test_connect4_consistency(np_random):
    init_state = Connect4Board.init()
    state_1full = Connect4Board(
        np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ]),
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
        ]),
    )

    probabilities = [np_random.random(7), np_random.random(7)]
    state_values = [np_random.rand()*2-1 for _ in range(2)]

    net = nn.new_model(Connect4Board)
    net.fit_game_state(
        [init_state, state_1full],
        probabilities,
        state_values,
    )
