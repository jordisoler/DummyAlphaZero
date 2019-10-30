import numpy as np

import mcts


def test_mcts_convergence(counter_game, counter_game_nn):
    root_state = counter_game()
    root_node = mcts.Node(root_state, np.array([0.3, 0.5]), counter_game_nn())
    for _ in range(500):
        root_node.expand()

    assert root_node.edges[0].Q - root_node.edges[1].Q > 0.5


def test_get_leaf_states(counter_game, counter_game_nn):
    root_state = counter_game()
    root_node = mcts.Node(root_state, np.array([0.3, 0.5]), counter_game_nn())

    for _ in range(2):
        root_node.expand()

    leaf_states = list(root_node.get_leaf_states())
    assert len(leaf_states) != 0
