import numpy as np

from games import GameState, GameOutcomes
from mcts import init_tree, mcts


def selfplay(nn, game: GameState, **game_args):
    states = []
    optimal_pis = []
    game_outcome = None

    state = game.init(**game_args)
    tree = init_tree(state, nn)
    turn = -1

    while game_outcome is None:
        turn += 1
        if turn % 2:
            print("Turn {}".format(turn))
            print(str(state))

        optimal_pi = mcts(tree)

        states.append(state)
        optimal_pis.append(optimal_pi)

        edge = sample_edge(tree, optimal_pi)
        action = edge.action
        tree = edge.node
        state = tree.state

        game_outcome = state.game_outcome(last_move=action)

    print("Final turn {}".format(turn))
    print(str(state))
    if game_outcome == GameOutcomes.DRAW:
        print("It was a draw!!")
    else:
        print(("First" if turn % 2 == 0 else "Second") + " player won")

    if game_outcome == GameOutcomes.DRAW:
        z = [0]*len(states)
    elif game_outcome == GameOutcomes.LOSS:
        z = [(-1)**(i+1) for i in range(len(states), 0, -1)]
    else:
        raise Exception('Invalid game outcome: {}'.format(game_outcome))

    nn.fit_game_state(states, optimal_pis, z)


def sample_edge(tree, optimal_pi):
    masked_optimal_pi = optimal_pi[tree.state.possible_actions_mask()]
    return np.random.choice(tree.edges, p=masked_optimal_pi)


if __name__ == "__main__":
    from games import Connect4Board

    from neural_network import new_model

    nn = new_model(Connect4Board)
    selfplay(nn, Connect4Board)
