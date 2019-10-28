from argparse import ArgumentParser

import numpy as np

from games import games, GameState, GameOutcomes
from mcts import init_tree, mcts


def interactive_play(nn, game: GameState, start, **game_args):
    game_outcome = None

    state = game.init(**game_args)
    tree = init_tree(state, nn)
    turn = -1 if start else 0

    while game_outcome is None:
        turn += 1

        is_player_turn = (turn % 2 == 0)

        if is_player_turn:
            print("Turn {}".format(turn//2))
            print(str(state))
            edge = None
            while edge is None:
                move = int(input("What's your next move? (0..6): "))
                possible_edges = [edge for edge in tree.edges if edge.action == move]
                if len(possible_edges) == 0:
                    print("Invalid action, pick another one")
                else:
                    edge = possible_edges[0]
                    edge.expand()
            action = edge.action
            tree = edge.node
            state = tree.state
        else:
            optimal_pi = mcts(tree)

            edge = sample_edge(tree, optimal_pi)
            action = edge.action
            tree = edge.node
            state = tree.state

        game_outcome = state.game_outcome(last_move=action)

    print("Game finished")
    if game_outcome == GameOutcomes.DRAW:
        print("It was a draw!!")
    elif is_player_turn:
        print("You won!")
        state.inverse()
        print(str(state))
    else:
        print("You loose :(")
        print(str(state))


def sample_edge(tree, optimal_pi):
    return np.random.choice(tree.edges, p=optimal_pi)


class NNMock:
    def train(*args, **kwargs):
        pass

    def evaluate(*args, **kwargs):
        return np.random.random(7), np.random.choice(7)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        'game', choices=games.keys(),
        help="The game you want to play",
    )
    parser.add_argument(
        '--no-start', '-n', default=False, action="store_true",
        help="Let the algorithm start (furst turn)"
    )
    args = parser.parse_args()

    interactive_play(NNMock(), games[args.game], start=(not args.no_start))
