from argparse import ArgumentParser

import numpy as np

from games import games, GameState, GameOutcomes
from selfplay import sample_action
from mcts import MCTS
from neural_network import new_model


def interactive_play(game: GameState, nn, start, **game_args):
    game_outcome = None

    state = game.init(**game_args)
    mcts = MCTS(game, nn)
    turn = -1 if start else 0

    while game_outcome is None:
        turn += 1

        is_player_turn = (turn % 2 == 0)

        if is_player_turn:
            print("Turn {}".format(turn//2))
            print(str(state))
            valid_action = False
            while valid_action is False:
                action = int(input("What's your next move? (0..6): "))
                try:
                    mcts.next_turn(action)
                    valid_action = True
                except ValueError:
                    print("Invalid action, pick another one")
                    valid_action = False

            state = state.take_action(action)

        else:
            optimal_pi = mcts.search()

            action = sample_action(state, optimal_pi)
            mcts.next_turn(action)
            state = state.take_action(action)

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

    game = games[args.game]
    nn = new_model(game)
    interactive_play(game, nn, start=(not args.no_start))
