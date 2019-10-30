import numpy as np
from time import time

from games import GameState, GameOutcomes
from mcts import MCTS


def selfplay(nn, game: GameState, **game_args):
    states = []
    optimal_pis = []
    game_outcome = None

    state = game.init(**game_args)
    mcts = MCTS(game, nn)
    turn = -1

    times = [time()]
    while game_outcome is None:
        turn += 1
        if turn % 2:
            print("Turn {}".format(turn))
            print(str(state))

        optimal_pi = mcts.search()

        states.append(state)
        optimal_pis.append(optimal_pi)

        action = sample_action(state, optimal_pi)
        mcts.next_turn(action)
        state = state.take_action(action)

        game_outcome = state.game_outcome(last_move=action)
        t_i = time()
        print("Move time: {}".format(t_i - times[-1]))
        times.append(t_i)

    print("Final turn {}".format(turn))
    print("Total time {}".format(times[-1] - times[0]))
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


def sample_action(state, optimal_pi):
    masked_optimal_pi = optimal_pi[state.possible_actions_mask()]
    return np.random.choice(state.possible_actions(), p=masked_optimal_pi)


if __name__ == "__main__":
    from games import Connect4Board

    from neural_network import new_model

    nn = new_model(Connect4Board)
    selfplay(nn, Connect4Board)
