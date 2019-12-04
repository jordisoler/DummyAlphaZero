from argparse import ArgumentParser

from games import games
from neural_network import new_model
from selfplay import selfplay


def main(game, nn, epochs, saving_freq):
    for idx in range(epochs):
        if idx != 0 and idx % saving_freq == 0:
            save_model(game, nn, idx)
        print(f"*** GAME NUMBER {idx} **")
        print()
        selfplay(nn, game)

    if idx % saving_freq != 0:
        save_model(game, nn, idx)


def save_model(game, nn, idx):
    nn.save(f'models/{game.__name__}/{idx}')


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        'game', choices=games.keys(),
        help="The game you want to play",
    )
    parser.add_argument(
        '--saving-freq', '-s', type=int,
        help="Number of epochs after which to checkpoint the model"
    )
    parser.add_argument('--epochs', '-e', type=int, help="Number of selfplay games to run")

    args = parser.parse_args()

    game = games[args.game]
    nn = new_model(game)
    main(game, nn, args.epochs, args.saving_freq)
