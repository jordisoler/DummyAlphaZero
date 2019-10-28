# flake8: noqa

from .game import GameOutcomes, GameState
from .connect4.core import Connect4Board

games = {
    'connect4': Connect4Board,
}