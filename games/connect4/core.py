import numpy as np

from ..game import GameState, GameOutcomes


class Connect4Board(GameState):
    H = 6
    W = 7

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def __str__(self):
        out = ['+' + '-'*(self.W*2 + 1) + '+']
        for row in range(self.H):
            line = '| '
            for col in range(self.W):
                if self.player1[row, col] == 1:
                    char = '+'
                elif self.player2[row, col] == 1:
                    char = 'o'
                else:
                    char = ' '
                line += char + ' '
            line += '|'
            out.append(line)
        out.append(out[0])
        return '\n'.join(out)

    @classmethod
    def init(cls):
        return cls(np.zeros((cls.H, cls.W), dtype=int), np.zeros((cls.H, cls.W), dtype=int))

    def take_action(self, action):
        next_player1 = self.player2.copy()
        next_player2 = self.player1.copy()

        next_player2[self._next_moves_heights()[action], action] = 1
        return Connect4Board(next_player1, next_player2)

    def possible_actions(self):
        return np.arange(7, dtype=int)[self.possible_actions_mask()]

    def possible_actions_mask(self):
        return self._next_moves_heights() != -1

    def game_outcome(self, last_move=None):
        if last_move is None:
            raise NotImplementedError("We only check if final state from last move")
        if self.won_with_last_move(last_move):
            outcome = GameOutcomes.LOSS
        elif self.is_draw():
            outcome = GameOutcomes.DRAW
        else:
            outcome = None

        return outcome

    def won_with_last_move(self, last_move):
        col = last_move
        row = self._next_moves_heights()[col]+1

        return (
            _check_4(self.player2[row, :]) or
            _check_4(self.player2[:, col]) or
            _check_4(self.player2.diagonal(col-row)) or
            _check_4(np.fliplr(self.player2).diagonal(self.W-col-row-1))
        )

    def is_draw(self):
        return self.player1.sum() + self.player2.sum() == self.W*self.H

    def _next_moves_heights(self):
        return self.H - (self.player1 + self.player2).sum(axis=0) - 1


def _check_4(arr):
    return (
        len(arr) >= 4 and
        sum(arr) >= 4 and
        any(np.convolve(np.ones(4), arr) == 4)
    )
