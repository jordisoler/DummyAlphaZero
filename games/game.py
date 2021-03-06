from enum import Enum, auto

import numpy as np


class GameOutcomes(Enum):
    WIN = auto()
    LOSS = auto()
    DRAW = auto()


class GameState:
    def __init__(self):
        """
        This should define the game state
        """

    @classmethod
    def init(cls):
        """
        Create the initial game state
        """

    def __hash__(self):
        raise NotImplementedError

    def take_action(self, action):
        """
        Define the state transition after a given action
        """
        raise NotImplementedError

    def possible_actions(self):
        raise NotImplementedError

    def possible_actions_mask(self):
        raise NotImplementedError

    def action_space_mask(self):
        return self.possible_actions_mask().astype(np.float32)

    def game_outcome(self):
        """
        Check if this is a terminal state and get the corresponding game outcomeself.
        Return None for unfinished games
        """
        raise NotImplementedError
