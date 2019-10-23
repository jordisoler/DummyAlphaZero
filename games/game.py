from enum import Enum, auto


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

    def take_action(self, action):
        """
        Define the state transition after a given action
        """
        raise NotImplementedError

    def possible_actions(self):
        raise NotImplementedError

    def game_outcome(self):
        """
        Check if this is a terminal state and get the corresponding game outcomeself.
        Return None for unfinished games
        """
        raise NotImplementedError
