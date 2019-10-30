import pytest
import numpy as np

from games import GameState, GameOutcomes


class CounterGame(GameState):
    """
    Dummy game for testing purposes. Two players. In turns
    each player is asked wether she wants to respond True or False
    the first one with N Trues wins
    """

    def __init__(self, count_a=0, count_b=0, N=4):
        self.count_a = count_a
        self.count_b = count_b
        self.N = N

    def __hash__(self):
        return hash((self.count_a, self.count_b))

    @classmethod
    def init(cls):
        return cls()

    def take_action(self, action):
        count_a = self.count_a + int(bool(action))
        return CounterGame(count_a=self.count_b, count_b=count_a)

    def possible_actions(self):
        return [True, False]

    def game_outcome(self, *args, **kwargs):
        if self.count_a == self.N:
            return GameOutcomes.WIN
        elif self.count_b == self.N:
            return GameOutcomes.LOSS


class CounterGameNN:
    def __init__(self, probability=0.4):
        self.probability = probability

    def predict_from_state(self, state: CounterGame, **kwargs):
        ps = [self.probability, 1-self.probability]
        v = 0.5
        return ps, v


class CounterGameOptimalNN:
    def evaluate(state, *args, **kwargs):
        v = state.count_a / state.N
        return [1, 0], v


@pytest.fixture
def counter_game():
    return CounterGame


@pytest.fixture
def counter_game_nn():
    return CounterGameNN


@pytest.fixture
def counter_game_optimal_nn():
    return CounterGameOptimalNN


@pytest.fixture
def np_random():
    np.random.seed(11)
    return np.random
