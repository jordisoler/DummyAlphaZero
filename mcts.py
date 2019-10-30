from itertools import chain

import numpy as np

from games import GameState, GameOutcomes


C_PUCT = 1
DEFAULT_TAU = 1.0
MCTS_ITERATIONS = 100
_STATES_PREDICTION_CACHE = {}
_CACHE_HIT = {'yes': 0, 'no': 0}


class Node:
    def __init__(self, state: GameState, probabilities, nn):
        self.state = state
        self.nn = nn

        if len(probabilities):
            actions = state.possible_actions()
            self.edges = [Edge(state, action, prob, nn) for prob, action in zip(probabilities, actions)]
        else:
            self.edges = []

    def expand(self):
        if len(self.edges) > 1:
            best_action_idx = np.argmax(self.edges_value())
        else:
            best_action_idx = 0

        return self.edges[best_action_idx].expand()

    def cache_leaf_nodes(self):
        leaf_states = list(self.get_leaf_states())
        to_request = [state for state in leaf_states if hash(state) not in _STATES_PREDICTION_CACHE]
        if to_request:
            ps, vs = self.nn.predict_from_states(to_request)

            for state, p, v in zip(to_request, ps, vs):
                _STATES_PREDICTION_CACHE[hash(state)] = (p, v)

    def get_leaf_states(self, depth=2):
        values = np.argsort(self.edges_value())[-depth:]
        return chain.from_iterable(
            edge.get_leaf_states()
            for idx, edge in enumerate(self.edges)
            if idx in values
        )

    def edges_value(self):
        nominator = np.sqrt(np.sum([edge.N for edge in self.edges]))
        us = [C_PUCT * edge.P * nominator / (1 + edge.N) for edge in self.edges]
        return [edge.Q + ui for ui, edge in zip(us, self.edges)]


class TerminalNode(Node):
    def __init__(self, state, game_outcome: GameOutcomes):
        self.v = {
            GameOutcomes.WIN: 1,
            GameOutcomes.LOSS: -1,
            GameOutcomes.DRAW: 0,
        }[game_outcome]
        super().__init__(state, [], None)

    def expand(self):
        return self.v

    def get_leaf_states(self):
        return []


class Edge:
    def __init__(self, parent_state: GameState, action, p, nn):
        self.node = None
        self.parent_state = parent_state
        self.action = action
        self.nn = nn
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = p

    def expand(self):
        if self.node is None:
            v = self.create_node()
        else:
            v = self.node.expand()

        self.N += 1
        self.W -= v
        self.Q = self.W/self.N
        return -v

    def create_node(self):
        if self.node is not None:
            raise Exception("Can't create a node. Already having a non null one")

        new_state = self.parent_state.take_action(self.action)
        outcome = new_state.game_outcome(last_move=self.action)

        if outcome is None:
            ps, v = self.get_state_prediction(new_state)
            self.node = Node(new_state, ps, self.nn)
        else:
            self.node = TerminalNode(new_state, outcome)
            v = self.node.expand()
        return v

    def get_leaf_states(self):
        if self.node is None:
            return [self.parent_state.take_action(self.action)]
        else:
            return self.node.get_leaf_states()

    def get_state_prediction(self, state):
        h_state = hash(state)
        if h_state in _STATES_PREDICTION_CACHE:
            ps, v = _STATES_PREDICTION_CACHE[h_state]
            _CACHE_HIT['yes'] += 1
        else:
            ps, v = self.nn.predict_from_state(state)
            _CACHE_HIT['no'] += 1
        return ps, v


def mcts(tree: Node, max_iterations=MCTS_ITERATIONS):
    if _CACHE_HIT['yes'] or _CACHE_HIT['no']:
        print('Cache store:', len(_STATES_PREDICTION_CACHE))
        print('Cache hit: {:.2f}%'.format(100*_CACHE_HIT['yes']/(_CACHE_HIT['yes']+_CACHE_HIT['no'])))
    for i in range(max_iterations):
        if i % 10 == 0:
            tree.cache_leaf_nodes()
        tree.expand()
    return compute_pi(tree)


def init_tree(initial_state: GameState, nn):
    ps, _ = nn.predict_from_state(initial_state)
    return Node(initial_state, ps, nn)


def compute_pi(tree: Node, tau=DEFAULT_TAU):
    ns = np.array([edge.N for edge in tree.edges])
    ns_norm = ns**(1/tau)
    pi = np.zeros(tree.state.action_space_size())
    pi[tree.state.possible_actions_mask()] = ns_norm / ns_norm.sum()
    return pi
