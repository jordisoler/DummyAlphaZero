from itertools import chain

import numpy as np

import config
from games import GameState, GameOutcomes


class MCTS:
    def __init__(self, game, nn, max_iterations=config.MCTS_ITERATIONS):
        self.nn = nn
        self.max_iterations = max_iterations
        self.cache = StatesCache()
        self.tree = init_tree(game, nn, self.cache)

    def search(self):
        print(f"Cache status: {self.cache}")

        for i in range(self.max_iterations):
            if i % 10 == 0:
                self.tree.cache_leaf_nodes()
            self.tree.expand()
        return compute_pi(self.tree)

    def next_turn(self, action):
        self.tree = self.tree.take_action(action)


class StatesCache:
    def __init__(self):
        self._cache = {}
        self.requested = 0
        self.found = 0

    def __len__(self):
        return len(self._cache)

    def __contains__(self, state):
        return hash(state) in self._cache

    def __repr__(self):
        return f"<StatesCache | {len(self)} elements | {100*self.hit:.1f}% hit>"

    @property
    def hit(self):
        if self.requested:
            hit = self.found / (self.found+self.requested)
        else:
            hit = np.nan
        return hit

    def get_state_output(self, state: GameState):
        self.requested += 1

        h_state = hash(state)
        if h_state in self._cache:
            self.found += 1
            return self._cache[h_state]

    def store_state(self, state, result):
        self._cache[hash(state)] = result


class Node:
    def __init__(self, state: GameState, probabilities, nn, cache=None):
        self.state = state
        self.nn = nn
        self.cache = cache

        if len(probabilities):
            actions = state.possible_actions()
            self.edges = [
                Edge(state, action, prob, nn, cache=cache)
                for prob, action in zip(probabilities, actions)
            ]
        else:
            self.edges = []

    def expand(self):
        if len(self.edges) > 1:
            best_action_idx = np.argmax(self.edges_value())
        else:
            best_action_idx = 0

        return self.edges[best_action_idx].expand()

    def take_action(self, action):
        edge = [edge for edge in self.edges if edge.action == action][0]
        return edge.node

    def cache_leaf_nodes(self):
        leaf_states = list(self.get_leaf_states())
        to_request = [state for state in leaf_states if state not in self.cache]
        if to_request:
            ps, vs = self.nn.predict_from_states(to_request)

            for state, p, v in zip(to_request, ps, vs):
                self.cache.store_state(state, (p, v))

    def get_leaf_states(self, depth=2):
        values = np.argsort(self.edges_value())[-depth:]
        return chain.from_iterable(
            edge.get_leaf_states()
            for idx, edge in enumerate(self.edges)
            if idx in values
        )

    def edges_value(self):
        nominator = np.sqrt(np.sum([edge.N for edge in self.edges]))
        us = [config.C_PUCT * edge.P * nominator / (1 + edge.N) for edge in self.edges]
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
    def __init__(self, parent_state: GameState, action, p, nn, cache=None):
        self.node = None
        self.parent_state = parent_state
        self.action = action
        self.nn = nn
        self.cache = cache
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
            self.node = Node(new_state, ps, self.nn, cache=self.cache)
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
        prediction = self.cache.get_state_output(state) if self.cache else None
        if prediction is None:
            prediction = self.nn.predict_from_state(state)
        return prediction


def init_tree(game: type, nn, cache: StatesCache):
    initial_state = game.init()
    ps, _ = nn.predict_from_state(initial_state)
    return Node(initial_state, ps, nn, cache=cache)


def compute_pi(tree: Node, tau=config.DEFAULT_TAU):
    ns = np.array([edge.N for edge in tree.edges])
    ns_norm = ns**(1/tau)
    pi = np.zeros(tree.state.action_space_size())
    pi[tree.state.possible_actions_mask()] = ns_norm / ns_norm.sum()
    return pi
