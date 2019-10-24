import numpy as np

from games import GameState, GameOutcomes


C_PUCT = 1


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
        nominator = np.sqrt(np.sum([edge.N for edge in self.edges]))
        us = [C_PUCT * edge.P * nominator / (1 + edge.N) for edge in self.edges]
        best_action_idx = np.argmax([edge.Q + ui for ui, edge in zip(us, self.edges)])

        return self.edges[best_action_idx].expand()


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
            ps, v = self.nn.evaluate(new_state)
            self.node = Node(new_state, ps, self.nn)
        else:
            self.node = TerminalNode(new_state, outcome)
            v = self.node.expand()
        return v
