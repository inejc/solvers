import copy

import numpy as np

from kuhn_poker import N_PLAYERS, N_ACTIONS, N_CARDS, VALID_ACTIONS, State


class Node:
    def __init__(self, state):
        self.state = state
        # beliefs are for both players at each node
        self.beliefs = np.ones((N_PLAYERS, N_CARDS))
        # regrets and strategy are for node's current player
        self.cumulative_regrets = np.zeros((N_ACTIONS, N_CARDS))
        self.current_strategy = np.full((N_ACTIONS, N_CARDS), fill_value=0.5)
        self.cumulative_strategy = np.full(
            (N_ACTIONS, N_CARDS), fill_value=0.5,
        )
        self.children = []

    def update_beliefs(self, action_prob, parent_beliefs):
        # action_prob is probability of the current
        # node action for each hand, i.e. shape = (N_CARDS,)
        self.beliefs = parent_beliefs.copy()
        other_player = 0 if self.state.current_player == 1 else 1
        self.beliefs[other_player] = action_prob * parent_beliefs[other_player]

    def regret_matching(self):
        # computes a new current strategy
        non_negative_regrets = np.maximum(
            self.cumulative_regrets,
            np.zeros(self.cumulative_regrets.shape),
        )

        normalizing_sum = np.sum(non_negative_regrets, axis=0)

        # if normalizing_sum == 0.0 ignore the divide by zero warning
        # and set the strategy to uniform random after the division
        with np.errstate(divide='ignore', invalid='ignore'):
            new_strategy = non_negative_regrets / normalizing_sum
        new_strategy[:, normalizing_sum == 0.0] = 0.5

        self.current_strategy = new_strategy
        self.cumulative_strategy += \
            (self.beliefs[self.state.current_player] * new_strategy)

    def get_average_strategy(self):
        normalizing_sum = np.sum(self.cumulative_strategy, axis=0)

        # if normalizing_sum == 0.0 ignore the divide by zero warning
        # and set the strategy to uniform random after the division
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_strategy = self.cumulative_strategy / normalizing_sum
        avg_strategy[:, normalizing_sum == 0.0] = 0.5

        return avg_strategy

    def evaluate_terminal_state(self, player):
        # computes the counterfactual values of the terminal state
        if not self.state.is_terminal():
            raise ValueError(
                "evaluate_terminal_state called on a non-terminal state",
            )

        opponent = 0 if player == 1 else 1

        if self.state.history[-2:] == "bp":
            # no showdown
            if self.state.current_player == player:
                # player won
                payoff = self.state.bets[opponent]
            else:
                # opponent won
                payoff = -self.state.bets[player]

            return np.full(N_CARDS, fill_value=payoff)

        # showdown
        # possible hands are (J, Q, K)
        payoffs = np.zeros(N_CARDS)

        for my_hand in range(N_CARDS):
            for opponent_hand in range(N_CARDS):
                if my_hand == opponent_hand:
                    # blocker
                    continue

                if my_hand > opponent_hand:
                    payoff = self.state.bets[opponent]
                else:
                    payoff = -self.state.bets[player]

                payoffs[my_hand] += \
                    (self.beliefs[opponent][opponent_hand] * payoff)

        return payoffs

    def __repr__(self):
        average_strategy = self.get_average_strategy()
        result = (
            "node(\n"
            f"  player = {self.state.current_player},\n"
            f"  bets = {self.state.bets},\n"
            f"  history = {self.state.history},\n"
            f"  is_terminal = {self.state.is_terminal()},\n"
            "  beliefs = \n"
            f"    player 0 = {self.beliefs[0]}\n"
            f"    player 1 = {self.beliefs[1]},\n"
        )

        if not self.state.is_terminal():
            result += (
                "  average strategy = \n"
                f"    bet  = {average_strategy[0]}\n"
                f"    pass = {average_strategy[1]},\n"
            )

        return result + ")"


def build_tree(node=None):
    if node is None:
        root = Node(State())
        build_tree(root)
        return root

    if node.state.is_terminal():
        return

    for action in VALID_ACTIONS:
        child_state = copy.deepcopy(node.state)
        child_state.apply_action(action)

        child_node = Node(child_state)
        node.children.append(child_node)

        build_tree(child_node)


def print_tree(root_node):
    nodes_to_traverse = [root_node]
    result = ""

    while len(nodes_to_traverse) > 0:
        current_node = nodes_to_traverse.pop(0)
        result += f"{current_node}\n\n"
        nodes_to_traverse += current_node.children

    print("final tree (hand order is (J, Q, K))")
    print("------------------------------------\n")
    print(result)
