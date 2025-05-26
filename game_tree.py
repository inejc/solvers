import copy

import numpy as np

from kuhn_poker import N_PLAYERS, N_ACTIONS, N_CARDS, Action, State


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
            self.beliefs[self.state.current_player] * new_strategy

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
                    self.beliefs[opponent][opponent_hand] * payoff

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

    for action in (Action.BET, Action.PASS):
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


def calculate_exploitability(root_node):
    result = []

    for player in range(N_PLAYERS):
        root_node_copy = copy.deepcopy(root_node)
        result.append(_calculate_exploitability(root_node_copy, player))

    return np.array(result)


def _calculate_exploitability(root_node, player):
    _set_average_strategy_as_current_strategy(root_node)
    average_strategy_ev = _calculate_cfvs(root_node, player)

    _set_best_response_strategy_as_current_strategy(root_node, player)
    pure_best_response_strategy_ev = _calculate_cfvs(root_node, player)

    return np.sum(pure_best_response_strategy_ev) - np.sum(average_strategy_ev)


def _set_average_strategy_as_current_strategy(node):
    if node.state.is_terminal():
        return

    node.current_strategy = node.get_average_strategy()
    for child in node.children:
        _set_average_strategy_as_current_strategy(child)


def _calculate_cfvs(node, player):
    # recursively calculates counterfactual values of all nodes, additionally
    # stores the result as a .cfvs attribute on each node object
    if node.state.is_terminal():
        node.cfvs = node.evaluate_terminal_state(player)
        return node.cfvs

    node.cfvs = np.zeros(N_CARDS)

    for action_i, child in enumerate(node.children):
        # action_prob is probability of the child node
        # action for each hand, i.e. shape = (N_CARDS,)
        action_prob = node.current_strategy[action_i]

        child.update_beliefs(
            action_prob=action_prob,
            parent_beliefs=node.beliefs,
        )

        child_cfvs = _calculate_cfvs(child, player)

        if node.state.current_player == player:
            node.cfvs += child_cfvs * action_prob
        else:
            node.cfvs += child_cfvs

    return node.cfvs


def _set_best_response_strategy_as_current_strategy(node, player):
    # sets the current strategy of a player to be the pure best response
    # strategy, this function expects .cfvs attribute to be defined
    # on each node object
    if node.state.is_terminal():
        return

    if node.state.current_player == player:
        children_cfvs = np.array([child.cfvs for child in node.children])
        best_action_i = np.argmax(children_cfvs, axis=0)
        node.current_strategy[best_action_i, :] = 1.0
        node.current_strategy[~best_action_i, :] = 0.0

    for child in node.children:
        _set_best_response_strategy_as_current_strategy(child, player)
