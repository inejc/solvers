import copy

import numpy as np

from kuhn_poker import N_PLAYERS, N_ACTIONS, N_CARDS


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
            node.cfvs += (child_cfvs * action_prob)
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
        node.current_strategy = best_response_strategy(children_cfvs)

    for child in node.children:
        _set_best_response_strategy_as_current_strategy(child, player)


def best_response_strategy(actions_cfvs):
    # returns a pure best response strategy based on actions' cfvs,
    # actions_cfvs is expected to be of shape (N_ACTIONS, N_CARDS)
    row_i = np.argmax(actions_cfvs, axis=0)
    col_i = np.arange(row_i.shape[0])

    best_response = np.zeros((N_ACTIONS, N_CARDS))
    best_response[row_i, col_i] = 1.0
    return best_response
