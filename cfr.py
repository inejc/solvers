import numpy as np

from game_tree import build_tree, print_tree, calculate_exploitability
from kuhn_poker import N_CARDS, N_PLAYERS


def main():
    num_cfr_iterations = 10_000
    root_node = build_tree()

    for _ in range(num_cfr_iterations):
        for player in range(N_PLAYERS):
            cfr(root_node, learning_player=player)

    print_tree(root_node)

    exploitability = calculate_exploitability(root_node)
    print(f"exploitability for player 0: {exploitability[0]:.4f}")
    print(f"exploitability for player 1: {exploitability[1]:.4f}")


def cfr(node, learning_player):
    if node.state.is_terminal():
        return node.evaluate_terminal_state(learning_player)

    node_cfvs = np.zeros(N_CARDS)
    children_cfvs = []

    for action_i, child in enumerate(node.children):
        # action_prob is probability of the child node
        # action for each hand, i.e. shape = (N_CARDS,)
        action_prob = node.current_strategy[action_i]

        child.update_beliefs(
            action_prob=action_prob,
            parent_beliefs=node.beliefs,
        )

        child_cfvs = cfr(child, learning_player)
        children_cfvs.append(child_cfvs)

        if node.state.current_player == learning_player:
            node_cfvs += child_cfvs * action_prob
        else:
            node_cfvs += child_cfvs

    if node.state.current_player == learning_player:
        node.cumulative_regrets += np.array(children_cfvs) - node_cfvs
        node.regret_matching()

    return node_cfvs


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    main()
