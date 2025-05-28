import numpy as np

from game_tree import build_tree, print_tree
from kuhn_poker import N_CARDS, N_PLAYERS
from metrics import calculate_exploitability, best_response_strategy


def main():
    num_fp_iterations = 50_000
    print(f"running an FP solver for {num_fp_iterations} iterations\n")

    root_node = build_tree(cumulative_strategy_initial_value=1.0)

    for iteration_i in range(num_fp_iterations):
        for player in range(N_PLAYERS):
            fp(root_node, learning_player=player)

        if iteration_i % 100 == 0 or iteration_i == num_fp_iterations - 1:
            exploitability = calculate_exploitability(root_node)
            print(f"iteration {iteration_i}:")
            print(f"    exploitability for player 0: {exploitability[0]:.4f}")
            print(f"    exploitability for player 1: {exploitability[1]:.4f}")
            print()

    print()
    print_tree(root_node)


def fp(node, learning_player):
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

        child_cfvs = fp(child, learning_player)
        children_cfvs.append(child_cfvs)

        if node.state.current_player == learning_player:
            node_cfvs += (child_cfvs * action_prob)
        else:
            node_cfvs += child_cfvs

    if node.state.current_player == learning_player:
        node.cumulative_strategy += best_response_strategy(
            np.array(children_cfvs),
        )
        node.current_strategy = node.get_average_strategy()

    return node_cfvs


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    main()
