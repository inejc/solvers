import numpy as np

# Kuhn poker game definition is equivalent to the one described in
# https://www.ma.imperial.ac.uk/~dturaev/neller-lanctot.pdf (section 3.1).

N_PLAYERS = 2
N_ACTIONS = 2
N_CARDS = 3  # hand order is always assumed to be (J, Q, K)


class Action:
    BET = "b"
    PASS = "p"


VALID_ACTIONS = (Action.BET, Action.PASS)


class State:
    def __init__(self):
        self.current_player = 0
        self.bets = np.array([1, 1])
        self.history = ""

    def apply_action(self, action):
        if action not in VALID_ACTIONS:
            raise ValueError("apply_action received an invalid action")

        if self.is_terminal():
            raise ValueError("can't call apply_action on a terminal state")

        if action == Action.BET:
            self.bets[self.current_player] += 1

        self.history += action
        self.current_player = 0 if self.current_player == 1 else 1

    def is_terminal(self):
        return self.history[-2:] == "bp" or \
            self.history[-2:] == "bb" or \
            self.history[-2:] == "pp"

    def is_showdown(self):
        # showdown happens only when both players bet or when both players pass
        if not self.is_terminal():
            raise ValueError("can't call is_showdown on a non-terminal state")
        return self.history[-2:] != "bp"
