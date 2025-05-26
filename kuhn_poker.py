import numpy as np

# Kuhn poker game definition is equivalent to the one described in
# https://www.ma.imperial.ac.uk/~dturaev/neller-lanctot.pdf (section 3.1).

N_PLAYERS = 2
N_ACTIONS = 2
# hand order is (J, Q, K)
N_CARDS = 3


class Action:
    BET = "b"
    PASS = "p"


class State:
    def __init__(self):
        self.current_player = 0
        self.bets = np.array([1, 1])
        self.history = ""

    def apply_action(self, action):
        if action == Action.BET:
            self.bets[self.current_player] += 1
        self.history += action
        self.current_player = 0 if self.current_player == 1 else 1

    def is_terminal(self):
        return self.history[-2:] == "bp" or \
            self.history[-2:] == "bb" or \
            self.history[-2:] == "pp"
