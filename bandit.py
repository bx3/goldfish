import sys

class Bandit:
    def __init__(self, node_id, out_lim, num_arm):
        self.H = None
        self.id = node_id,
        self.num_pull = out_lim
        self.num_arm = num_arm

    def update_rewards(self, H):
        if self.H == None:
            self.H = H
        else:
            # update ucb H
            pass

    def pull_arm(self):
        peers = []
        # use H to select arm 

        return peers
