import sys
import config
import numpy as np
import random
import math

# debug
def print_matrix(A):
    for i in range(A.shape[0]):
        print(list(np.round(A[i], 3)))

class UcbEntry:
    def __init__(self, l, a, node_id, ucb_method):
        self.id = node_id
        self.l = l  # l for region
        self.a = a  # a for arm 
        self.n = 0  # number times being pull, number of rewards

        self.samples = []

        self.times = []
        self.times_index = [] # when times is added
        self.shares = []
        self.score_list = []
        self.T_list = []
        self.max_time_list = []
        self.ucb_method = ucb_method

    def time_to_ucb_reward(self, t, max_time):
        re = (max_time - t) / max_time
        if re < 0:
            print('need larger config.time_constant')
            print(re, t, config.time_constant)
        return re
    def time_to_lcb_reward(self, t, max_time):
        re = t / max_time
        return re

    # T is used for debug, indicating when reward is given
    def update(self, t, share, T):
        sample = (t, T, share)
        self.samples.append(sample)

    def notify_pulled(self, score, T, max_time):
        self.score_list.append(score)
        self.T_list.append(T)
        self.max_time_list.append(max_time)

    def get_bound(self, T, max_time):
        if self.ucb_method == 'ucb':
            return self.get_upper_bound(T, max_time)
        elif self.ucb_method == 'lcb':
            return self.get_lower_bound(T, max_time)
        else:
            print('Error. Unknown ucb method')
            sys.exit(1)

    def get_lower_bound(self, T, max_time):
        n = len(self.samples)
        if n == 0:
            return -1.0 * config.time_constant

        # get means
        sum_reward = 0.0
        sum_share = 0.0
        for sample in self.samples:
            t, T, s = sample
            sum_reward += self.time_to_lcb_reward(t, max_time) * s
            sum_share += s

        empirical_mean = sum_reward/sum_share
        lower_bound = empirical_mean - math.sqrt(config.alpha * math.log(T) / 2.0 / n)
        return lower_bound

    def get_upper_bound(self, T, max_time):
        n = len(self.samples)
        if n == 0:
            return config.time_constant

        # get means
        sum_reward = 0.0
        sum_share = 0.0
        for sample in self.samples:
            t, T, s = sample
            sum_reward += self.time_to_ucb_reward(t, max_time) * s
            sum_share += s
        empirical_mean = sum_reward/sum_share 

        upper_bound = empirical_mean + math.sqrt(config.alpha * math.log(T) / 2.0 / n)
        return upper_bound

class Bandit:
    def __init__(self, node_id, num_region, num_node):
        self.ucb_table = {}
        self.id = node_id
        self.num_region = num_region
        self.num_node = num_node 
        self.alpha = config.alpha 
        self.is_init = False # is false, if we don't record randomization data
        self.max_time = 0
        self.T = 0    # this T is used for ucb, count number of conn changes
        self.num_msgs = [0 for _ in range(num_region)] # key is region, value is number recv msg

        self.special_num = config.time_constant
        if config.ucb_method == 'lcb':
            self.special_num = -1*config.time_constant

        for i in range(num_region):
            for j in range(num_node):
                if j != self.id:
                    # node cannot pull itself
                    self.ucb_table[(i,j)] = UcbEntry(i, j, node_id, config.ucb_method)

    def log_table(self, logger, epoch, origins):
        logger.write_str('>'+str(epoch)+'.ucb.' + str(origins))
        for pair, entry in self.ucb_table.items():
            l, i = pair
            if len(entry.samples) > 0:
                logger.write_ucb(l, i, entry.samples)

    def log_scores(self, logger, epoch, origins):
        score_table = np.zeros((self.num_region, self.num_node))
        for l in range(self.num_region):
            for i in range(self.num_node):
                score = self.special_num
                if i != self.id:
                    score = self.ucb_table[(l,i)].get_bound(self.T, self.max_time)
                score_table[l, i] = score

        comment = '>'+str(epoch)+'.ucb time:'+str(self.T)+'.max:'+str(int(self.max_time))+'.origins:'+str(origins)+'.ucb scores'
        logger.write_float_mat(score_table, comment, self.special_num)

    def soft_update(self, times, shares):
        assert(len(shares) == self.num_region)
        for l in range(len(shares)):
            share = shares[l]
            self.num_msgs[l] += share 
            if share > 0:
                for i in range(len(times)):
                    t = times[i]
                    if i != self.id and t != 0:
                        # selected that arm
                        self.ucb_table[(l,i)].update(t, share, self.T)

    def hard_update(self, times, shares):
        origin = np.argmax(shares)
        self.num_msgs[origin] += 1
        for i in range(len(times)):
            t = times[i]
            if i != self.id and t != 0:
                self.ucb_table[(origin, i)].update(t, 1, self.T)

    def get_ucb_score(self, l, j):
        if config.is_ucb:
            return self.ucb_table[(l,j)].get_upper_bound(self.T, self.max_time)
        else:
            return self.ucb_table[(l,j)].get_lower_bound(self.T, self.max_time)

    # one update per msg
    def update_one_msg(self, times, shares):
        if config.hard_update:
            self.hard_update(times, shares)
        else:
            self.soft_update(times, shares) 

    # always collect rewards from the last time
    def update_ucb_table(self, W, X, num_msg, max_time):
        self.T += 1
        if self.max_time < max_time:
            self.max_time = max_time
        shares, observation = None, None
        num_row = W.shape[0]

        if self.is_init:
            for i in range(W.shape[0]):
                obs = X[i]
                shares = W[i]
                self.update_one_msg(obs, shares)
            self.is_init = False
        else:
            # print(self.id, 'update', num_msg)
            seen = set()
            for i in range(num_row - num_msg, num_row):
                shares = W[i] # which sums to 1
                # TODO simple trick not letting two updates in one config
                # if config.hard_update:
                    # origin = np.argmax(shares)
                    # if origin in seen:
                        # continue
                # seen.add(origin)

                observation = X[i]
                self.update_one_msg(observation, shares)
        return np.argmax(W, axis=1)

    # masks if some arms cannot be pulled
    def pull_arms(self, valid_arms, required_arm):
        arms = []
        while len(arms) < required_arm:
            for l in range(self.num_region):
                scores = []
                for j in valid_arms:
                    assert(j != self.id)
                    score = self.ucb_table[(l,j)].get_bound(self.T, self.max_time)
                    scores.append(score)

                selected_arm, best_score = self.choose_best_region_arm(valid_arms, scores)
                
                selected_ucb_entry = self.ucb_table[(l, selected_arm)]
                selected_ucb_entry.notify_pulled(best_score, self.T, self.max_time)

                arms.append((l, selected_arm))
                valid_arms.remove(selected_arm)
        return arms 

    def choose_best_region_arm(self, valid_arms, scores):
        assert(len(valid_arms)==len(scores))
        num_arms = len(valid_arms)
        best_arms = []
        best_score = None
        if config.ucb_method == 'ucb':
            best_score = max(scores)
            for k, score in enumerate(scores):
                if score == best_score:
                    best_arms.append(valid_arms[k])
        elif config.ucb_method == 'lcb':
            best_score = min(scores)
            for k, score in enumerate(scores):
                if score == best_score:
                    best_arms.append(valid_arms[k])
        else:
            print('Error. invalid ucb method')
            sys.exit(1)

        random.shuffle(best_arms)
        return best_arms[0], best_score

    # pure exploration
    def argmin_arms(self, H):
        argsorted_peers = np.argsort(H, axis=1)
        return argsorted_peers

    # return num arms not pulled for each region
    def get_num_not_pulled(self):
        npull = [0 for i in range(self.num_region)] 
        for l in range(self.num_region):
            for j in range(self.num_node):
                if self.id != j:
                    if self.ucb_table[(l,j)].n == 0:
                        npull[l] += 1
        return npull 

    def get_pulled_arms(self):
        arm_list = []
        for i in range(self.num_region):
            for j in range(self.num_node):
                if self.id != j:
                    if self.ucb_table[(i,j)].n > 0:
                        arm_list.append((i,j))
        return arm_list

    def greedy(self):
        pass

    def epsilon_greedy(self):
        pass
