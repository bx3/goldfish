import sys
import config
import numpy as np
import random
import math

class UcbEntry:
    def __init__(self, l, a, node_id):
        self.id = node_id
        self.l = l  # l for region
        self.a = a  # a for arm 
        self.n = 0  # number times being pull
        self.times = []
        self.shares = []

    def time_to_reward(self, t):
        re = t / config.time_constant
        if re < 0:
            print('need larger config.time_constant')
            print(re, t, config.time_constant)
        return re

    def update(self, t, share):
        self.n += 1
        self.times.append(t)
        self.shares.append(share)

    def get_lower_bound(self):
        if self.n == 0:
            # print('node', self.id, 'has not pulled region', self.l, 'arm', self.a)
            return -1.0 * config.time_constant

        # get means
        sum_reward = 0.0
        for i in range(len(self.times)):
            t = self.times[i]
            s = self.shares[i]
            sum_reward += self.time_to_reward(t) * s
        if sum(self.shares) == 0:
            print('share', self.shares)
            print('n', self.n)
        empirical_mean = 1.0/sum(self.shares) * sum_reward
        lower_bound = empirical_mean - math.sqrt(config.alpha * math.log(t) / 2.0 / self.n)
        return lower_bound

    def get_upper_bound(self):
        if self.n == 0:
            # print('node', self.id, 'has not pulled region', self.l, 'arm', self.a)
            return  config.time_constant

        # get means
        sum_reward = 0
        for i in range(len(self.times)):
            t = self.times[i]
            s = self.shares[i]
            sum_reward += self.time_to_reward(t) * s
        if sum(self.shares) == 0:
            print('share', self.shares)
            print('n', self.n)
        empirical_mean = 1.0/sum(self.shares) * sum_reward

        # upper bound
        upper_bound = empirical_mean + math.sqrt(config.alpha * math.log(t) / 2.0 / self.n)
        return upper_bound

class Bandit:
    def __init__(self, node_id, num_region, num_node):
        self.ucb_table = {}
        self.id = node_id
        self.num_region = num_region
        self.num_node = num_node 
        self.alpha = config.alpha 
        self.is_init = True 
        for i in range(num_region):
            for j in range(num_node):
                if j != self.id:
                    # node cannot pull itself
                    self.ucb_table[(i,j)] = UcbEntry(i, j, node_id)
        

    def soft_update(self, times, shares):
        assert(len(shares) == self.num_region)
        for l in range(len(shares)):
            share = shares[l]
            if share > 0:
                for i in range(len(times)):
                    t = times[i]
                    if i != self.id and t != 0:
                        # selected that arm
                        self.ucb_table[(l, i)].update(t, share)

    def hard_update(self, times, shares):
        origin = np.argmax(shares)
        # print('shares', shares)
        # print('times', times)
        # print('origin', origin)
        # rewards = time_to_reward(last_observation)
        for i in range(len(times)):
            t = times[i]
            if i != self.id and t != 0:
                # selected that arm
                self.ucb_table[(origin, i)].update(t, 1)

    def update(self, times, shares):
        if config.hard_update:
            self.hard_update(times, shares)
        else:
            self.soft_update(times, shares) 

    def init_ucb_table(self, W, X):
        num_msg = W.shape[0]
        for i in range(num_msg):
            obs = X[i]
            shares = W[i]
            self.update(obs, shares)

    # always collect rewards from the last time
    def update_times(self, W, X):
        shares, observation = None, None
        if self.is_init:
            self.init_ucb_table(W, X)
            self.is_init = False

        else:
            shares = W[-1] # which sums to 1
            observation = X[-1]
            self.update(observation, shares)


    # masks if some arms cannot be pulled
    def pull_arms(self, valid_arms):
        arms = []

        for i in range(self.num_region):
            best_arm, best_score = [], None
            scores = []
            for j in valid_arms:
                if j != self.id:
                    score = self.ucb_table[(i,j)].get_lower_bound()
                    scores.append(score)
                    if best_score == None or score < best_score:
                        best_score = score
                        best_arm = [j]
                    elif score == best_score:
                        best_arm.append(j)

            # print(best_score, best_arm, len(best_arm))
            # print(scores)
            # randomize equally good
            random.shuffle(best_arm)
            selected_arm = best_arm[0]
            # print(scores)

            arms.append(selected_arm)
            valid_arms.remove(selected_arm)
        return arms 

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
