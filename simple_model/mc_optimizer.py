import numpy as np
import torch
import sys
import time
import math
from simple_model.loss_functions import ClusterLoss
from simple_model.loss_functions import ElementLoss
from collections import defaultdict

class Indicator:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        # 0: easy, 1: ambiguous/ambiguous 2: unable
        self.indicators = np.zeros(4)
        self.peering_rows = {}
        self.text = ['ea', 'am', 'un'] 

    def set_easy(self):
        self.indicators[0] = 1
    def set_ambiguous(self):
        self.indicators[1] = 1
    # def set_plus(self):
        # self.indicators[1] = 1
    def set_unable(self):
        self.indicators[2] = 1
    def add_peering_row(self, k, comm):
        self.peering_rows[k] = comm
    def get(self):
        if np.sum(self.indicators) > 1:
            print('element',i,j,'multiple indicator', self.indicators)
            sys.exit(1)
        elif np.sum(self.indicators) == 0:
            print('element',i,j,'no indicator', self.indicators)
            sys.exit(1)
        else:
            for i in range(3):
                if self.indicators[i] != 0:
                    return i
    def get_text(self):
        if np.sum(self.indicators) > 1:
            print('element',i,j,'multiple indicator', self.indicators)
            sys.exit(1)
        elif np.sum(self.indicators) == 0:
            print('element',i,j,'no indicator', self.indicators)
            sys.exit(1)
        else:
            for i in range(3):
                if self.indicators[i] != 0:
                    return self.text[i]

class McOptimizer:
    def __init__(self, i, epochs, lr, top_n_peer):
        self.id = i
        self.epochs = epochs
        self.lr = lr 
        self.top_n_peer = top_n_peer 

    # data in numpy
    def col_mean_init(self, X_in, M_in, nM_in):
        T, N = X_in.shape
        col_means = []
        X = X_in.copy()
        for i in range(N):
            col_sum = sum(X[:,i]*M_in[:,i])
            num_sum = sum(M_in[:,i])
            m = col_sum / num_sum
            X[:,i] = X[:,i]*M_in[:,i] + m*(1-M_in[:,i])
            col_means.append(m)
        return X 

    def run(self, X_in, M_in, nM_in, max_time):
        return self.run_element_completion(X_in, M_in, nM_in, max_time)
        # completed_table,C = self.run_cluster_loss(X_in, M_in, nM_in, max_time)
        # completed_table = nM_in*completed_table + (1-nM_in)*9999
        # return completed_table, C

    # as long as there is a common 'num' elements, regardless if interested col is + or numeric
    def has_common_known_by_row(self,ind, known_pos_by_col, known_pos_by_row, known_plus_pos_by_col, num):
        i_row_known = set(known_pos_by_row[ind.i])
        has_peering_row = False
        assert(len(set(known_pos_by_col[ind.j]).intersection(known_plus_pos_by_col[ind.j])) == 0)
        for k in known_pos_by_col[ind.j]+known_plus_pos_by_col[ind.j]:
            if ind.i != k:
                k_row_known = known_pos_by_row[k]
                common_elements = [t for t in k_row_known if t in i_row_known]
                if len(common_elements) >= num:
                    ind.add_peering_row(k, common_elements)
                    has_peering_row = True
        return has_peering_row

    def run_element_completion(self, X_in, M_in, nM_in, max_time):
        X_in = X_in * M_in
        X_in = X_in / max_time
        
        T, N = X_in.shape
        unknown_pos_by_row = defaultdict(list)
        known_pos_by_row = defaultdict(list)
        known_plus_pos_by_row = defaultdict(list)

        unknown_pos_by_col = defaultdict(list)
        known_pos_by_col = defaultdict(list)
        known_plus_pos_by_col = defaultdict(list)

        num_known_numeric = 0
        num_known_plus = 0

        for i in range(T):
            for j in range(N):
                if M_in[i,j]==0 and nM_in[i,j]==1:
                    unknown_pos_by_row[i].append(j)
                    unknown_pos_by_col[j].append(i)
                elif M_in[i,j]==0 and nM_in[i,j]==0:
                    known_plus_pos_by_row[i].append(j)
                    known_plus_pos_by_col[j].append(i)
                    num_known_plus += 1
                elif M_in[i,j]==1:
                    known_pos_by_row[i].append(j)
                    known_pos_by_col[j].append(i)
                    num_known_numeric += 1
                else:
                    print('Error. Unknown i,j classification')
                    sys.exit(1)
        # easy estimatible, i.e. does it has known value or '+' in other rows
        easy_estimate_by_row = defaultdict(list)
        ambi_estimate_by_row = defaultdict(list)
        un_estimate_by_row = defaultdict(list)

        easy_estimates = []
        ambi_estimates = []
        un_estimates = []

        indicators = {}
        for i in range(T):
            for j in unknown_pos_by_row[i]:
                indicator = Indicator(i, j)
                # something is just impossible
                if len(known_pos_by_col[j])>0 or len(known_plus_pos_by_col[j])>0:
                    if len(known_pos_by_row[i]) <= 1:
                        # if has no common element, or at most 1 common -> impossible
                        indicator.set_unable()
                    elif not self.has_common_known_by_row(indicator, known_pos_by_col,known_pos_by_row, known_plus_pos_by_col, 2):
                        indicator.set_unable()
                    else:
                        # the rest is estimatible
                        if len(known_plus_pos_by_col[j])==0: # other >0
                            # the case is easy, when all peering row only contains values, not +
                            indicator.set_easy()
                        elif len(known_pos_by_col[j])-1 < self.top_n_peer: # when selecting top_n_peer, it must have selected + if there is not sufficient
                            indicator.set_ambiguous()
                        else:                             # both 
                            indicator.set_ambiguous()
                else:
                    print('Error. all other columns are unknown', i, j)
                    sys.exit(1)

                indicators[(i,j)] = indicator
                # easy to estimate
                e_class = indicator.get()
                if e_class == 0:
                    easy_estimate_by_row[i].append(j)
                    easy_estimates.append((i,j))
                elif e_class == 1:
                    ambi_estimate_by_row[i].append(j)
                    ambi_estimates.append((i,j))
                elif e_class == 2:
                    un_estimate_by_row[i].append(j)
                    un_estimates.append((i,j))
                else:
                    print('Error. Unknown class', e_class)
                    sys.exit(1)
        print('estimating easy', len(easy_estimates))
        print('estimating ambi', len(ambi_estimates))
        print('estimating unab', len(un_estimates))

        # Start construct pytorch 
        X = torch.tensor(X_in, dtype=torch.float32) 
        M = torch.tensor(M_in, dtype=torch.float32)
        nM = torch.tensor(nM_in, dtype=torch.float32)

        
        tensor_ind_map = {}
        # compute row-wise score in tensor
        element_peer_scores = defaultdict(list) # key is ij, value is list of (weight, k-row) 
        softmax_func = torch.nn.Softmax(dim=0)

        plus_estimates = []
        plus_estimate_by_row = defaultdict(list)
        for i,j in ambi_estimates:
            ind = indicators[(i,j)]
            num_peer = len(ind.peering_rows)
            scores = np.ones(T) * float("inf")
            for k in sorted(ind.peering_rows):
                common_elements = ind.peering_rows[k]
                sel_diff = X_in[i][common_elements] - X_in[k][common_elements]
                scores[k] = np.var(sel_diff, ddof=1)

            contain_plus = False
            sorted_ind = np.argsort(scores)
            for k in sorted_ind[:self.top_n_peer]:
                if j in known_plus_pos_by_row[k]:
                    contain_plus = True
                    # print(i, j, 'detect ambiguous in tops ', k, sorted_ind)
                    break
            if not contain_plus:
                low_peer = sorted_ind[self.top_n_peer-1]
                low_selected_score = scores[low_peer]
                for k in sorted_ind[self.top_n_peer:]:
                    if scores[k] == low_selected_score:
                        if j in known_plus_pos_by_row[k]:
                            # print(i, j, 'detect ambiguous by looking extension', k, sorted_ind)
                            contain_plus = True
                            break
                    else:
                        break

            if not contain_plus:
                ind.set_easy()
                easy_estimate_by_row[i].append(j)
                easy_estimates.append((i,j))
                # print('reset ambi',i,j,'to easy')
            else:
                plus_estimate_by_row[i].append(j)
                plus_estimates.append((i,j))


        t = 0
        for i,j in easy_estimates:
            tensor_ind_map[(i,j)] = t
            t += 1
            ind = indicators[(i,j)]
            num_peer = len(ind.peering_rows)
            scores = torch.ones(T) * float("inf")
            for k in sorted(ind.peering_rows):
                common_elements = ind.peering_rows[k]
                sel_diff = X[i][common_elements] - X[k][common_elements]
                scores[k] = torch.var(sel_diff, unbiased=True)
            randomized_scores = scores + torch.rand(T, dtype=float) * 0.00001
            # print(i,j,k, scores, sel_diff)
            num_peer = min(self.top_n_peer, num_peer)
            topk, topk_ind = torch.topk(randomized_scores, num_peer, largest=False)
            weight = softmax_func(-1*topk)

            for c in range(len(topk_ind)):
                # row in py number, not in tensor
                k = topk_ind[c].item()
                select_mask = torch.zeros(N, dtype=int)
                select_mask[j] = 1
                select_mask[ind.peering_rows[k]] = 1
                element_peer_scores[(i,j)].append((weight[c], k, select_mask.gt(0)))

        total_cell_num = num_known_numeric+num_known_plus+len(easy_estimates)+len(plus_estimates)+len(un_estimates)
        print('Table of size', T, N)
        print('total num cell', total_cell_num)
        print('known numeric', num_known_numeric)
        print('known plus', num_known_plus)
        print('estimating unknown easy', len(easy_estimates))
        print('estimating unknown plus', len(plus_estimates))
        print('estimating unknown unab', len(un_estimates))
        assert(total_cell_num==T*N )


        X = X*nM 

        # for i in range(N):
            # print(np.round(X_in[i]*max_time, 0))

        num_var = len(easy_estimates)
        A = torch.rand(num_var, dtype=torch.float32) 
        C = torch.rand(T, requires_grad=True, dtype=torch.float32) 
        A.requires_grad_()

        relu = torch.nn.ReLU()
        criterion = ElementLoss()
        s_time = time.time()
        for e in range(self.epochs):
            loss = criterion(X,A,C, easy_estimate_by_row, known_pos_by_row,element_peer_scores, tensor_ind_map, T)
            loss.backward()
            with torch.no_grad():
                num_loss = loss.item()
                if e % 100 ==0:
                    print(e, 'normalized loss', num_loss, 'in', round(time.time()-s_time,2))
                    s_time = time.time()

                if C.isnan().any() or A.isnan().any():
                    print(e, 'Loss explodes before relu', A.grad, C.grad)
                    sys.exit(1)

                A = relu(A - self.lr * A.grad)
                C = relu(C - self.lr * C.grad)
                
                A.grad = None    
                C.grad = None
                A.requires_grad_()
                C.requires_grad_()

        C_out = C.detach().numpy()*max_time       
        C_out = C_out.reshape((len(C_out), 1))
        H_out = (X.numpy() *max_time + C_out)*nM_in + (1-nM_in)*9999 

        unkn_plus_mask = np.zeros((T,N))
        unkn_unab_mask = np.zeros((T,N))

        completed_A = A.detach().numpy()
        for i,j in easy_estimates:
            a = completed_A[tensor_ind_map[(i,j)]] * max_time 
            H_out[i,j] = a 

        for i,j in plus_estimates:
            unkn_plus_mask[i,j] = 1
            H_out[i,j] = 7777

        for i,j in un_estimates:
            H_out[i,j] = 5555 
            unkn_unab_mask[i,j] = 1

        return H_out, C_out, unkn_plus_mask, unkn_unab_mask

    def run_cluster_loss(self, X_in, M_in, nM_in, max_time):
        # X_in = self.col_mean_init(X_in, M_in, nM_in)
        X_in = X_in/max_time

        X = torch.tensor(X_in, dtype=torch.float32) 
        M = torch.tensor(M_in, dtype=torch.float32)
        nM = torch.tensor(nM_in, dtype=torch.float32)

        T, N = X_in.shape

        H = torch.rand(T, N, dtype=torch.float32) 
        C = torch.rand(T, 1, requires_grad=True, dtype=torch.float32) 
        H.requires_grad_()

        row_scores = {}
        completion_rows = set()
        for i in range(T):
            scores = torch.ones(T) * float("inf")
            for j in range(T):
                if i != j:
                    #            1 at miss    1 at non- None value
                    i_missings = ((1-M[i]) * nM[i]).gt(0)
                    j_values = (M[j] * nM[j]).gt(0)
                    if (torch.masked_select(j_values, i_missings)).any():
                        selected = torch.masked_select(X[i]-X[j], (M[i]*M[j]) >0)
                        if torch.numel(selected) > 1: 
                            scores[j] = torch.var(selected, unbiased=True)
                            completion_rows.add(i)
                        # else:
                            # print('pair', i, j)
                            # print(i_missings, j_values)
                            # print()
            row_scores[i] = scores

        completion_rows = sorted(list(completion_rows))
        # for i in mc_rows:
            # print(i, row_scores[i])

        criterion = ClusterLoss()
        prox_plus = torch.nn.Threshold(0,0)
        s_time = time.time()
        for e in range(self.epochs):
            loss = criterion(X, H, C, M, T, nM, row_scores, completion_rows)
            loss.backward()

            with torch.no_grad():
                s = (X - (H - C) )*M
                if e % 100 ==0:
                    print(e, 'normalized loss', torch.norm(s), 'in', round(time.time()-s_time,2))
                    s_time = time.time()
                if torch.norm(s).isnan().any():
                    print('Loss explodes', H.grad, C.grad)
                    sys.exit(1)


                H = prox_plus(H - self.lr * H.grad)
                C = prox_plus(C - self.lr * C.grad)

                H.requires_grad_()
                C.requires_grad_()

                H.grad = None
                C.grad = None

        H_com = H.detach().numpy()*max_time
        for i in range(T):
            if i not in completion_rows:
                H_com[i] = H_com[i]*M_in[i] + (1-M_in[i])*9999
        # for i in range(T):
            # print(list(H_com[i]))
        # sys.exit(1)
        return H_com, C.detach().numpy()*max_time
                
        # row_elements_pos = {}
        # for i in range(T):
            # pos_list = []
            # for j in range(N):
                # if i != j:
                    # i_missings = ((1-M[i]) * nM[i]).gt(0)
                    # j_values = (M[j] * nM[j]).gt(0)
                    # if (torch.masked_select(j_values, i_missings)).any():
                        # selected = torch.masked_select(X[i]-X[j], (M[i]*M[j]) >0)
                        # if torch.numel(selected) > 1:
                            # pos_list.append(j)
# debug
        # print('known_pos_by_row', known_pos_by_row)
        # for i in range(T):
            # line = ''
            # for j in range(N):
                # if j in unknown_pos_by_row[i]:
                    # ind = indicators[(i,j)]
                    # line += "{m:<3}".format(m=ind.get_text())
                    # # print("{i:<3} {j:<3}".format(i=i, j=j), ind.get_text(), ind.peering_rows)
                # elif j in known_plus_pos_by_row[i]:
                    # line += "{m:<3}".format(m='pl')
                # elif j in known_pos_by_row[i]:
                    # line += "{m:<3}".format(m='kn')
                # else:
                    # print('Error. Unknown class')
                    # sys.exit(1)
            # print(line)
        # sys.exit(1)
