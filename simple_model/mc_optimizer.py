import numpy as np
import torch
import sys
import time
from simple_model.loss_functions import ClusterLoss
from simple_model.loss_functions import ElementLoss

class McOptimizer:
    def __init__(self, i, epochs, lr):
        self.id = i
        self.epochs = epochs
        self.lr = lr 

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
        completed_table,C = self.run_cluster_loss(X_in, M_in, nM_in, max_time)
        completed_table = nM_in*completed_table + (1-nM_in)*9999
        return completed_table, C
        # return self.run_element_completion(X_in, M_in, nM_in, max_time)

    def run_element_completion(self, X_in, M_in, nM_in, max_time):
        X_in = X_in * M_in
        # X_in = self.col_mean_init(X_in, M_in, nM_in)
        X_in = X_in / max_time

        X = torch.tensor(X_in, dtype=torch.float32) 
        M = torch.tensor(M_in, dtype=torch.float32)
        nM = torch.tensor(nM_in, dtype=torch.float32)
        
        T, N = X_in.shape
        num_var = int(np.sum(1-M_in) - np.sum(1-nM_in))
        # print('num_var', num_var)
        # print(M_in)
        # print(nM_in)
        row_elements_pos = {}
        for i in range(T):
            pos_list = [j for j in range(N) if (M_in[i,j]==0 and nM_in[i,j]==1)]
            row_elements_pos[i] = torch.tensor(pos_list)
        print(row_elements_pos)

        A = torch.rand(num_var, dtype=torch.float32) 
        A.requires_grad_()

        criterion = ElementLoss()
        for e in range(self.epochs):
            loss = criterion(X, A, M, T, nM, row_elements_pos, max_time)
            loss.backward()
            with torch.no_grad():
                if e % 100 ==0:
                    print(e)
                A.grad = None        
                A.requires_grad_()
        
        print('before detach') 
       
        elements = A.detach().numpy()
        print(elements) 
        k = 0
        for i in range(T):
            for j in row_elements_pos[i]:
                X_in[i, j] = elements[k]
                k += 1

        return X_in * max_time

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
                

