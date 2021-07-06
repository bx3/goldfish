import torch
import numpy as np
import sys
import time
from collections import defaultdict

epochs = 200
row_penalty = 1
lr = 0.05
NoneTime=9999

class CompletionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_fn = torch.nn.MSELoss(reduction='sum')

    # def forward(self, X, H, C, M, T, non_zero_row, argmins):
        # row_loss = torch.tensor(0.0)
        # softmax_func = torch.nn.Softmax(dim=0)
        # for i in range(T):
            # curr_loss = torch.tensor(0.0)
            # scores = torch.ones(T) * float(99999)
            # non_zero = non_zero_row[i]
            # for j in range(T):
                # if i != j:
                    # if ((M[i] != M[j]).any() and 
                          # (torch.argsort(H[i])[:non_zero] == argmins[i][:non_zero]).all()) :
                        # selected = torch.masked_select(H[i]-H[j], (M[i]*M[j]) >0)
                        # scores[j] = torch.var(selected)
            # min_row = torch.argmin(scores)
            # curr_loss += torch.norm(H[i] - H[min_row])
            
            # # print('nonzero', non_zero)
            # # print('H[i]', H[i])
            # # print('torch.argsort(H[i])', torch.argsort(H[i]))
            # # print('argmins[i]', argmins[i])
            # # print()
            # row_loss += curr_loss
        # loss = self.mse_fn(X*M, (H-C)*M) +  row_penalty*row_loss
        # return loss

    def forward(self, X, H, C, M, T):
        row_loss = torch.tensor(0.0) 
        softmax_func = torch.nn.Softmax(dim=0)
        for i in range(T):
            scores = torch.ones(T) * float(9999)
            for j in range(T):
                if i != j:
                    if (M[i] != M[j]).any():
                        selected = torch.masked_select(H[i]-H[j], (M[i]*M[j]).ge(0))
                        scores[j] = torch.sqrt(torch.var(selected)) # / 1000

            topk, topk_ind = torch.topk(scores, 8, largest=False)
            weight = softmax_func(-1*topk)

            # print('scores', scores)    
            # print(weight, topk)
            # print('---------------------')
            
            for k in range(len(topk_ind)):
                j = topk_ind[k]
                row_loss += weight[k]*torch.norm(H[i] - H[j])
        loss = self.mse_fn(X*M, (H-C)*M) +  row_penalty*row_loss  
        return loss

def complete_mat(X_numpy, M_numpy,max_time):
    T, N = X_numpy.shape
    X = torch.tensor(X_numpy)
    M = torch.tensor(M_numpy)
    penalties_record = []
    nonzero_row = np.sum(M_numpy, axis=1, dtype=int)
    modified = torch.tensor((1-M_numpy) * float(999999)) + X
    argmins = torch.argsort(modified, dim=1)

    H = torch.rand(T, N) * max_time # X.clone().requires_grad_(True)
    C = torch.rand(T, 1)
    H.requires_grad_()
    C.requires_grad_()

    criterion = CompletionLoss()
    prox_plus = torch.nn.Threshold(0,0)
    for e in range(epochs):
        loss = criterion(X, H, C, M, T)
        loss.backward()

        with torch.no_grad():
            penalties_record.append((torch.norm((X-(H-C) )*M)).item())

            H = prox_plus(H - lr * H.grad)
            C = prox_plus(C - lr * C.grad)

            H.requires_grad_()
            C.requires_grad_()

            H.grad = None
            C.grad = None   
    
    H_return = H.detach().numpy()
    C_return = C.detach().numpy()
    del H
    del C
    return H_return, C_return, penalties_record


# mask whose i,j entry is 1 if value is present, i.e. two conditions
# 1. has a value. 2. value is not None
# none mask whose i,j entry is 1 if the value is None
def construct_table(slots, node_id, log_directions):
    id_set = set()
    id_list = {}
    ids_both_direction = defaultdict(set)
    for slot in slots:
        # if node_id == 0: 
            # print(len(slots))
            # print(slot)
        for p, t, direction in slot:
            ids_both_direction[p].add(direction)
            if direction in log_directions:
                id_set.add(p)


    ids_with_direction = sorted(ids_both_direction.items())
    ids = sorted(list(id_set))
    id_map = {}

    N = len(ids)
    T = len(slots)
    
    for i in range(N):
        id_map[ids[i]] = i

    X = np.zeros((T, N)) 
    mask = np.zeros((T, N))
    none_mask = np.zeros((T, N))
    i = 0 # row
    max_time = 0 

    for slot in slots:
        # if node_id == 0: 
            # print('h', len(slots))
            # print('h', slot)
        for p, t, direction in slot:
            if direction in log_directions:
                if t is not None:
                    j = id_map[p]
                    X[i, j] = t
                    mask[i, j] = 1
                    max_time = max(t, max_time)
                # TODO think about if None be infinite 
                else:
                    j = id_map[p]
                    none_mask[i,j] = 1
                    
                    # X[i, j] = 9999
                    # mask[i, j] = 1

        i += 1
    return X, mask, none_mask, max_time, ids, ids_with_direction

def run(i, slots, directions):
    X, M, none_M, max_time, ids = construct_table(slots, i, directions)
    H_cpl, C_cpl, opt = complete_mat(X, M, max_time)
    return H_cpl, C_cpl, opt, i, ids



def print_mats(mats):
        lines = []
        num_mat = len(mats)
        num_line = len(mats[0])
        for i in range(num_line): 
            line = ''
            for mat in mats:
                line += mat[i] + '\t'
            print(line)

def format_mat(A, label_ticks, is_float):
    lines = []

    if label_ticks is not None:
        if not is_float:
            ticks = [str(a).rjust(4, '_') for a in label_ticks]
        else:
            ticks = [str(a).rjust(5, '_') for a in label_ticks]
   
    else:
        if not is_float:
            ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
        else:
            ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]

    lines.append('_' + '_'.join(ticks)+'_ ')
    for i in range(A.shape[0]):
        if not is_float:
            text = ["{:4d}".format(int(a)) for a in A[i]]
        else:
            text = ["{:5.2f}".format(a) for a in A[i]]
        line = ' '.join(text)
        lines.append('[' + line + ' ]')
    return lines

def format_masked_mat(A, mask, label_ticks, is_float):
    lines = []
    if label_ticks is not None:
        if not is_float:
            ticks = [str(a).rjust(4, '_') for a in label_ticks]
        else:
            ticks = [str(a).rjust(5, '_') for a in label_ticks]
   
    else:
        if not is_float:
            ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
        else:
            ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]

    lines.append('_' + '_'.join(ticks)+'_ ')
    for i in range(A.shape[0]):
        row = A[i]
        text = []
        if not is_float:
            for j in range(len(row)):
                if mask[i,j] == 1:
                    text.append("{:4d}".format(int(row[j]))) 
                elif mask[i,j] == 0:
                    text.append("{:>4}".format('*')) 
        else:
            for j in range(len(row)):
                if mask[i,j] == 1:
                    text.append("{:5.2f}".format(row[j])) 
                elif mask[i,j] == 0:
                    text.append("{:>5}".format('*')) 
        line = ' '.join(text)
        lines.append('[' + line + ' ]')
    return lines

def format_array(a, title):
    lines = []
    lines.append('_' + ''.join(title)+'_')
    for i in range(len(a)):
        text = ["{:5s}".format(str(a[i]))]
        line = ' '.join(text)
        lines.append(' ' + line + ' ')
    return lines
