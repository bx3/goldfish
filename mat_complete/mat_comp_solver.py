import torch
import numpy as np
import sys
import time

epochs = 1000
row_penalty = 10
lr = 1e-2
NoneTime=9999

class CompletionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_fn = torch.nn.MSELoss(reduction='sum')

    def forward(self, X, H, C, M, T):
        row_loss = torch.tensor(0.0)
        softmax_func = torch.nn.Softmax(dim=0)
        for i in range(T):
            scores = torch.ones(T) *9999
            for j in range(T):
                if i != j:
                    if torch.argmin(M[i]) != torch.argmin(M[j]):
                        # scores[j] = torch.norm((H[i]/torch.max(H[i])-H[j]/torch.max(H[j]))*M[i]*M[j]) # old normalize
                        selected = torch.masked_select(H[i]-H[j], (M[i]*M[j]) >0)
                        scores[j] = torch.var(selected)
            min_row = torch.argmin(scores)
            row_loss += torch.norm(H[i] - H[min_row])
        loss = self.mse_fn(X*M, (H-C)*M) +  row_penalty*row_loss  #para_penalty*para_loss +
        return loss

    # def forward(self, X, H, C, M, T):
        # row_loss = torch.tensor(0.0) 
        # softmax_func = torch.nn.Softmax(dim=0)
        # for i in range(T):
            # scores = torch.ones(T) * float("inf")
            # for j in range(T):
                # if i != j:
                    # if torch.argmin(M[i]) != torch.argmin(M[j]):
                        # selected = torch.masked_select(H[i]-H[j], (M[i]*M[j]) >0)
                        # scores[j] = torch.var(selected)  
                        
            # topk, topk_ind = torch.topk(scores, 2, largest=False)
            # weight = softmax_func(-1*topk)
            
            # for k in range(len(topk_ind)):
                # j = topk_ind[k]
                # row_loss += weight[k]*torch.norm(H[i] - H[j])
        # loss = self.mse_fn(X*M, (H-C)*M) +  row_penalty*row_loss  
        # return loss

def construct_table(slots, node_id, log_directions):
    id_set = set()
    id_list = {}
    for slot in slots:
        # if node_id == 0: 
            # print(len(slots))
            # print(slot)
        for p, t, direction in slot:
            if direction in log_directions:
                id_set.add(p)
    ids = sorted(list(id_set))
    id_map = {}

    N = len(ids)
    T = len(slots)
    
    for i in range(N):
        id_map[ids[i]] = i

    X = np.zeros((T, N)) 
    mask = np.zeros((T, N))
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
                    if max_time < t:
                        max_time = t
                # TODO think about if None be infinite 
                else:
                    j = id_map[p]
                    X[i, j] = 9999
                    mask[i, j] = 1

        i += 1
    # if i ==0:
        # print(slots)
        # print(mask)
        # print(ids)
    return X, mask, max_time, ids

def run(i, slots, broads):
    # if i == 0:
        # print('mat comp', slots)
    X, M, max_time, ids = construct_table(slots, i, ['outgoing'])

    # if i == 0:
        # print_mats([format_masked_mat(X, M, ids, False), format_array(broads, 'src')])
    H_cpl, C_cpl, opt = complete_mat(X, M)
    # if i == 0:
        # print_mats([format_mat(H_cpl, ids, False), format_array(broads, 'src')])
    # print('finish i')
    return H_cpl, C_cpl, opt, i, ids
    # return None, None, None, i, None

def complete_mat(X_numpy, M_numpy):
    T, N = X_numpy.shape
    X = torch.tensor(X_numpy)
    M = torch.tensor(M_numpy)

    H = X.clone().requires_grad_(True)
    C = torch.rand(T, 1, requires_grad=True)

    criterion = CompletionLoss()
    prox_plus = torch.nn.Threshold(0,0)
    for e in range(epochs):
        loss = criterion(X, H, C, M, T)
        loss.backward()

        with torch.no_grad():
            s = (X-(H-C) )*M

            H = prox_plus(H - lr * H.grad)
            C = prox_plus(C - lr * C.grad)

            H.requires_grad_()
            C.requires_grad_()

            H.grad = None
            C.grad = None   
    return H.detach().numpy(), C.detach().numpy(), s

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
