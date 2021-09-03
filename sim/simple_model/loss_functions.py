import torch
import sys

CONSTANT = 1e-10

class ClusterLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_fn = torch.nn.MSELoss(reduction='sum')
        self.softmax_func = torch.nn.Softmax(dim=0)

    # def forward(self, X, H, C, M, T, nM):
        # sim_loss = torch.tensor(0.0, dtype=torch.float32)
        # softmax_func = torch.nn.Softmax(dim=0)
        # for i in range(T):
            # row_loss = torch.tensor(0.0, dtype=torch.float32) 
            # scores = torch.ones(T) * float("inf")
            # for j in range(T):
                # if i != j:
                    # if (M[i] != M[j]).any():
                        # selected = torch.masked_select(X[i]-X[j], (M[i]*M[j]) >0)
                        # scores[j] = torch.sqrt(torch.var(selected))

            # topk, topk_ind = torch.topk(scores, 1, largest=False)
            # weight = softmax_func(-1*topk)

            # for k in range(len(topk_ind)):
                # j = topk_ind[k]
                # row_loss += weight[k]*torch.norm(H[i]-H[j])
            # sim_loss += row_loss
        # loss = self.mse_fn(X*M, (H-C)*M) + sim_loss + 0.1*torch.norm(C)
        # return loss

    def forward(self, X, H, C, M, T, nM, row_scores, mc_rows):
        sim_loss = torch.tensor(0.0, dtype=torch.float32)
        for i in mc_rows:
            topk, topk_ind = torch.topk(row_scores[i], 3, largest=False)
            weight = self.softmax_func(-1*topk)

            for k in range(len(topk_ind)):
                j = topk_ind[k]
                sim_loss += weight[k]*torch.norm(H[i]-H[j])
                # print(i, j, H[i], H[j], weight, sim_loss, topk, topk_ind, row_scores[i])
                
        loss = self.mse_fn(X*M, (H-C)*M) + sim_loss + 0.1*torch.norm(C) + 0.1*torch.norm(H)
        return loss



class ElementLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, A, C, easy_estimates_by_row, known_pos_by_row, element_peer_scores, tensor_ind_map, T):
        sim_loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(T):
            if i in easy_estimates_by_row:
                row_loss = torch.tensor(0.0, dtype=torch.float32)
                easy_unknown = easy_estimates_by_row[i]

                for j in easy_unknown:
                    for weight, k, mask in element_peer_scores[(i,j)]:

                        t = tensor_ind_map[(i,j)]
                        a = (X[k] + C[k]) - (X[i] + C[i])
                        a[j] -= (A[t] - C[i] ) # without - X[i,j], since it is 0
                        b = torch.masked_select(a, mask)
                        # print(i,j, 'close to', k, 'weight', weight, mask, a, b)
                        # print(X[k]*max_time)
                        # print(X[i]*max_time)

                        row_loss += weight/float(len(easy_unknown))*torch.norm(b)

                sim_loss += row_loss
                
        return sim_loss + 1*torch.norm(C) + 1*torch.norm(A)



# def forward(self, X, A, C, easy_estimates, known_pos_by_row, element_peer_scores, tensor_ind_map, indicators, N):
    # sim_loss = torch.tensor(0.0, dtype=torch.float32)
    # for i,j in easy_estimates:
        # ele_loss = torch.tensor(CONSTANT, dtype=torch.float32)
        # for weight, k in element_peer_scores[(i,j)]:
            # a = (X[k,j] + C[k] - A[tensor_ind_map[(i,j)]]) 
            # ele_loss += weight*torch.square(a)
        # sim_loss += torch.sqrt(ele_loss)

