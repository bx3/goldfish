import torch

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
                
        loss = self.mse_fn(X*M, (H-C)*M) + sim_loss + 0.1*torch.norm(C) + 0.01*torch.norm(H)
        return loss



class ElementLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, A, M, T, nM, row_elements_pos, max_time):
        sim_loss = torch.tensor(0.0, dtype=torch.float32)
        softmax_func = torch.nn.Softmax(dim=0)

        # h_A_list = []
        # k = 0
        # for i in range(T):
            # indices = row_elements_pos[i]
            # if len(indices) > 0:
                # h_i_A = X[i].index_add(0, indices, A[k:k+len(indices)]) 
                # h_A_list.append(h_i_A)
                # k += len(indices)
            # else:
                # h_A_list.append(X[i])
        # h_A = torch.stack(h_A_list)
        h_A = X

        q = 0
        for i in range(T):
            indices = row_elements_pos[i]
            if len(indices) > 0:
                scores = torch.ones(T) * float("inf")
                for j in range(T):
                    if i != j:
                        if (M[i] != M[j]).any():
                            selected = torch.masked_select(h_A[i]-h_A[j], (M[i]*M[j]) >0)
                            scores[j] = torch.sqrt(torch.var(selected))

                topk, topk_ind = torch.topk(scores, 3, largest=False)
                # print(i, scores)
                weight = softmax_func(-1*topk)
                # print(i, indices, len(indices))
                for r in range(len(indices)):
                    index = indices[r]
                    for k in range(len(topk_ind)):
                        j = topk_ind[k]
                        print(q, i,  index, A[q], h_A[j, index])
                        sim_loss += weight[k]*torch.norm(A[q]-h_A[j, index])
                    q += 1
        # print('finish', A)
        return sim_loss
