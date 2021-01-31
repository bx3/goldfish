import numpy as np
import solver
import nndsvd
from scipy.sparse.linalg import svds
from scipy.linalg import eigh
import sys
from numpy.linalg import inv
from sklearn.decomposition import NMF
from numpy import linalg as LA
import itertools
import matplotlib.pyplot as plt

def print_matrix(A):
    for i in range(A.shape[0]):
        print(list(np.round(A[i], 3)))

def test_mf(T, N, L, std, num):
    results = []
    print('T', T, 'N', N, 'L', L)
    stds = [i*10 for i in range(9)]
    std_results = []
    for std in stds:
        print('std', std)
        results = []
        for _ in range(num):
            r = test_single_mf(T, N, L, std)
            results.append(sum(r)/len(r))
        std_results.append(sum(results)/len(results))
    print(stds, std_results)
    plt.scatter(stds, std_results)
    plt.title('mf analysis by brute forcing closest W')
    plt.xlabel('gaussian noise std. data mean is set 100. ' + 'each dot is mean of ' + str(num) + 'mf')
    plt.ylabel('correctness ratio')
    plt.show()

def test_single_mf(T, N, L, std):
    W = np.zeros((T, L))
    broad_nodes = []
    for i in range(T):
        j = np.random.randint(L)
        broad_nodes.append(j)
        W[i,j] = 1

    # closeness matrix
    H = np.random.rand(L, N) * 100
    # construct X
    X = W.dot(H)

    X_noise = X + np.random.normal(0, std, size=(T, N)) 
    X_noise[X_noise<0] = 0

    num_alt = 5000
    tol_obj = 0.001
    rho_W = 0
    rho_H = 0

    W_init, H_init = nndsvd.initial_nndsvd(X_noise, L, 10) # svd_init(X, L) 
    W_est, H_est, opt = solver.alternate_minimize(W_init, H_init, X_noise, L, None, 0, 
            num_alt, tol_obj, rho_W, rho_H)
    
    est_to_ori = get_distance_table(H, H_est)
    # print(sorted(est_to_ori.items()))
    W_re = reorder_W(W_est, est_to_ori)

    result = (np.argmax(W, axis=1) == np.argmax(W_re, axis=1)) 
    return result 

def reorder_W(W_est, est_to_ori):
    row, col = W_est.shape
    W_re = np.zeros((row, col))
    for i in range(col):
        re_idx = est_to_ori[i]
        W_re[:, re_idx] = W_est[:, i]
    return W_re    

def best_perm(table):
    num_node = table.shape[0]
    nodes = [i for i in range(num_node)]
    best_comb = None
    best = 999
    hist = {}
    for comb in itertools.permutations(nodes, num_node):
        score = 0
        for i in range(num_node):
            j = comb[i]
            score += table[i,j]  
        hist[comb] = score
        if best_comb is None or score < best:
            best_comb = comb
            best = score
    est_to_ori = {}
    for i in range(num_node):
        est = best_comb[i]
        est_to_ori[est] = i

    # print(best_comb, best)
    return est_to_ori


def get_distance_table(H, H_est):
    table = np.zeros((len(H), len(H_est)))
    est_to_ori = {}
    for i in range(len(H)):
        u = H[i,:]
        for j in range(len(H_est)):
            v = H_est[j,:]
            dis = LA.norm(u-v) 
            table[i,j] = dis
    # print(table)
    est_to_ori = best_perm(table)

    # for i in range(len(H)):
        # est = np.argmin(table[i,:])
        # est_to_ori[est] = i 
    # print(m)
    # print(est_to_ori)
    return est_to_ori 

def svd_init(X, L):
    A, S, B = svds(X, L)
    I = np.sign(A.sum(axis=0)) # 2 * int(A.sum(axis=0) > 0) - 1
    A = A.dot(np.diag(I))
    B = np.transpose((B.T).dot(np.diag(S*I)))
    return A, B

def uniform_init(X, L):
    T, N = X.shape
    W_init = np.random.rand(T, L)
    H_init = np.random.rand(L, N) * 100
    return W_init, H_init

def debug(W, H, W_est, H_est, X):
    print('W')
    print_matrix(W)
    print('H')
    print_matrix(H)   
    print('X')
    print_matrix(X)   
    print('W estimate')
    print_matrix(W_est)
    print('H estimate')
    print_matrix(H_est)   


def square_root(X, L):
    T, N = X.shape
    p = X.T.dot(X)

    # model = NMF(n_components=L, init='nndsvd', random_state=0)
    # A = model.fit_transform(p)
    # H = model.components_
    # W, H = nndsvd.initial_nndsvd(p, L, 1) #svd_init(X, L)  # 

    w, v = eigh(p, subset_by_index=[N-L, N-1])
    # print(w/max(w)*T)
    # sys.exit(2) 
    # print_matrix(A.T)
    # print()

    # print_matrix(H)
    # print()
    # te = inv(H.dot(H.T))
    # inverse_H = H.T.dot(te)
    # W = X.dot(inverse_H)
    # print_matrix(W)


    #w, v = eigh(p, subset_by_index=[N-L, N-1]) #, subset_by_index=[N-L, N-1]) # largest L eigenvalue
    # print(v)
    # print(w)

    # print_matrix(H_init)
    # c = H_init.T.dot(H_init)
    # print_matrix(X)
    # print_matrix(H_init)

    # print('W')
    # print_matrix(W_init)
    # sys.exit(2)
    return W, H

    

