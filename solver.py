import sys
#import autograd.numpy as np
import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import svds
# import cvxpy as cvx
import time
import config
import nndsvd

# A is W, B is H, k is rank which is L, X is observation, new_msgs_ind is start index
def run_pgd_nmf(i, slots, N, L, W, H, new_msgs_ind, init_new):
    X, mask, max_time = construct_table(N, slots)
    W_init, H_init = init_matrix(L, X, mask, W, H, new_msgs_ind, init_new)
    W_est, H_est, opt = alternate_minimize(W_init, H_init, X, mask, L, None, i, 
            config.num_alt, config.tol_obj, config.rho_W, config.rho_H)

    return W_est, H_est, opt

def print_mat(A):
    for i in range(A.shape[0]):
        print(list(np.round(A[i], 3)))

# init A, B, X is observation matrix
def init_matrix(L, X, mask, W, H, new_msg_start, init_new):
    if init_new : #or (not config.feedback_WH) or (not config.prior_WH)
        if config.init_nndsvd:
            A, B = nndsvd.initial_nndsvd(X*mask, L, config.nndsvd_seed)
        else:
            A, S, B = svds(X, L)
            I = np.sign(A.sum(axis=0)) # 2 * int(A.sum(axis=0) > 0) - 1
            A = A.dot(np.diag(I))
            B = np.transpose((B.T).dot(np.diag(S*I)))
        return A, B
    else :
        shape = W.shape 
        T = W.shape[0]
        N = H.shape[1]
        L = W.shape[1]

        W_out = np.zeros(W.shape)
        num_msg = T-new_msg_start
        new_noise = np.random.rand(num_msg, L)
        W_out[:new_msg_start] = W[num_msg:]
        W_out[new_msg_start:] = new_noise

        # A[:new_msg_start] = W[:new_msg_start] + np.random.normal(
                    # config.W_noise_mean, 
                    # config.W_noise_std, size=(new_msg_start, L)) 
        # A[new_msg_start:] = np.ones((T-new_msg_start, L))  # np.random.rand(T-new_msg_start, L)
        # np.ones((T-new_msg_start, L))
        # H_mean = np.mean(H)
        return W_out, H

def construct_table(N, slots):
    T = len(slots)
    X = np.zeros((T, N)) 
    mask = np.zeros((T, N))
    i = 0 # row

    max_time = 0 
    for slot in slots:
        for p, t in slot:
            X[i, p] = t
            mask[i, p] = 1
            if max_time < t:
                max_time = t
        i += 1
    return X, mask, max_time




def update_right(A, S, X):
    """Update right factor for matrix completion objective."""
    m, n = A.shape
    _, k = X.shape
    Y = np.zeros((n, k))
    # For each row, solve a k-dimensional regression problem
    # only over the nonzero projection entries. Note that the
    # projection changes the least-squares matrix siX so we
    # cannot vectorize the outer loop.
    for i in range(n):
        si = S[:, i]
        sia = A[si, i]
        siX = X[si]
        Y[i,:] = np.linalg.lstsq(siX, sia, rcond=-1)[0]
    return Y

def update_left(A, S, Y):
    return update_right(A.T, S.T, Y)





# def update_right(X, I, W):
    # T, N = X.shape
    # _, L = W.shape
    # W_tilde = np.zeros(L, T)
    
    # for i in range(N):
        # si = I[:, i]
        # six = X[si, i]
        # siW = W.T[si]
        # print(siW)
        # print(six)
        # W_tilde[:,i] = np.linalg.lstsq(siW, six)[0]
    # return W_tilde

# def update_left(X, I, H):
    # return update_right(X.T, I.T, H) 

def mc_obj(X, I, W, H):
    return 0.5 * np.linalg.norm(X - np.multiply(np.dot(W, H.T), I))**2

# mask out non date entry, W is A, H is B
def alternate_minimize(W_init, H_init, X, I, L, prev_H, node_id, 
        num_alt, tol_obj, rho_W, rho_H):
    start = time.time()
    W = W_init
    H = H_init
    P = (W.dot(H) - X) * I 
    num_row = W.shape[0]

    prev_opt = 9999
    opts = []
    for epoch in range(num_alt):
        # norm_delta_A = 9999
        step_A = 0
        # update A
        #while step_A < 10 and diff_A < 0.01: 
        grad_W = P.dot(H.T)
        # if LA.norm( grad_W.dot(H) * I, 'fro')**2 == 0:
            # print('Error. denom is 0')
            # sys.exit(1)

        t_W = 0.01 * LA.norm(grad_W, 'fro')**2 / LA.norm( grad_W.dot(H) * I, 'fro')**2
        #t_W = np.trace(P.T.dot(grad_W.dot(H))) / np.trace(H.T.dot(grad_W.T).dot(grad_W).dot(H))
        # t_A = 1/LA.norm(B.dot(B.T), 'fro')**2

        W = W - t_W * (grad_W + rho_W*(W) ) # 
        # W_tilde = update_left(X, I, H.T) 
        # W = W - grad(lambda W: mc_obj(X, I, W, H))(W)
        for i in range(num_row):
            # A[i,:] = euclidean_proj_simplex(A_tilde[i], 1)
            # A[i,:] = proj_simplex_cvxpy(1, A_tilde[i])
            # A[i,:] =  projection_simplex_bisection(A_tilde[i], 1)
            # A[i,:] = projection_simplex_pivot(A_tilde[i])
            W[i,:] = projection_simplex_sort(W[i])

        grad_H = (W.T).dot(P)
        # t_H = 0.1 * LA.norm(grad_H, 'fro')**2 / LA.norm( W.dot(grad_H) * I, 'fro')**2
        t_H = 0.01 * LA.norm(grad_H, 'fro')**2 / LA.norm( W.dot(grad_H) * I, 'fro')**2

        # t_H = np.trace(P.T.dot(W).dot(grad_H)) / np.trace(grad_H.T.dot(W.T).dot(W).dot(grad_H))
        # # t_H = 1/ LA.norm(A.T.dot(A), 'fro')**2

        # l1 norm derivative of H row
        # H_l1_deri = 2*(grad_H > 0) - 1


        H = H - t_H * (grad_H + rho_H*(H))  
        # H = update_right(X, I, W).T
        # H = H - grad(lambda H: mc_obj(X, I, W, H))(H)
        H[H<0] = 0 

        P = (W.dot(H) - X) * I
        opt = 0.5 * LA.norm(P, 'fro')
        opts.append(opt)
        diff = prev_opt - opt
        if diff > 0 and diff < tol_obj: # and opt < 10
            # print('opt', prev_opt - opt)
            break
        prev_opt = opt

    # if node_id == 39:
    # print('node', node_id, 
            # round(time.time() - start,3), 
            # round(prev_opt - opt), 
            # round(opt, 3), 
            # round(LA.norm(B, 'fro'), 3), 
            # round(np.mean(B), 3),
            # round(np.max(B), 3))
        # print('Matrix A')
        # print_matrix( A)
        # print('Matrix B')
        # print_matrix(B)
        # print(opts)
        # print(prev_opt - opt)
    # if prev_B is not None:
        # sys.exit(2)
    return W, H, opt 



# code from 
# gist daien/simplex_projection.py 
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def proj_simplex_cvxpy(a, y):
    # setup the objective and constraints and solve the problem
    x = cvx.Variable(len(y))
    obj = cvx.Minimize(cvx.sum_squares(x - y))
    constr = [x >= 0, sum(x) <= 1]
    prob = cvx.Problem(obj, constr)
    prob.solve()


# code from
# https://gist.github.com/mblondel/6f3b7aaad90606b98f71
def projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
    lower = 0
    upper = np.max(v)
    current = np.inf

    for it in range(max_iter):
        if np.abs(current) / z < tau and current < 0:
            break

        theta = (upper + lower) / 2.0
        w = np.maximum(v - theta, 0)
        current = np.sum(w) - z
        if current <= 0:
            upper = theta
        else:
            lower = theta
    return w

def projection_simplex_pivot(v, z=1, random_state=None):
    rs = np.random.RandomState(random_state)
    n_features = len(v)
    U = np.arange(n_features)
    s = 0
    rho = 0
    while len(U) > 0:
        G = []
        L = []
        k = U[rs.randint(0, len(U))]
        ds = v[k]
        for j in U:
            if v[j] >= v[k]:
                if j != k:
                    ds += v[j]
                    G.append(j)
            elif v[j] < v[k]:
                L.append(j)
        drho = len(G) + 1
        if s + ds - (rho + drho) * v[k] < z:
            s += ds
            rho += drho
            U = L
        else:
            U = G
    theta = (s - z) / float(rho)
    return np.maximum(v - theta, 0)

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w
