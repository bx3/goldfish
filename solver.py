import sys
import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import svds
# import cvxpy as cvx
import time
import config

# A is W, B is H, k is rank which is L, X is observation
def run_pgd_nmf(L, X):
    # init A_init, B_init with svd
    A_init, B_init = init_matrix(X, L)
    A_est, B_est = alternate_minimize(A_init, B_init, X, L)
    return A_est, B_est

def print_matrix(A):
    for i in range(A.shape[0]):
        print(list(np.round(A[i], 3)))

# init A, B, X is observation matrix
def init_matrix(X, L):
    A, S, B = svds(X, L)
    # print('A')
    # print_matrix(A)
    # mask = X > 0
    # re = A.dot(np.diag(S)).dot(B)
    # diff = re - X 
    # print_matrix(diff)
    # # print(np.sum(mask*(< 0)))
    # print(np.sum(mask))
    # sys.exit(2)


    I = np.sign(A.sum(axis=0)) # 2 * int(A.sum(axis=0) > 0) - 1
    A = A.dot(np.diag(I))

    B = np.transpose((B.T).dot(np.diag(S*I)))
    # print('B')
    # print_matrix(B)
    # sys.exit(2)
    return A, B


# mask out non date entry, W is A, H is B
def alternate_minimize(A_init, B_init, X, L):
    start = time.time()
    A = A_init
    B = B_init
    I = X > 0 
    P = (A.dot(B) - X) * I 
    # print('P')
    # print_matrix(P)
    # print(np.sum(I))
    # print(np.sum(P < 0))
    # sys.exit(2)
    num_row = A.shape[0]
    # print('init A', A.shape)
    # print_matrix(A)

    prev_opt = 9999
    opts = []
    for epoch in range(config.num_alt):
        # norm_delta_A = 9999
        step_A = 0
        # update A
        while step_A < config.max_step:
            #prev_A = np.copy(A)
            grad_A = P.dot(B.T)
            t_A = 0.25 * LA.norm(grad_A, 'fro')**2 / LA.norm( grad_A.dot(B) * I, 'fro')**2
            A_tilde = A - t_A * grad_A
            for i in range(num_row):
                # A[i,:] = euclidean_proj_simplex(A_tilde[i], 1)
                # A[i,:] = proj_simplex_cvxpy(1, A_tilde[i])
                # A[i,:] =  projection_simplex_bisection(A_tilde[i], 1)
                # A[i,:] = projection_simplex_pivot(A_tilde[i])
                A[i,:] = projection_simplex_sort(A_tilde[i])

            P = (A.dot(B) - X) * I 
            step_A += 1
            #norm_delta_A = LA.norm(A-prev_A, 'fro')

        # update B
        # norm_delta_B = 9999
        step_B = 0
        while step_B < config.max_step:
            #prev_B = np.copy(B)
            grad_B = (A.T).dot(P)
            t_B = 0.25 * LA.norm(grad_B, 'fro')**2 / LA.norm( A.dot(grad_B) * I, 'fro')**2
            B = B - t_B * grad_B
            B[B<0] = 0 
            step_B += 1
            #norm_delta_B = LA.norm(B-prev_B, 'fro')

        P = (A.dot(B) - X) * I
        opt = 0.5 * LA.norm(P, 'fro')
        opts.append(opt)
        if prev_opt - opt < config.tol_obj:
            # print('opt', prev_opt - opt)
            # print_matrix(A)
            # print_matrix(B)
            #print('good opt')
            # sys.exit(2)
            # print('out', epoch, 'A', step_A, 'B', step_B)
            break
        prev_opt = opt
    # print(time.time() - start)
    # print_matrix(A)
    # print_matrix(B)
    # print(opts)
    # sys.exit(2)
    return A, B



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
