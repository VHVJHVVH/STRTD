import numpy as np
from numpy.linalg import inv as inv
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)

def svt_tnn(mat, alpha, rho, theta):
    tau = alpha / rho
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices = 0)
        s = np.sqrt(s)
        idx = np.sum(s > tau)
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta:idx] = (s[theta:idx] - tau) / s[theta:idx]
        return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
    elif m > 2 * n:
        return svt_tnn(mat.T, tau, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices = 0)
    idx = np.sum(s > tau)
    vec = s[:idx].copy()
    vec[theta:idx] = s[theta:idx] - tau
    return u[:, :idx] @ np.diag(vec) @ v[:idx, :]
def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])
def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]


def LRTC(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter):

    dim = np.array(sparse_tensor.shape)
    pos_missing = np.where(sparse_tensor == 0)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))

    X = np.zeros(np.insert(dim, 0, len(dim)))  # \boldsymbol{\mathcal{X}}
    T = np.zeros(np.insert(dim, 0, len(dim)))  # \boldsymbol{\mathcal{T}}
    Z = sparse_tensor.copy()
    last_tensor = sparse_tensor.copy()
    snorm = np.sqrt(np.sum(sparse_tensor ** 2))
    it = 0
    while True:
        rho = min(rho * 1.05, 1e5)
        for k in range(len(dim)):
            X[k] = mat2ten(svt_tnn(ten2mat(Z - T[k] / rho, k), alpha[k], rho, theta), dim, k)
        Z[pos_missing] = np.mean(X + T / rho, axis=0)[pos_missing]
        T = T + rho * (X - np.broadcast_to(Z, np.insert(dim, 0, len(dim))))
        tensor_hat = np.einsum('k, kmnt -> mnt', alpha, X)
        tol = np.sqrt(np.sum((tensor_hat - last_tensor) ** 2)) / snorm
        last_tensor = tensor_hat.copy()
        import scipy.io as io
        X_hat = tensor_hat
        io.savemat('X_hat.mat', {'X_hat': X_hat})

        it += 1
        if (it + 1) % 50 == 0:
            print('Iter: {}'.format(it + 1))
            print('RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
            print()
        if (tol < epsilon) or (it >= maxiter):
            break

    print('Imputation MAPE: {:.6}'.format(compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])))
    print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
    print()

    return tensor_hat
import numpy as np
import pandas as pd
import time
import scipy.io

for r in [0.3 ]:
    print('Missing rate = {}'.format(r))
    missing_rate = r
    dense_tensor = scipy.io.loadmat('datasets/tensor.mat')['tensor']
    random_matrix = scipy.io.loadmat('datasets/random_matrix.mat')['random_matrix']
    ## Random missing (RM)
    dim1, dim2, dim3 = dense_tensor.shape
    np.random.seed(1000)
    sparse_tensor = dense_tensor * np.round(random_matrix + 0.5 - missing_rate)[:, :, None]
    start = time.time()
    alpha = np.ones(3) / 3
    rho = 5e-5
    theta = 9
    if r > 0.3 and r < 0.6:
        theta = 8
        rho = 5e-5
    if r > 0.5 and r < 0.8:
        theta = 7
        rho = 5e-5
    if r > 0.7and r<0.9:
        theta = 5
        rho = 5e-5
    epsilon = 1e-4
    maxiter = 100
    LRTC(dense_tensor, sparse_tensor, alpha, rho, theta, epsilon, maxiter)
    end = time.time()
    print('Running time: %d seconds'%(end - start))
    print()