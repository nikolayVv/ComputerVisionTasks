import math

import numpy as np


def kalman_step(A, C, Q, R, y, x, V):
    # INPUTS:
    # A - the system matrix
    # C - the observation matrix
    # Q - the system covariance
    # R - the observation covariance
    # y(:)   - the observation at time t
    # x(:) - E[X | y(:, 1:t-1)] prior mean
    # V(:,:) - Cov[X | y(:, 1:t-1)] prior covariance
    # OUTPUTS (where X is the hidden state being estimated)
    # xnew(:) =   E[ X | y(:, 1:t) ]
    # Vnew(:,:) = Var[ X(t) | y(:, 1:t) ]
    # VVnew(:,:) = Cov[ X(t), X(t-1) | y(:, 1:t) ]
    # loglik = log P(y(:,t) | y(:,1:t-1)) log-likelihood of innovation

    xpred = np.matmul(A, x)
    Vpred = np.matmul(np.matmul(A, V), A.transpose()) + Q

    e = y - np.matmul(C, xpred)  # error (innovation)
    S = np.matmul(np.matmul(C, Vpred), C.transpose()) + R
    ss = max(V.shape)
    loglik = gaussian_prob(e, np.zeros((1, e.size), dtype=np.float32), S, use_log=True)
    K = np.linalg.lstsq(S.transpose(), np.matmul(Vpred, C.transpose()).transpose(), rcond=None)[0].transpose()  # Kalman gain matrix
    # If there is no observation vector, set K = zeros(ss).
    xnew = xpred + np.matmul(K, e)
    Vnew = np.matmul((np.eye(ss) - np.matmul(K, C)), Vpred)
    VVnew = np.matmul(np.matmul((np.eye(ss) - np.matmul(K, C)), A), V)

    return xnew, Vnew, loglik, VVnew

def gaussian_prob(x, m, C, use_log=False):
    # Evaluate multivariate Gaussian density
    # p(i) = N(X(:,i), m, C) where C = covariance matrix and each COLUMN of x is a datavector
    # p = gaussian_prob(X, m, C, 1) returns log N(X(:,i), m, C) (to prevents underflow).
    # If X has size dxN, then p has size Nx1, where N = number of examples

    if m.size == 1:
        x = x.flatten().transpose()

    d, N = x.shape

    m = m.flatten()
    M = np.reshape(m * np.ones(m.shape, dtype=np.float32), x.shape)
    denom = (2 * math.pi)**(d/2) * np.sqrt(np.abs(np.linalg.det(C)))
    mahal = np.sum(np.linalg.solve(C.transpose(), (x - M)) * (x - M))   # Chris Bregler's trick

    if np.any(mahal < 0):
        print('Warning: mahal < 0 => C is not psd')

    if use_log:
        p = -0.5 * mahal - np.log(denom)
    else:
        p = np.divide(np.exp(-0.5 * mahal), (denom + 1e-20))

    return p

def sample_gauss(mu, sigma, n):
    # sample n samples from a given multivariate normal distribution
    return np.random.multivariate_normal(mu, sigma, n)
