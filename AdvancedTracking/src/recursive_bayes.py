import numpy as np
import sympy as sp
from ex4_utils import kalman_step, sample_gauss

def kalman_filter(x, y, q=1, r=1, model_type='NCA'):
    A, Q_i = get_dynamic_model(model_type, q)
    C, R_i = get_observation_model(model_type, r)

    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

    sx[0] = x[0]
    sy[0] = y[0]

    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]
    covariance = np.eye(A.shape[0], dtype=np.float32)

    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)

        sx[j] = state[0]
        sy[j] = state[1]

    return sx, sy

def get_dynamic_model(model_type='NCA', q_val=1):
    if model_type == 'RW':
        F = [
            [0, 0],
            [0, 0]
        ]

        L = [
            [1, 0],
            [0, 1]
        ]
    elif model_type == 'NCV':
        F = [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]

        L = [
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]
    else:
        F = [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
        
        L = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]

    T, q = sp.symbols('T q')    
    F = sp.Matrix(F)
    L = sp.Matrix(L)

    A = sp.exp(F * T)
    Q = sp.integrate((A * L) * q * (A * L).T, (T, 0, T))

    return np.array(A.subs(T, 1), dtype=np.float32), np.array(Q.subs({T: 1, q: q_val}), dtype=np.float32)

def get_observation_model(model_type='NCA', r=1):
    if model_type == 'RW':
        C = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float32)
    elif model_type == 'NCV':
        C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
    else:
        C = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

    R = np.array([
        [r, 0],
        [0, r]
    ], dtype=np.float32)

    return C, R