import numpy as np
from scipy.linalg import solve_lyapunov

def getLyaponovPair(M,V):
    D = -(np.matmul(M,V) + np.matmul(V,np.transpose(M)))
    return D

M = np.matrix([[11,4],[-10,14]])
V = np.matrix([[1,2],[7,1]])

D = getLyaponovPair(M, V)

print(solve_lyapunov(M, -D))