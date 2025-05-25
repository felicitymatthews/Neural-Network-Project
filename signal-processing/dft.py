import numpy as np

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype = np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += x[n]*np.exp(-2j * np.pi / N * n * k)
    return X
'''
helper functions
'''
def is_power_of_two(n):
    return n & (n-1) == 0

def dft_matrix_vector(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W_N = np.exp(-2j * np.pi * k * n / N)
    M_x = np.dot(W_N, k)
    return M_x
'''
Fast Fourier Transform Methods
'''
def fft(x):
    N = len(x)
    if not is_power_of_two(N):
        return dft_matrix_vector(x)
    X = np.zeros(N, dtype=np.complex128)
    if N == 1:
        return x
    even = fft(x[::2])
    odd = fft(x[1::2])
    W = np.exp(-2j* np.pi * np.arange(N) / N )
    half_N = N//2
    for k in range(half_N):
        X[k] = even[k] + W[k] * odd[k]
        X[k + half_N] = even[k] - W[k] * odd[k]
    return X
''' inverse fast fourier transform + helpers '''
def idft_matrix_vector(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W_N_i = np.exp(2j*np.pi*k*n/N)
    x_time = (1/N) * np.dot(W_N_i, x)
    return x_time
def ifft(X):
    N = len(X)
    if not is_power_of_two(N):
        return idft_matrix_vector(X)
    else:
        return ifft_helper(X)/N
def ifft_helper(X_freq):
    N = len(X_freq)
    if N == 1:
        return X_freq
    even = ifft_helper(X_freq[::2])
    odd = ifft_helper(X_freq[1::2])
    W = np.exp(2j * np.pi * np.arange(N) / N)
    x_time = np.zeros(N, dtype=np.complex128)
    half_N = N//2
    for k in range(half_N):
        x_time[k] = even[k] + W[k] * odd[k]
        x_time[k + half_N] = even[k] - W[k] * odd[k]
    return x_time
