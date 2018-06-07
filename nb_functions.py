import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def generate_noise(sigma, N_x, N_rep, level=2, noise_seed=456):
    """Generates the noise which is added to the data, can generate level 1 or
    level 2 noise
    """
    np.random.seed(noise_seed)
    eta = sigma*np.random.randn(N_x)
    if level == 2:
        delta = sigma*np.random.randn(N_x, N_rep)
    else:
        delta = 0
    noise = np.zeros((N_x, N_rep))
    for k in range(N_rep):
        noise[:, k] = eta
    return noise + delta

def generate_x(N_x, x_i, x_f, x_seed=123):
    """Generates the x points"""
    np.random.seed(x_seed)
    return  x_i + np.random.rand(N_x)*(x_f-x_i)

def generate_hoc(x, N_model, N_law, N_rep=1):
    """Generates the higher order coefficients when law has more terms than
    fitting model, these need to be added to the noise vector during the fit
    """
    N_max = max([N_model, N_law])
    hoc = np.zeros(x.shape[0])
    for i in range(N_model, N_max):
        hoc += x**i
    hoc_rep = np.zeros((x.shape[0], N_rep))
    for k in range(N_rep):
        hoc_rep[:, k] = hoc
    return hoc_rep

def fit_law(x, noise, hoc_rep, N_model, N_law):
    """Functions takes x points, noise and higher order coefficients, and fits
    the law with N_law coefficients using a model with N_model coefficients
    returns N_coeff*N_rep array of coefficients
    """
    N_rep = noise.shape[1]
    N_c = min(N_law, N_model)
    c_rep = np.zeros((N_model, N_rep))
    c_rep[:N_c, :] = np.ones((N_c, N_rep))
    A_inverse = np.zeros((N_model, N_model))
    v = np.zeros((N_model, N_rep))
    noise_hoc = hoc_rep + noise
    for k in range(N_rep):
        for i in range(N_model):
            v[i, k] = (noise_hoc[:, k]*(x**(i))).mean()
            for j in range(N_model):
                A_inverse[i, j] = (x**(i+j)).mean()
    A = inv(A_inverse)
    return np.matmul(A, v) + c_rep

def generate_y(x, coeff):
    y = np.zeros(x.shape)
    for i, c in enumerate(coeff):
        y += c*(x**(i))
    return y

def tv_split(N_x):
    """returns two sets of indices for making a training validation split"""
    return np.split(np.arange(N_x)[np.random.permutation(N_x)], 2)
    
