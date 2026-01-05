import numpy as np

def mass(h):
    return np.sum(h)

def momentum(h,u):
    return np.squeeze(np.asarray(h)).dot(np.squeeze(np.asarray(u)))