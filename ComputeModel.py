import numpy as np

def compute_model(x,w,b):

    m=len(x)
    f = np.zeros(m)

    for i in range (0,m):
        f[i] = w * x[i] + b

    return f