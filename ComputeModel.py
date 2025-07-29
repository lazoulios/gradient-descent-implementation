import numpy as np

def compute_model(x,w,b):

    f = np.dot(x,w) + b

    return f