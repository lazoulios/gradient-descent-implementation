import numpy as np
import ComputeCost as cc

def compute_gradient(x,y,m,w,b,a,num_iter):
    j_hist=[]
    
    for i in range(0,num_iter):
        dj_dw = 0
        dj_db = 0
        for j in range (0,m):
            f_wb=w*x[j]+b
            dj_dw += (f_wb - y[j]) * x[j]
            dj_db += (f_wb - y[j])
        dj_dw = dj_dw/m
        dj_db = dj_db/m
        w = w - a*dj_dw
        b = b - a*dj_db

        cost = cc.compute_cost(x, y, m, w, b)
        j_hist.append(cost)

        if i%100==0: #testing the cost, w and b every once in a while to make sure in not overshooting
            print(f"at {i}th iteration cost: {cost}, w: {w}, b: {b}")

    return w,b,j_hist