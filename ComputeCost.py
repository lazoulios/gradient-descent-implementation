def compute_cost(x,y,m,w,b):
    j_wb = 0
    for i in range(0,m):
        f_wb = w*x[i]+b
        j_wb = j_wb + (f_wb - y[i])**2
        print("cost at",i,"th iteration: ",j_wb)
    j_wb = (1/(2*m))*j_wb
    return j_wb