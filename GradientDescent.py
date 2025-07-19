import matplotlib.pyplot as plt
import numpy as np
import ComputeModel as cm
import ComputeCost as cc

x = np.array([12, 25, 30, 42, 73, 15, 24, 26, 53, 72, 84, 16])
y = np.array([10, 57, 30, 74, 14, 37, 90, 27 ,38 ,49, 42, 69])

m = len(x) #lenght of arrays

w = 0
b = 0
f = cm.compute_model(x,w,b) 
j_wb = cc.compute_cost(x,y,m,w,b)

plt.plot(x,f,c='b',label='Prediction')
plt.scatter(x, y, marker='x', c="red")
plt.title("Testing Gradient Descent")
plt.show()
