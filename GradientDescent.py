import matplotlib.pyplot as plt
import numpy as np
import ComputeModel as cm
import ComputeCost as cc
import ComputeGradient as cg

#TODO: Add multi linear functionality (more features) 
x = np.array([12, 25, 30, 42, 73, 15, 24, 26, 53, 72, 84, 16])
y = np.array([10, 57, 30, 74, 14, 37, 90, 27 ,38 ,49, 42, 69])

#normalization of the feature
mu = np.mean(x,axis=0)
sigma = np.std(x,axis=0)
x_norm = (x-mu)/sigma

m = len(x_norm) #lenght of arrays

w = 0
b = 0

#input for learning rate and # of iterations
a = float(input('Enter leaning rate of gradient descent (suggested learning rate: 0.0009): '))
num_iter = int(input('Enter number of iterations for gradient descent: '))

#running gradient descent for the computations of the parameters
w_final, b_final, cost_history = cg.compute_gradient(x_norm, y, m, w, b, a, num_iter)

#computing the model wiht the final parameters
f_wb = cm.compute_model(x_norm,w_final,b_final)

#setup for the two plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#plotting the prediction model
ax1.plot(x_norm,f_wb,c='b',label='Prediction')
ax1.scatter(x_norm, y, marker='x', c="red")
ax1.set_xlabel("x")
ax1.set_title("Testing Gradient Descent")
ax1.legend()

#plotting the prediction model
ax2.plot(range(len(cost_history)), cost_history, c='green')
ax2.set_title("Cost vs Iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Cost")
ax2.grid(True)

#making them both appear
plt.suptitle("Gradient Descent Results", fontsize=14)
plt.tight_layout()
plt.show()
