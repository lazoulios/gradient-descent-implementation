import matplotlib.pyplot as plt
import numpy as np
import ComputeModel as cm
import ComputeCost as cc
import ComputeGradient as cg
import pandas as pd

#importing the csv values
#TODO: Add multi linear functionality (more features) 
data = pd.read_csv("gradient-descent-implementation/house_data.csv")

x = data['SquareFootage'].values
y = data['Price'].values

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

#computing the model wiht the final parameters (first with normalized values, finally with unscaled for clearance)
f_wb_normalized = cm.compute_model(x_norm,w_final,b_final)

#setup for the two plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#plotting the prediction model (with unscaled values for x)
ax1.plot(x,f_wb_normalized,c='b',label='Prediction')
ax1.scatter(x, y, marker='x', c="red")
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
