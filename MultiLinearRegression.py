import matplotlib.pyplot as plt
import numpy as np
import ComputeMultiLinearModel as cmlm
import ComputeCost as cc
import ComputeMultiLinearGradient as cmlg
import pandas as pd


def multiLinearRegression():
    #importing the csv values
    data = pd.read_csv("gradient-descent-implementation/house_data_multi_feature.csv")
    X = data[['SquareFootage', 'Bedrooms','HouseAge']].values
    y = data['Price'].values

    #normalization of the feature
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X-mu)/sigma

    m = len(X_norm) #lenght of arrays

    w = np.array([0, 0, 0])
    b = 0

    f = cmlm.compute_model(X[1],w,b)
    
    #input for learning rate and # of iterations
    a = float(input('Enter leaning rate of gradient descent (suggested learning rate: 0.0009): '))
    num_iter = int(input('Enter number of iterations for gradient descent: '))
    '''
    #running gradient descent for the computations of the parameters
    w_final, b_final, cost_history = cmlg.compute_gradient(X_norm, y, m, w, b, a, num_iter)

    #computing the model wiht the final parameters (first with normalized values, finally with unscaled for clearance)
    f_wb_normalized = cmlm.compute_multi_linear_model(X_norm,w_final,b_final)
    '''
a = multiLinearRegression()
