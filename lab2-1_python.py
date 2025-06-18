# COURSE TITLE
# COURSE CODE
# Created by:
# Submitted to:
# Date Created:
# Date Revised:

# Machine Learning Exercise 1 - Linear Regression

# 1 INTRODUCTION (Give a short background about the topic)

# 2 CODE DESIGN
# Linear regression with one variable

# Code segment no. 1 (kindly explain, what are each libraries for?)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Code segment no. 2 (put your comments for this code segment)
import os
path = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()

# Code segment no. 3 (put your comments for this code segment)
data.describe()

# Code segment no. 4 (put your comments for this code segment)
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

# Code segment no. 5 (put your comments for this code segment)
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
	
# Code segment no. 6 (put your comments for this code segment)
data.insert(0, 'Ones', 1)

# Code segment no. 7 (put your comments for this code segment)
# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# Code segment no. 8 (put your comments for this code segment)
X.head()

# Code segment no. 9 (put your comments for this code segment)
y.head()

# Code segment no. 10 (put your comments for this code segment)
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# Code segment no. 11 (put your comments for this code segment)
theta

# Code segment no. 12 (put your comments for this code segment)
X.shape, theta.shape, y.shape

# Code segment no. 13 (put your comments for this code segment)
computeCost(X, y, theta)

# Code segment no. 14 (put your comments for this code segment)
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost
	
# Code segment no. 15 (put your comments for this code segment)
alpha = 0.01
iters = 1000

# Code segment no. 16 (put your comments for this code segment)
g, cost = gradientDescent(X, y, theta, alpha, iters)
g

# Code segment no. 17 (put your comments for this code segment)
computeCost(X, y, g)

# Code segment no. 18 (put your comments for this code segment)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

# Code segment no. 19 (put your comments for this code segment)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# Linear regression with multiple variables

# Code segment no. 20 (put your comments for this code segment)
path = os.getcwd() + '\data\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head()

# Code segment no. 21 (put your comments for this code segment)
data2 = (data2 - data2.mean()) / data2.std()
data2.head()

# Code segment no. 22 (put your comments for this code segment)
# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
computeCost(X2, y2, g2)

# Code segment no. 23 (put your comments for this code segment)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

# Code segment no. 24 (put your comments for this code segment)
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)

# Code segment no. 25 (put your comments for this code segment)
x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# 3 ANALYSIS OF RESULTS

# 4 CONCLUSION



