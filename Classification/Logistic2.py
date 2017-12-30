# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 11:19:25 2017

@author: VictorChen
"""

import numpy as np
import matplotlib.pyplot as plt

'''
    Basic Logistic Regression to deal with Classification 
'''

def loadData(filePath):
    return np.loadtxt(filePath,delimiter=',')

def plotData(x,y):
    index0 = (y.reshape(-1)==0)
    index1 = (y.reshape(-1)==1)
    x0 = x[index0]
    x1 = x[index1]
    pz = plt.scatter(x0[:,0],x0[:,1],c='red')
    po = plt.scatter(x1[:,0],x1[:,1],c='green')
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.legend((pz,po),('$y==0$','$y==1$'))
    plt.title('Scatter Fig')
    return plt
    
def plotBoudary(theta,X):
    yp = predict(theta,X)
    x = X[:,1:3]
    p = plotData(x,yp)
    i = np.arange(x.min(),x.max())
    yb = (theta[1]*i+theta[0])/float(-theta[2])
    p.plot(i,yb,'black')
    return p
    
def sigmoid(z):
    return 1/(np.exp(-z)+1)

def predict(theta,X):
    return (X.dot(theta) >= 0.5)*1

def CostFunction(theta,X,y):
    k = y.reshape(-1)
    error = sigmoid(X.dot(theta))
    index0 = (k==0)
    index1 = (k==1)
    error0 = error[index0]
    error1 = error[index1]
    m = k.size
    return -( np.sum(np.log10(error1)) + np.sum(np.log10(1-error0)) )/float(m)

'''
    Use BSG to find theta to minimize cost
'''
def GradientDescend(theta,X,y,iterNum=100,alpha=0.01):
    m = y.size
    k = y.reshape(-1,1)
    J_history = np.zeros(iterNum)
    for i in range(iterNum):
        error = sigmoid(X.dot(theta)) - k
        temp = X.transpose().dot(error)
        minusPart = temp*alpha/m
        theta = theta - minusPart
        J_history[i] = CostFunction(theta,X,y)
    plotHistory(J_history)
    return theta

'''
    History of Cost during Gradient Descend
'''
def plotHistory(history):
    m = len(history)
    x = np.arange(m)
    plt.scatter(x,history,linewidths=0.1)
    plt.plot(x,history,linewidth = 0.05)
    plt.show()
  
if __name__=='__main__':
    data = loadData('ex2data1.txt')
    x = data[:,0:-1]
    y = data[:,-1]
    m = y.size
    X = np.hstack((np.ones([m,1]),x))
    plotData(x,y).show()#画出给定数据集
    theta = np.array([[-25],[0.4],[0.1]])
    theta = GradientDescend(theta,X,y,iterNum=1000,alpha=0.001)
    plotData(x,predict(theta,X)).show()#画出预测数据集
    plotBoudary(theta,X)
    print CostFunction(theta,X,y)
    