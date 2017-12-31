# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 10:50:10 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
# 使用mplot3d 画三维图像
from mpl_toolkits.mplot3d import Axes3D

def loadData(filePath):
    return np.loadtxt(filePath,delimiter=',')

def plotData(x,y):
    plt.scatter(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Fig')
    return plt

def computeCost(theta,X,y):
    m = y.size
    return np.sum( (X.dot(theta) - y.reshape(-1,1))**2 )/float(2*m)

'''
    BGD algorithm to solve theta to minimize cost
'''
def GradientDescend(theta,X,y,iterNum=100,alpha=0.01):
    m = y.size
    J_history = np.zeros([iterNum,1])
    for i in range(iterNum):
        predictError = X.dot(theta) - y.reshape(-1,1)
        temp = np.transpose(predictError).dot(X)
        minusPart = alpha/m * np.transpose(temp)
        theta = theta - minusPart
        J_history[i] = computeCost(theta,X,y)
    return [theta,J_history]

def plotHistory(history):
    plt.title('cost during iteration')
    plt.plot(history)
    return plt

'''
    Normal Equation to solve theta to minimize cost
'''
def NormalEquation(X,y):
    M = np.matrix(X)
    K = M.T.dot(M)
    B = X.transpose().dot(y.reshape(-1,1))
    return np.array(K.I.dot(B))

def plotfit(theta,X,y):
    x = X[:,1:2]
    p = plotData(x,y)
    yp = X.dot(theta)
    p.plot(x,yp,'r-')
    return p

def predict(theta,X):
    return X.dot(theta)

if __name__=='__main__':
    filePath = 'E:\Workspace\pythonCode\MLrelated\LinearRegression\ex1data1.txt'
    data = loadData(filePath)
    x = data[:,:-1]
    y = data[:,-1]
    m = y.size
    X = np.hstack((np.ones([m,1]),x))
    # scatter of given data
    plotData(x,y).show()
    theta = np.array([[0],[0]])
    
    # use BGD solve theta
    result = GradientDescend(theta,X,y,iterNum=1500,alpha=0.005)
    theta = np.array(result[0])
    J_history = np.array(result[1])
    
    #use Normal Equation 
    #theta = NormalEquation(X,y)
    
    print 'solution found:\n\ttheta0:%.12f\n\ttheta1:%.12f' %(theta[0],theta[1])
    print 'Cost:',computeCost(theta,X,y)
    # cost during iteration
    plotHistory(J_history).show()
    plotfit(theta,X,y).show()
    
    # plot 3D figure of cost function
    # data preperation
    theta0_vals = np.linspace(-2,3,100)
    theta1_vals = np.linspace(-1,1,100)
    J_vals = np.zeros((theta0_vals.size,theta1_vals.size))
    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
            J_vals[i,j] = computeCost(np.transpose(np.array([theta0_vals[i],theta1_vals[j]])),X,y)
    # plot fig
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(theta0_vals,theta1_vals,J_vals)
    ax.set_xlabel('$\theta_0$')
    ax.set_ylabel('$\theta_1$')
    ax.set_zlabel('cost')
    ax.set_title('figure of cost function')
    # 改变三维视角 elev为仰角，azim为xy平面绕z轴的角度
    ax.view_init(elev=45.,azim=-45)
    plt.show()
