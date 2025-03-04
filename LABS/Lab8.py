#Lab8

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import solve

#PreLab

# Build a code for a line 

def line(x0, fx0, x1, fx1, a):
    
    # y = mx+b here x = a

    if x0==x1:
        raise ValueError("x0 and x1 cannot be the same")
    m = (fx1 - fx0) / (x1 - x0) 
    b = fx0 - m * x0

    return m * a + b

# During Lab

# 3.2 Exercise 

def driver():

    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1

    # create points you want to evaluate at
    Neval = 100
    xeval = np.linspace(a,b,Neval)

    # number of intervals
    Nint = 10

    # evaluate the linear spline
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)

    # evaluate f at the evaluation points
    fex = f(xeval)
    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
    plt.legend(["Exact", "Linear Spline"])
    plt.show()

    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.show()

def eval_lin_spline(xeval,Neval,a,b,f,Nint):

    #create the intervals for piecewise approximations

    xint = np.linspace(a,b,Nint+1)

    #create vector to store the evaluation of the linear splines
    yeval = np.zeros(Neval)
    
    for jint in range(Nint):

    # find indices of xeval in interval (xint(jint),xint(jint+1))
    #let ind denote the indices in the intervals

        atmp = xint[jint]
        btmp= xint[jint+1]
    
    # find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        n = len(xloc)
    
    # temporarily store your info for creating a line in the interval of interest
        fa = f(atmp)
        fb = f(btmp)
        yloc = np.zeros(len(xloc))
        for kk in range(n):
        #use your line evaluator to evaluate the spline at each location
            yloc[kk