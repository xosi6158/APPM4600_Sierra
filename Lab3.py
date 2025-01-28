# import libraries
import numpy as np

def driver():
# use routines
    f = lambda x: (x**2)*(x-1)
    a = 0.5
    b = 2
# f = lambda x: np.sin(x)
# a = 0.1
# b = np.pi+0.1
    tol = 1e-7
    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
# define routines
def bisection(f,a,b,tol):
# Inputs:
# f,a,b - function and endpoints of initial interval
# tol - bisection stops when interval length < tol
# Returns:
# astar - approximation of root
# ier - error 