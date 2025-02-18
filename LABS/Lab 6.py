# Lab 6

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

# Before Lab

f = np.cos
x = (np.pi)/2
h = 0.01 * 2. ** (-np.arange(0, 10))

# Forward Difference

for_diff = (f(x+h) - f(x)) / h
cent_diff = (f(x+h) - f(x-h)) / (2*h)
f_prime = -np.sin(x)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(h, for_diff, label='Forward Difference', marker='o')
plt.plot(h, cent_diff, label='Centered Difference', marker='x')
plt.axhline(y=f_prime, color='r', linestyle='--', label='Exact Derivative')
plt.xscale('log')
plt.xlabel('h')
plt.ylabel('Derivative Approximation')
plt.title('Finite Difference Approximations of Derivative of cos(x) at x = pi/2')
plt.legend()
plt.grid(True)
plt.show()

# Display the results as text
for i in range(len(h)):
    print(f"h: {h[i]:.10f}, Forward: {for_diff[i]:.10f}, Centered: {cent_diff[i]:.10f}, Exact: {f_prime:.10f}")

# Second Order Approximation

# During Lab

# Lazy Newton's Code with conditions

#First, we define F(x) and its Jacobian.
def F(x):
    x1, x2 = x[0], x[1]
    return np.array([
        4 * x1**2 + x2**2 - 4, x1 + x2 - np.sin(x1 - x2)])
def JF(x):
    x1, x2 = x[0], x[1]
    return np.array([[8 * x1, 2 * x2],[1 - np.cos(x1 - x2), 1 + np.cos(x1 - x2)]])

# Apply Lazy Newton (chord iteration)


def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):
# Initialize arrays and function value
    xn = x0 #initial guess
    rn = x0 #list of iterates
    Fn = f(xn) #function value vector
# compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn)
# Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn)
    n=0
    nf=1
    nJ=1 #function and Jacobian evals
    npn=1

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|")
    every_five = 1
    while npn>tol and n<=nmax:

        every_five = every_five+1
        if (every_five%5==0):
            Jn = Jf(xn)
            lu, piv = lu_factor(Jn)

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)))
# Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn);
        xn = xn + pn
        npn = np.linalg.norm(pn) #size of Newton step

        n+=1
        rn = np.vstack((rn,xn))
        Fn = f(xn)
        nf+=1
    r=xn
    
    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" %(nmax,np.linalg.norm(Fn)));
    else:
        print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" %(n,np.linalg.norm(Fn)))

    return (r,rn,nf,nJ)

nmax=1000
x0 = np.array([1.0,1.0])
tol=1e-10
(rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F,JF,x0,tol,nmax,True)
print(rLN)
print(rnLN)
print(nfLN)
print(nJLN)

def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):
# Initialize arrays and function value
    xn = x0 #initial guess
    rn = x0 #list of iterates
    Fn = f(xn) #function value vector
# compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn)
# Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn)
    n=0
    nf=1
    nJ=1 #function and Jacobian evals
    npn=1

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|")
    while npn>tol and n<=nmax:

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)))
# Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn);
        xn = xn + pn
        npn = np.linalg.norm(pn) #size of Newton step

        n+=1
        rn = np.vstack((rn,xn))
        Fn = f(xn)
        nf+=1
    r=xn
    
    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" %(nmax,np.linalg.norm(Fn)));
    else:
        print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" %(n,np.linalg.norm(Fn)))

    return (r,rn,nf,nJ)

nmax=1000
x0 = np.array([1.0,1.0])
tol=1e-10
(rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F,JF,x0,tol,nmax,True)
print(rLN)
print(rnLN)
print(nfLN)
print(nJLN)

# Goes from 19 iterations to 8 iterations so much faster when I have the jacobian being every 5 iterartions

