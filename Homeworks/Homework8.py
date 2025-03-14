# Homework 8

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.interpolate import CubicSpline


# Question 1

# Lagrange and Hermite

def driver():
    f = lambda x: 1 / (1 + x**2)
    fder = lambda x: -2 * x / (1 + x**2)**2
    
    ''' Set values of N to loop over '''
    N_values = [5, 10, 15, 20]
    
    ''' interval '''
    a = -5
    b = 5
    
    ''' create points for evaluating the Lagrange and Hermite interpolating polynomials'''
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    
    for N in N_values:
        ''' create equispaced interpolation nodes '''
        xint = np.linspace(a, b, N + 1)
        
        ''' create interpolation data '''
        yint = np.array([f(x) for x in xint])
        ypint = np.array([fder(x) for x in xint])
        
        ''' Evaluate interpolations '''
        yevalL = np.array([eval_lagrange(x, xint, yint, N) for x in xeval])
        yevalH = np.array([eval_hermite(x, xint, yint, ypint, N) for x in xeval])
        
        ''' create vector with exact values '''
        fex = np.array([f(x) for x in xeval])
        
        ''' Plotting the functions '''
        plt.figure()
        plt.plot(xeval, fex, 'ro-', label='True Function')
        plt.plot(xeval, yevalL, 'bs--', label='Lagrange') 
        plt.plot(xeval, yevalH, 'c.--', label='Hermite')
        plt.title(f'Comparison for N = {N}')
        plt.semilogy()
        plt.legend()
        plt.show()

        ''' Plotting errors '''
        plt.figure() 
        error_L = np.abs(yevalL - fex)
        error_H = np.abs(yevalH - fex)
        plt.semilogy(xeval, error_L, 'ro--', label='Lagrange Error')
        plt.semilogy(xeval, error_H, 'bs--', label='Hermite Error')
        plt.title(f'Error of Langrange vs Hermite for N = {N}')
        plt.legend()
        plt.show()

def eval_hermite(xeval, xint, yint, ypint, N):
    ''' Evaluates Hermite polynomial at xeval using interpolation nodes, values, and derivatives '''
    lj = np.ones(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lj[count] *= (xeval - xint[jj]) / (xint[count] - xint[jj])

    ''' Construct the l_j'(x_j) '''
    lpj = np.zeros(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lpj[count] += 1. / (xint[count] - xint[jj])

    yeval = 0.
    for jj in range(N + 1):
        Qj = (1. - 2. * (xeval - xint[jj]) * lpj[jj]) * lj[jj]**2
        Rj = (xeval - xint[jj]) * lj[jj]**2
        yeval += yint[jj] * Qj + ypint[jj] * Rj

    return yeval

def eval_lagrange(xeval, xint, yint, N):
    ''' Evaluates Lagrange polynomial at xeval using interpolation nodes and values '''
    lj = np.ones(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lj[count] *= (xeval - xint[jj]) / (xint[count] - xint[jj])

    yeval = 0
    for jj in range(N + 1):
        yeval += yint[jj] * lj[jj]
  
    return yeval
       
if __name__ == '__main__':
    driver()

# Natural and Clamped Cubic Splines

f = lambda x: 1/(1+x**2)
fder = lambda x: -2*x/(1+x**2)**2
a, b = -5, 5
n = [5, 10, 15, 20]  
x_vals = np.linspace(a, b, 1000)
y_vals = f(x_vals)

plt.figure(figsize=(10, 8))
for i, w in enumerate(n):
    x_nodes = np.linspace(a, b, w+1)
    y_nodes = f(x_nodes)
    dy_nodes = fder(x_nodes)

    # Natural cubic spline
    natural_spline = CubicSpline(x_nodes, y_nodes, bc_type=((2,0),(2,0)))
    y_natural = natural_spline(x_vals)

    # Clamped cubic spline
    clamped_spline = CubicSpline(x_nodes, y_nodes, bc_type=((1, fder(x_nodes[0])), (1, fder(x_nodes[-1]))))
    y_clamped = clamped_spline(x_vals)
    
    # Plot spline results
    plt.subplot(2, 2, i + 1)
    plt.plot(x_vals, y_vals, 'k-', label='True Function')
    plt.plot(x_vals, y_natural, 'r--', label='Natural Cubic Spline')
    plt.plot(x_vals, y_clamped, 'b-.', label='Clamped Cubic Spline')
    plt.title(f'Comparison for n={w}')
    plt.legend()

plt.tight_layout()
plt.show()

# Second figure for error plots
plt.figure(figsize=(10, 8))
for i, w in enumerate(n):
    x_nodes = np.linspace(a, b, w+1)
    y_nodes = f(x_nodes)
    dy_nodes = fder(x_nodes)
    natural_spline = CubicSpline(x_nodes, y_nodes, bc_type=((2,0),(2,0)))
    y_natural = natural_spline(x_vals)
    clamped_spline = CubicSpline(x_nodes, y_nodes, bc_type=((1, fder(x_nodes[0])), (1, fder(x_nodes[-1]))))
    y_clamped = clamped_spline(x_vals)
    
    error_N = abs(y_natural - y_vals)
    error_C = abs(y_clamped - y_vals)
    
    # Plot error results
    plt.subplot(2, 2, i + 1)
    plt.semilogy(x_vals, error_N, 'ro--', label='Natural')
    plt.semilogy(x_vals, error_C, 'bs--', label='Clamped')
    plt.title(f'Error of Natural vs Clamped Splines for n={w}')
    plt.legend()

plt.tight_layout()
plt.show()


# Question 2

# Lagrange and Hermite with Chebvychev nodes


def driver():
    f = lambda x: 1 / (1 + x**2)
    fder = lambda x: -2 * x / (1 + x**2)**2
    a, b = -5, 5
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)

    for N in [5, 10, 15, 20]:
        # Using Chebychev nodes
        x_cheb = [0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * j + 1) * np.pi / (2 * N)) for j in range(N)]
        y_cheb = [f(x) for x in x_cheb]
        yp_cheb = [fder(x) for x in x_cheb]

        yevalL = np.array([eval_lagrange(x, x_cheb, y_cheb, N - 1) for x in xeval])
        yevalH = np.array([eval_hermite(x, x_cheb, y_cheb, yp_cheb, N - 1) for x in xeval])
        fex = np.array([f(x) for x in xeval])

        # Plotting the true function and approximations
        plt.figure()
        plt.plot(xeval, fex, 'ro-', label='True Function')
        plt.plot(xeval, yevalL, 'bs--', label='Lagrange')
        plt.plot(xeval, yevalH, 'c.--', label='Hermite')
        plt.title(f'ChebyChev Comparison for n={N}')
        plt.semilogy()
        plt.legend()
        plt.show()

        # Error plotting
        err_LC = abs(yevalL - fex)
        err_HC = abs(yevalH - fex)
        plt.figure()
        plt.semilogy(xeval, err_LC, 'ro--', label='Lagrange Error')
        plt.semilogy(xeval, err_HC, 'bs--', label='Hermite Error')
        plt.title(f'Error of Lagrange vs Hermite with ChebyChev Nodes for N = {N}')
        plt.legend()
        plt.show()

def eval_hermite(xeval, xint, yint, ypint, N):
    lj = np.ones(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lj[count] *= (xeval - xint[jj]) / (xint[count] - xint[jj])
    lpj = np.zeros(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lpj[count] += 1. / (xint[count] - xint[jj])
    yeval = 0.
    for jj in range(N + 1):
        Qj = (1. - 2. * (xeval - xint[jj]) * lpj[jj]) * lj[jj]**2
        Rj = (xeval - xint[jj]) * lj[jj]**2
        yeval += yint[jj] * Qj + ypint[jj] * Rj
    return yeval

def eval_lagrange(xeval, xint, yint, N):
    lj = np.ones(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if jj != count:
                lj[count] *= (xeval - xint[jj]) / (xint[count] - xint[jj])
    yeval = 0
    for jj in range(N + 1):
        yeval += yint[jj] * lj[jj]
    return yeval

if __name__ == '__main__':
    driver()


# Natural and Clamped with ChebyChev Nodes

f = lambda x: 1 / (1 + x**2)
fder = lambda x: -2 * x / (1 + x**2)**2
a, b = -5, 5
n = [5, 10, 15, 20]  
x_vals = np.linspace(a, b, 1000)
y_vals = f(x_vals)

# Generate plots for each n
plt.figure(figsize=(10, 8))
for i, w in enumerate(n):
    # Calculate Chebychev nodes
    x_chebnodes = np.zeros(w)
    for j in range(w):
        x_chebnodes[j] = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2*j + 1) * np.pi / (2 * w))
    
    x_chebnodes = np.sort(x_chebnodes)
   
    y_chebnodes = f(x_chebnodes)
    dy_nodes = fder(x_chebnodes)

    # Natural cubic spline
    natural_spline = CubicSpline(x_chebnodes, y_chebnodes, bc_type='natural')
    y_natural = natural_spline(x_vals)

    # Clamped cubic spline (assuming derivative at endpoints known)
    clamped_spline = CubicSpline(x_chebnodes, y_chebnodes, bc_type=((1, fder(x_chebnodes[0])), (1, fder(x_chebnodes[-1]))))
    y_clamped = clamped_spline(x_vals)
    
    # Plot results
    plt.subplot(2, 2, i + 1)
    plt.plot(x_vals, y_vals, 'k-', label='True Function')
    plt.plot(x_vals, y_natural, 'r--', label='Natural Cubic Spline')
    plt.plot(x_vals, y_clamped, 'b-.', label='Clamped Cubic Spline')
    plt.title(f'Chebychev Comparison for n={w}')
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
for i, w in enumerate(n):
    x_chebnodes = np.zeros(w)
    for j in range(w):
        x_chebnodes[j] = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2*j + 1) * np.pi / (2 * w))
    
    x_chebnodes = np.sort(x_chebnodes)
   
    y_chebnodes = f(x_chebnodes)
    dy_chebnodes = fder(x_chebnodes)
    
    natural_spline = CubicSpline(x_chebnodes, y_chebnodes, bc_type=((2,0),(2,0)))
    y_natural = natural_spline(x_vals)
    clamped_spline = CubicSpline(x_chebnodes, y_chebnodes, bc_type=((1, fder(x_chebnodes[0])), (1, fder(x_chebnodes[-1]))))
    y_clamped = clamped_spline(x_vals)
    
    error_NC = abs(y_natural - y_vals)
    error_CC = abs(y_clamped - y_vals)
    
    # Plot error results
    plt.subplot(2, 2, i + 1)
    plt.semilogy(x_vals, error_NC, 'ro--', label='Natural')
    plt.semilogy(x_vals, error_CC, 'bs--', label='Clamped')
    plt.title(f'Chebychev Error of Natural vs Clamped Splines for n={w}')
    plt.legend()

plt.tight_layout()
plt.show()



