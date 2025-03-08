# Homework 7

import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Question 1b


def driver(): 
    f = lambda x: 1 / (1 + (100 * x**2))

   # Change N to see different interpolation points
   # N = 2,5,15, and 20
   # I also changed p(x) = 100 instead of 10 to look at behavior better 
    N = 2
    a = -1
    b = 1
    
    ''' Create interpolation nodes'''
    xint = np.linspace(a, b, N + 1)
    '''Create interpolation data'''
    yint = f(xint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint, N)

    ''' Compute the condition number of the Vandermonde matrix'''
    cond_V = condition_number(V)
    print(f"The condition number of the Vandermonde Matrix: {cond_V}")

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
    
    ''' Apply inverse to rhs to create the coefficients'''
    coef = Vinv @ yint

    # Validate the code
    Neval = 100    
    xeval = np.linspace(a, b, Neval + 1)
    yeval = eval_monomial(xeval, coef, N, Neval)

    # Exact function
    yex = f(xeval)
    err = abs(yex - yeval)
    
    plt.figure()    
    plt.plot(xeval, yeval, 'ro-', label='Interpolation')
    plt.plot(xeval, yex, 'b-', label='Exact Function')
    plt.legend()
    plt.title(f"Polynomial Interpolation for N={N}")
    plt.grid(True)
    plt.show()

    plt.figure() 
    plt.semilogy(xeval, err, 'ro--', label='Error (monomial)')
    plt.legend()
    plt.grid(True)
    plt.show()

def eval_monomial(xeval, coef, N, Neval):
    yeval = coef[0] * np.ones(Neval + 1)
    
    for j in range(1, N + 1):
        for i in range(Neval + 1):
            yeval[i] += coef[j] * xeval[i]**j
    
    return yeval

def Vandermonde(xint, N):
    V = np.zeros((N + 1, N + 1))
    
    ''' fill the first column'''
    for j in range(N + 1):
        V[j][0] = 1.0

    for i in range(1, N + 1):
        for j in range(N + 1):
            V[j][i] = xint[j]**i

    return V     

def condition_number(V):
    return np.linalg.cond(V)

driver()



# Question 2

# Will be looking at the same N points as in Question 1 

def f(x):
    return 1/(1+(10*x)**2)

def weight(x_interp):
    N = len(x_interp)
    w = np.ones(N)
    for j in range(N):
        prod = 1
        for i in range(N):
            if i != j:
                prod *= (x_interp[j] - x_interp[i])
        w[j] = 1 / prod
    return w

def phi_func(x_interp, x_eval):
    prod = np.ones_like(x_eval)
    for i in range(len(x_interp)):
        prod *= x_eval - x_interp[i]
    return prod

def barycentric(x_interp, y_interp, w, x_eval):
    sum = 0
    phi = phi_func(x_interp, x_eval)
    for j in range(len(x_interp)):
        sum += ((w[j] / (x_eval - x_interp[j])) * y_interp[j])
    return phi*sum

for N in range(2, 21):  
    x_interp = np.linspace(-1, 1, N)
    y_interp = f(x_interp)

    w = weight(x_interp)

    x_vals = np.linspace(-1, 1, 1001)
    y_vals = f(x_vals)
    y_bary = barycentric(x_interp, y_interp, w, x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'k-', label="f(x)")
    plt.plot(x_vals, y_bary, '--', label=f"Barycentric Interpolation (N={N})")
    plt.legend()
    plt.title(f"Barycentric Interpolation for N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    max_val = np.max(np.abs(y_bary))
    print(f"Max value for N={N}: {max_val}")

    if max_val > 100:
        break

# Question 3

def f(x):
    return 1/(1+(10*x)**2)

def weight(x_interp):
    N = len(x_interp)
    w = np.ones(N)
    for j in range(N):
        prod = 1
        for i in range(N):
            if i != j:
                prod *= (x_interp[j] - x_interp[i])
        w[j] = 1 / prod
    return w

def phi_func(x_interp, x_eval):
    prod = np.ones_like(x_eval)
    for i in range(len(x_interp)):
        prod *= x_eval - x_interp[i]
    return prod

def barycentric(x_interp, y_interp, w, x_eval):
    sum = 0
    phi = phi_func(x_interp, x_eval)
    for j in range(len(x_interp)):
        sum += ((w[j] / (x_eval - x_interp[j])) * y_interp[j])
    return phi*sum

# Look at N for N = 2, 15, 25, and 50 bc behavior changes 
# Had to look at the different Ns because behavior was different so N better
# Better one by one because 2 to 50 was too many graphs

for N in range(2, 3):  
    x_cheb = np.cos((2 * np.arange(1, N+1) - 1) * np.pi / (2 * N))
    y_cheb = f(x_cheb)

    w = weight(x_cheb)

    x_vals = np.linspace(-1, 1, 1001)
    y_vals = f(x_vals)
    y_bary = barycentric(x_cheb, y_cheb, w, x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'k-', label="f(x)")
    plt.plot(x_vals, y_bary, '--', label=f"Barycentric Interpolation with Chebyshev Nodes(N={N})")
    plt.legend()
    plt.title(f"Barycentric/Chebyshev Interpolation for N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    max_val = np.max(np.abs(y_bary))
    print(f"Max value for N={N}: {max_val}")

    if max_val > 100:
        break

for N in range(15, 16):  
    x_cheb = np.cos((2 * np.arange(1, N+1) - 1) * np.pi / (2 * N))
    y_cheb = f(x_cheb)

    w = weight(x_cheb)

    x_vals = np.linspace(-1, 1, 1001)
    y_vals = f(x_vals)
    y_bary = barycentric(x_cheb, y_cheb, w, x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'k-', label="f(x)")
    plt.plot(x_vals, y_bary, '--', label=f"Barycentric Interpolation with Chebyshev Nodes(N={N})")
    plt.legend()
    plt.title(f"Barycentric/Chebyshev Interpolation for N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    max_val = np.max(np.abs(y_bary))
    print(f"Max value for N={N}: {max_val}")

    if max_val > 100:
        break

for N in range(25, 26):  
    x_cheb = np.cos((2 * np.arange(1, N+1) - 1) * np.pi / (2 * N))
    y_cheb = f(x_cheb)

    w = weight(x_cheb)

    x_vals = np.linspace(-1, 1, 1001)
    y_vals = f(x_vals)
    y_bary = barycentric(x_cheb, y_cheb, w, x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'k-', label="f(x)")
    plt.plot(x_vals, y_bary, '--', label=f"Barycentric Interpolation with Chebyshev Nodes(N={N})")
    plt.legend()
    plt.title(f"Barycentric/Chebyshev Interpolation for N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    max_val = np.max(np.abs(y_bary))
    print(f"Max value for N={N}: {max_val}")

    if max_val > 100:
        break


for N in range(50, 51):  
    x_cheb = np.cos((2 * np.arange(1, N+1) - 1) * np.pi / (2 * N))
    y_cheb = f(x_cheb)

    w = weight(x_cheb)

    x_vals = np.linspace(-1, 1, 1001)
    y_vals = f(x_vals)
    y_bary = barycentric(x_cheb, y_cheb, w, x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'k-', label="f(x)")
    plt.plot(x_vals, y_bary, '--', label=f"Barycentric Interpolation with Chebyshev Nodes(N={N})")
    plt.legend()
    plt.title(f"Barycentric/Chebyshev Interpolation for N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    max_val = np.max(np.abs(y_bary))
    print(f"Max value for N={N}: {max_val}")

    if max_val > 100:
        break


    