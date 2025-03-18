# Lab9

# Pre-Lab

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
from scipy.integrate import quad

def eval_legendre(n, x):
    if n < 0:
        raise ValueError("Order n must be a non-negative integer")
    
    # Initialize the polynomial values array
    p = np.zeros(n + 1)
    
    # Base cases
    if n >= 0:
        p[0] = 1  # φ0(x) = 1
    if n >= 1:
        p[1] = x  # φ1(x) = x
    
    # Apply the recursion formula for higher orders
    for i in range(1, n):
        p[i + 1] = (1.0 / (i + 1)) * ((2 * i + 1) * x * p[i] - i * p[i - 1])
    
    return p


# Put it together

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

# Prelab code
def eval_legendre(n, x):
    if n < 0:
        raise ValueError("Order n must be a non-negative integer")
    
    p = np.zeros(n + 1)
    
    if n >= 0:
        p[0] = 1
    if n >= 1:
        p[1] = x
    
    for i in range(1, n):
        p[i + 1] = (1.0 / (i + 1)) * ((2 * i + 1) * x * p[i] - i * p[i - 1])
    
    return p

# def phi_j(x, j):
    # Let's think of x**j
    return x**j

# Subroutine to evaluate f(x) * φj(x) * w(x)
# def integrand_f_phi_w(x, j):
   # return f(x) * phi_j(x, j) * w(x)

# Subroutine to evaluate φj(x)^2 * w(x)
#def integrand_phi2_w(x, j):
    #return phi_j(x, j)**2 * w(x)

# Interval 
#a, b = -1,1
#j = 2 

# Compute the numerator integral
#numerator, _ = quad(integrand_f_phi_w, a, b, args=(j))

# Compute the denominator integral
#denominator, _ = quad(integrand_phi2_w, a, b, args=(j))

# Compute coefficient a_j
#a_j = numerator / denominator

#print(f"The coefficient a_j for j={j} is: {a_j}")

# one line would be:

#a_j_1 = quad(integrand_f_phi_w, a, b, args=(j)) / quad(integrand_phi2_w, a, b, args=(j))


# All Together

def eval_legendre_expansion(f, a, b, w, n, x):
    p = eval_legendre(n, x)
    pval = 0.0
    
    for j in range(0, n+1):
        # Function handles
        phi_j = lambda x: eval_legendre(j, x)[-1]
        phi_j_sq = lambda x: (eval_legendre(j, x)[-1]**2) * w(x)
        
        # Evaluate normalization
        norm_fac, _ = quad(phi_j_sq, a, b)
        
        # Function handle for phi_j(x) * f(x) * w(x) / norm_fac
        func_j = lambda x: phi_j(x) * f(x) * w(x) / norm_fac
        
        # Evaluate coefficient aj #fixed from my own
        aj, _ = quad(func_j, a, b)
        
        # Accumulate into pval
        pval += aj * p[j]
    
    return pval

def driver():
    f = lambda x: 1 / (1+(x**2))
    a = -1
    b = 1
    w = lambda x: 1.
    n = 2
    N = 1000
    xeval = np.linspace(a, b, N+1)
    pval = np.zeros(N+1)
    
    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f, a, b, w, n, xeval[kk])
    
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
    
    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, pval, 'bs--', label='Expansion')
    plt.legend()
    plt.show()
    
    err = abs(pval - fex)
    plt.semilogy(xeval, err, 'ro--', label='error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    driver()



