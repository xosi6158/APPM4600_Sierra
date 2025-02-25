# Lab 7
import numpy as np
import math
from numpy.linalg import inv
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.interpolate import lagrange




# Interpolation Code

def driver():
    f = lambda x: np.exp(x)
    N = 3
    # interval
    a = 0
    b = 1
   #create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
   #create interpolation data'''
    yint = f(xint)
   #create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
# Initialize and populate the first columns of the divided difference matrix. We will pass the x vector
    y = np.zeros( (N+1, N+1) )
    for j in range(N+1):
        y[j][0] = yint[j]
    y = dividedDiffTable(xint, y, N+1)
    # evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
        yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
 # create vector with exact values'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_l,'bs--')
    plt.plot(xeval,yeval_dd,'c.--')
    plt.legend()

    plt.figure()
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    plt.legend()
    plt.show()

def eval_lagrange(xeval,xint,yint,N):
    lj = np.ones(N+1)

    for count in range(N+1):
        for jj in range(N+1):
            if (jj != count):
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])
    yeval = 0.

    for jj in range(N+1):
        yeval = yeval + yint[jj]*lj[jj]
    return(yeval)

 # create divided difference matrix

def dividedDiffTable(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j]))
    return y;

def evalDDpoly(xval, xint,y,N):
# evaluate the polynomial terms 
    ptmp = np.zeros(N+1)

    ptmp[0] = 1.
    for j in range(N):
        ptmp[j+1] = ptmp[j]*(xval-xint[j])
    
    # evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
        yeval = yeval + y[0][j]*ptmp[j]

    return yeval

driver()

# Monomian Interpolation

def driver():
    f = lambda x: 2*x +4
    N = 10
    a = 0
    b = 1
 # Create interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    print('xint =',xint)
# Create interpolation data'''
    yint = f(xint)
    #print('yint =',yint)
# Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
    #print('V = ',V)

# Invert the Vandermonde matrix'''
    Vinv = inv(V)
    #print('Vinv = ' , Vinv)
# Apply inverse to rhs to create the coefficients'''

    coef = Vinv @ yint
    #print('coef = ', coef)


# No validate the code

    Neval = 100
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

# exact function
    yex = f(xeval)
    err = norm(yex-yeval)
    print('err = ', err)

    return

def eval_monomial(xeval,coef,N,Neval):
    yeval = coef[0]*np.ones(Neval+1)
    # print('yeval = ', yeval)
    for j in range(1,N+1):
        for i in range(Neval+1):

    # print('yeval[i] = ', yeval[i])
    #print('a[j] = ', a[j])
    # print('i = ', i)
    # print('xeval[i] = ', xeval[i])
            yeval[i] = yeval[i] + coef[j]*xeval[i]**j
    return yeval

def Vandermonde(xint,N):
    V = np.zeros((N+1,N+1))
 # fill the first column'''
    for j in range(N+1):
        V[j][0] = 1.0
    
    for i in range(1,N+1):
        for j in range(N+1):
            V[j][i] = xint[j]**i
    return V
driver()

# Pre Lab 

def interpolate_vandermonde(x_points, y_points):
    n = len(x_points)
    
    # Step 1: Construct the Vandermonde matrix
    V = np.vander(x_points, n, increasing=True)
    
    # Step 2: Solve the linear system for coefficients
    coefficients = np.linalg.solve(V, y_points)
    
    return coefficients


def evaluate_polynomial(coefficients, x):
    n = len(coefficients)
    y = sum(coefficients[i] * x**i for i in range(n))
    return y



# Exercise 3.1 1



def driver():
    def fun(x):
        return 1 / (1 + (10 * x) ** 2)

    nn = np.arange(5, 51, 5)
    y = np.linspace(-1, 1, 1000)

    # Monomial Expansion (Vandermonde)
    poly_interp_logerror(fun, nn, y)
    plt.ylim([-16, 3])
    plt.title('Vandermonde matrix interpolation log error')
    plt.show()

    # Lagrange Interpolation
    lagrange_interp_logerror(fun, nn, y)
    plt.ylim([-16, 3])
    plt.title('Lagrange interpolation log error')
    plt.show()

    # Newton Interpolation
    newton_interp_logerror(fun, nn, y)
    plt.ylim([-16, 3])
    plt.title('Newton interpolation log error')
    plt.show()


def poly_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    V = np.vander(xint, n + 1, increasing=True)
    c = np.linalg.solve(V, fi)
    return np.polyval(c[::-1], xtrg)


def poly_interp_logerror(f, nn, y):
    for n in nn:
        xi = np.linspace(-1, 1, n + 1)
        g1 = poly_interp(f, xi, y)
        plt.plot(y, np.log10(np.abs(f(y) - g1) + 1e-16), '-.', label='n =' + str(n))
    plt.legend()


def lagrange_interp_logerror(f, nn, y):
    for n in nn:
        xi = np.linspace(-1, 1, n + 1)
        poly = lagrange(xi, f(xi))
        g2 = poly(y)
        plt.plot(y, np.log10(np.abs(f(y) - g2) + 1e-16), '--', label='n =' + str(n))
    plt.legend()


def newton_interp_logerror(f, nn, y):
    for n in nn:
        xi = np.linspace(-1, 1, n + 1)
        g3, _ = newton_interp(f, xi, y)
        plt.plot(y, np.log10(np.abs(f(y) - g3) + 1e-16), ':', label='n =' + str(n))
    plt.legend()


def newton_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    D = np.zeros((n + 1, n + 1))
    D[:, 0] = fi
    for i in range(1, n + 1):
        D[i, :n + 1 - i] = (D[i - 1, 1:n + 2 - i] - D[i - 1, :n + 1 - i]) / (xint[i:n + 1] - xint[:n + 1 - i])
    cN = D[:, 0]
    g = cN[n] * np.ones(len(xtrg))
    for i in range(n - 1, -1, -1):
        g = g * (xtrg - xint[i]) + cN[i]
    return g, cN


if __name__ == "__main__":
    driver()

# Absolute Error & Approximation 

import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt


def driver():
    def fun(x):
        return 1 / (1 + (10 * x) ** 2)

    nn = np.arange(5, 51, 5)
    y = np.linspace(-1, 1, 1000)

    # Monomial Expansion (Vandermonde)
    plot_interpolation_and_error(fun, nn, y, 'Vandermonde', poly_interp)

    # Lagrange Interpolation
    plot_interpolation_and_error(fun, nn, y, 'Lagrange', lagrange_interp)

    # Newton Interpolation
    plot_interpolation_and_error(fun, nn, y, 'Newton', newton_interp)


def plot_interpolation_and_error(f, nn, y, method, interp_func):
    for n in nn:
        xi = np.linspace(-1, 1, n + 1)
        g = interp_func(f, xi, y)
        max_val = np.max(np.abs(g))
        max_error = np.max(np.abs(f(y) - g))

        # Print error information
        print(f"{method} Interpolation (n={n}): Max |p(x)| ≈ {max_val:.2f}, Max Absolute Error ≈ {max_error:.2e}")
        
        
        
        plt.figure(figsize=(12, 6))
        
        # Plot approximation
        plt.subplot(1, 2, 1)
        plt.plot(y, f(y), label='True function')
        plt.plot(y, g, label=f'{method} approximation (n={n})')
        plt.legend()
        plt.title(f'{method} Interpolation (n={n})')
        
        # Plot absolute error
        plt.subplot(1, 2, 2)
        plt.plot(y, np.abs(f(y) - g))
        plt.title(f'{method} Absolute Error (n={n})')
        plt.yscale('log')
        plt.show()


def poly_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    V = np.vander(xint, n + 1, increasing=True)
    c = np.linalg.solve(V, fi)
    return np.polyval(c[::-1], xtrg)


def lagrange_interp(f, xint, xtrg):
    poly = lagrange(xint, f(xint))
    return poly(xtrg)


def newton_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    D = np.zeros((n + 1, n + 1))
    D[:, 0] = fi
    for i in range(1, n + 1):
        D[i, :n + 1 - i] = (D[i - 1, 1:n + 2 - i] - D[i - 1, :n + 1 - i]) / (xint[i:n + 1] - xint[:n + 1 - i])
    cN = D[:, 0]
    g = cN[n] * np.ones(len(xtrg))
    for i in range(n - 1, -1, -1):
        g = g * (xtrg - xint[i]) + cN[i]
    return g


if __name__ == "__main__":
    driver()

# Tweak the code for N = 2,...10


def driver():
    def fun(x):
        return 1 / (1 + (10 * x) ** 2)

    nn = np.arange(2, 11)
    y = np.linspace(-1, 1, 1000)

    # Monomial Expansion (Vandermonde)
    plot_interpolation_and_error(fun, nn, y, 'Vandermonde', poly_interp)

    # Lagrange Interpolation
    plot_interpolation_and_error(fun, nn, y, 'Lagrange', lagrange_interp)

    # Newton Interpolation
    plot_interpolation_and_error(fun, nn, y, 'Newton', newton_interp)


def plot_interpolation_and_error(f, nn, y, method, interp_func):
    for n in nn:
        xi = np.linspace(-1, 1, n + 1)
        g = interp_func(f, xi, y)
        max_val = np.max(np.abs(g))
        max_error = np.max(np.abs(f(y) - g))

        # Print error information
        print(f"{method} Interpolation (n={n}): Max |p(x)| ≈ {max_val:.2f}, Max Absolute Error ≈ {max_error:.2e}")
        
        plt.figure(figsize=(12, 6))
        
        # Plot approximation
        plt.subplot(1, 2, 1)
        plt.plot(y, f(y), label='True function')
        plt.plot(y, g, label=f'{method} approximation (n={n})')
        plt.legend()
        plt.title(f'{method} Interpolation (n={n})')
        
        # Plot absolute error
        plt.subplot(1, 2, 2)
        plt.plot(y, np.abs(f(y) - g))
        plt.title(f'{method} Absolute Error (n={n})')
        plt.yscale('log')
        plt.show()


def poly_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    V = np.vander(xint, n + 1, increasing=True)
    c = np.linalg.solve(V, fi)
    return np.polyval(c[::-1], xtrg)


def lagrange_interp(f, xint, xtrg):
    poly = lagrange(xint, f(xint))
    return poly(xtrg)


def newton_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    D = np.zeros((n + 1, n + 1))
    D[:, 0] = fi
    for i in range(1, n + 1):
        D[i, :n + 1 - i] = (D[i - 1, 1:n + 2 - i] - D[i - 1, :n + 1 - i]) / (xint[i:n + 1] - xint[:n + 1 - i])
    cN = D[:, 0]
    g = cN[n] * np.ones(len(xtrg))
    for i in range(n - 1, -1, -1):
        g = g * (xtrg - xint[i]) + cN[i]
    return g


if __name__ == "__main__":
    driver()

# For N=11,....100

import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt


def driver():
    def fun(x):
        return 1 / (1 + (10 * x) ** 2)

    nn = np.arange(2, 21)
    y = np.linspace(-1, 1, 1000)

    # Monomial Expansion (Vandermonde)
    plot_interpolation_and_error(fun, nn, y, 'Vandermonde', poly_interp)

    # Lagrange Interpolation
    plot_interpolation_and_error(fun, nn, y, 'Lagrange', lagrange_interp)

    # Newton Interpolation
    plot_interpolation_and_error(fun, nn, y, 'Newton', newton_interp)


def plot_interpolation_and_error(f, nn, y, method, interp_func):
    for n in nn:
        xi = np.linspace(-1, 1, n + 1)
        g = interp_func(f, xi, y)
        max_val = np.max(np.abs(g))
        max_error = np.max(np.abs(f(y) - g))

        # Print error information
        print(f"{method} Interpolation (n={n}): Max |p(x)| ≈ {max_val:.2f}, Max Absolute Error ≈ {max_error:.2e}")
        
        plt.figure(figsize=(12, 6))
        
        # Plot approximation
        plt.subplot(1, 2, 1)
        plt.plot(y, f(y), label='True function')
        plt.plot(y, g, label=f'{method} approximation (n={n})')
        plt.legend()
        plt.title(f'{method} Interpolation (n={n}), max(|p(x)|) ≈ {max_val:.2f}')
        
        # Plot absolute error
        plt.subplot(1, 2, 2)
        plt.plot(y, np.abs(f(y) - g))
        plt.title(f'{method} Absolute Error (n={n})')
        plt.yscale('log')
        plt.show()

        # Stop if max value blows up
        if max_val >= 100:
            print(f"Stopping at n = {n} for {method} interpolation: max(|p(x)|) ≈ {max_val:.2f}")
            break


def poly_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    V = np.vander(xint, n + 1, increasing=True)
    c = np.linalg.solve(V, fi)
    return np.polyval(c[::-1], xtrg)


def lagrange_interp(f, xint, xtrg):
    poly = lagrange(xint, f(xint))
    return poly(xtrg)


def newton_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    D = np.zeros((n + 1, n + 1))
    D[:, 0] = fi
    for i in range(1, n + 1):
        D[i, :n + 1 - i] = (D[i - 1, 1:n + 2 - i] - D[i - 1, :n + 1 - i]) / (xint[i:n + 1] - xint[:n + 1 - i])
    cN = D[:, 0]
    g = cN[n] * np.ones(len(xtrg))
    for i in range(n - 1, -1, -1):
        g = g * (xtrg - xint[i]) + cN[i]
    return g


if __name__ == "__main__":
    driver()

# The results tell us that; Vandermonde and Lagrange: Good for small n but unstable for large n, while Newton is more reliable, even as  n grows.
# The largest errors occur at n =16, where the polynomial tries to "wiggle" more to fit the points, causing huge oscillations
