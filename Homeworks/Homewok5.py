# Homework 5

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import sympy as sp

# Question 1 Part A

# Define the iteration matrix M
M = np.array([[1/6, 1/18],
              [0, 1/6]])

# Define function F(x, y) = [f(x, y), g(x, y)]
def F(x):
    return np.array([3*x[0]**2 - x[1]**2, 3*x[0]*x[1]**2 - x[0]**3 - 1])

# Define the fixed-point function G(x) = x - M @ F(x)
def G(x):
    return x - M @ F(x)

# Define Jacobian of G (which is I - M*J, where J is the Jacobian of F)
def JF(x):
    return np.array([[6*x[0], -2*x[1]],
                     [3*x[1]**2 - 3*x[0]**2, 6*x[0]*x[1]]])

def JG(x):
    return np.eye(2) - M @ JF(x)

# Implement Fixed-Point Iteration
def fixed_point_method_nd(G, JG, x0, tol, nmax, verb=False):
    xn = x0  # Initial guess
    rn = [x0]  # Store iterates
    Gn = G(xn)  # Compute first step
    n = 0
    nf = 1  # Function evaluations

    while np.linalg.norm(Gn - xn) > tol and n <= nmax:
        if verb:
            rhoGn = np.max(np.abs(np.linalg.eigvals(JG(xn))))
            print(f"Iteration {n}: x = {xn}, |G(x) - x| = {np.linalg.norm(Gn - xn):.2e}, Spectral Radius = {rhoGn:.2f}")

        xn = Gn  # Update x
        rn.append(xn)  # Store iterate
        Gn = G(xn)  # Compute next step
        n += 1
        nf += 1

        if np.linalg.norm(xn) > 1e15:  # Prevent divergence
            n = nmax + 1
            nf = nmax + 1
            break

    if verb:
        if n >= nmax:
            print(f"Fixed-point iteration failed to converge, iterations = {nmax}, error = {np.linalg.norm(Gn - xn):.1e}\n")
        else:
            print(f"Fixed-point iteration converged, iterations = {n}, error = {np.linalg.norm(Gn - xn):.1e}\n")

    return np.array(xn), np.array(rn), n

# Set initial guess and tolerance
x0 = np.array([1.0, 1.0])
tol = 1e-6
nmax = 100

# Solve using Fixed-Point Iteration
solution, iterates, num_iterations = fixed_point_method_nd(G, JG, x0, tol, nmax, verb=True)

# Plot convergence
plt.plot(iterates[:, 0], iterates[:, 1], 'o-', label="Fixed-Point Iterates")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fixed-Point Iteration Convergence")
plt.legend()
plt.grid()
plt.show()

# Print final solution
print(f"Solution: x = {solution}")
print(f"Number of iterations: {num_iterations}")
print(f"Number of function evaluations: {num_iterations + 1}")

#Question 1 Part C

# did functions in different form
def F(X):
    x, y = X
    return np.array([
        3*x**2 - y**2,
        3*x*y**2 - x**3 - 1
    ])

def J(X):
    x, y = X
    return np.array([
        [6*x, -2*y],
        [3*y**2 - 3*x**2, 6*x*y]
    ])

def newton_method(F, J, x0, tol=1e-6, Nmax=100):
    X1 = np.array(x0, dtype=float)
    
    for i in range(Nmax):
        Jn = J(X1)
        Fn = F(X1)
        
        # From class 
        P = np.linalg.solve(Jn, -Fn)
        
        # Update estimate
        X1 = X1 + P

        # Check convergence
        if np.linalg.norm(P, ord=2) < tol:
            print(f"Converged in {i+1} iterations.")
            return X1
    
    print("Newton's method did not converge.")
    return None

# Initial guess
x0 = [1, 1]

# Run Newton's method
solution = newton_method(F, J, x0)

print(f"Approximate solution: x = {solution[0]:.6f}, y = {solution[1]:.6f}")

# Question 3 Part B
def f(x, y, z):
    return x**2 + 4*y**2 + 4*z**2 - 16

def grad_f(x, y, z):
    return np.array([2*x, 8*y, 8*z])

def newton(x0, y0, z0, tol=1e-10, Nmax=100):
    x, y, z = x0, y0, z0
    errors = []
    
    for _ in range(Nmax):
        fx = f(x, y, z)
        grad = grad_f(x, y, z)
        norm_sq = np.dot(grad, grad)
        
        if np.abs(fx) < tol:
            break     
        d = fx / norm_sq
        x_new = x - d * grad[0]
        y_new = y - d * grad[1]
        z_new = z - d * grad[2]

        # Erros that we know for convergence 
        error = np.abs(f(x_new, y_new, z_new))
        errors.append(error)
        
        x, y, z = x_new, y_new, z_new
    
    return x, y, z, errors

x_sol, y_sol, z_sol, errors = newton(1, 1, 1)

# Solution
print(f"Projected point: ({x_sol:.6f}, {y_sol:.6f}, {z_sol:.6f})")

ratios = [errors[i+1] / (errors[i]**2) for i in range(len(errors)-1)]

#quadratic convergence
for i in range(1, len(errors)):
    print(f"Iteration {i}: Error = {errors[i]:.2e}, Ratio = {errors[i] / errors[i-1]**2:.2e}")

