# Pre-Lab

import numpy as np

def eval_composite_trap(M, a, b, f):
    # Step size (h)
    h = (b - a) / M
    # Generate x values
    x_values = np.linspace(a, b, M + 1)
    # Apply the trapezoidal rule
    integral = (f(a) + f(b)) / 2
    integral += np.sum(f(x_values[1:-1]))
    integral *= h
    return integral, x_values, None

# 3. Composite Simpson's Rule
def eval_composite_simpsons(M, a, b, f):
    # Ensure M is even for Simpson's Rule
    if M % 2 != 0:
        M += 1  # Make M even if it's odd

    # Step size (h)
    h = (b - a) / M
    # Generate x values
    x_values = np.linspace(a, b, M + 1)
    # Apply Simpson's rule
    integral = f(a) + f(b)
    for i in range(1, M, 2):  # Odd indices (multiplied by 4)
        integral += 4 * f(x_values[i])
    for i in range(2, M, 2):  # Even indices (multiplied by 2)
        integral += 2 * f(x_values[i])
    integral *= h / 3
    return integral, x_values, None

def lgwt(N, a, b):
    """
    Generate the Legendre-Gauss nodes and weights for a given number of points N.
    a and b are the limits of integration.
    Returns the nodes (x) and weights (w) for Gauss-Legendre quadrature.
    """
    x = np.cos(np.pi * (4 * np.arange(1, N + 1) - 1) / (4 * N + 2))
    P0 = np.zeros(N)
    P1 = np.ones(N)

    # Iterate to find the roots
    for k in range(0, 100):
        P2 = ((2 * k + 1) * x * P1 - k * P0) / (k + 1)
        P0, P1 = P1, P2
        x = x - P2 / P1

    # Calculate the weights
    w = 2 / ((1 - x ** 2) * P1 ** 2)
    # Transform to the desired interval [a, b]
    x = 0.5 * (x + 1) * (b - a) + a
    w = 0.5 * (b - a) * w

    return x, w

# 2. Composite Trapezoidal Rule
def eval_composite_trap(M, a, b, f):
    # Step size (h)
    h = (b - a) / M
    # Generate x values
    x_values = np.linspace(a, b, M + 1)
    # Apply the trapezoidal rule
    integral = (f(a) + f(b)) / 2
    integral += np.sum(f(x_values[1:-1]))
    integral *= h
    return integral, x_values, None

# 3. Composite Simpson's Rule
def eval_composite_simpsons(M, a, b, f):
    # Ensure M is even for Simpson's Rule
    if M % 2 != 0:
        M += 1  # Make M even if it's odd

    # Step size (h)
    h = (b - a) / M
    # Generate x values
    x_values = np.linspace(a, b, M + 1)
    # Apply Simpson's rule
    integral = f(a) + f(b)
    for i in range(1, M, 2):  # Odd indices (multiplied by 4)
        integral += 4 * f(x_values[i])
    for i in range(2, M, 2):  # Even indices (multiplied by 2)
        integral += 2 * f(x_values[i])
    integral *= h / 3
    return integral, x_values, None

# 4. Gaussian Quadrature
def eval_gauss_quad(M, a, b, f):
    # Use Gauss-Legendre quadrature nodes and weights
    x, w = lgwt(M, a, b)
    I_hat = np.sum(f(x) * w)
    return I_hat, x, w

# 5. Adaptive Quadrature
def adaptive_quad(a, b, f, tol, M, method):
    """
    Adaptive numerical integrator for \int_a^b f(x)dx
    Input:
    a,b - interval [a,b]
    f - function to integrate
    tol - absolute accuracy goal
    M - number of quadrature nodes per bisected interval
    method - function handle for integrating on subinterval
    - eg) eval_gauss_quad, eval_composite_simpsons, etc.
    
    Output:
    I - the approximate integral
    X - final adapted grid nodes
    nsplit - number of interval splits
    """
    # 1/2^50 ~ 1e-15
    maxit = 50
    left_p = np.zeros((maxit,))
    right_p = np.zeros((maxit,))
    s = np.zeros((maxit, 1))
    left_p[0] = a
    right_p[0] = b
    # initial approx and grid
    s[0], x, _ = method(M, a, b, f)
    # save grid
    X = []
    X.append(x)
    j = 1
    I = 0
    nsplit = 1
    while j < maxit:
        # get midpoint to split interval into left and right
        c = 0.5 * (left_p[j - 1] + right_p[j - 1])
        # compute integral on left and right split intervals
        s1, x1, _ = method(M, left_p[j - 1], c, f)
        X.append(x1)
        s2, x2, _ = method(M, c, right_p[j - 1], f)
        X.append(x2)
        if np.max(np.abs(s1 + s2 - s[j - 1])) > tol:
            left_p[j] = left_p[j - 1]
            right_p[j] = 0.5 * (left_p[j - 1] + right_p[j - 1])
            s[j] = s1
            left_p[j - 1] = 0.5 * (left_p[j - 1] + right_p[j - 1])
            s[j - 1] = s2
            j = j + 1
        else:
            I = I + s1 + s2
            j = j - 1
        if j == 0:
            j = maxit
    return I, np.unique(X), nsplit

# 6. Testing with the Example

# Define the function to be integrated
def f(x):
    return np.sin(1 / x)

# Set parameters
a = 0.1  # Lower bound
b = 2    # Upper bound
tol = 1e-3  # Desired accuracy
M = 5  # Number of quadrature nodes per subinterval

# Use the adaptive quadrature with different methods
trap_result, trap_nodes, trap_splits = adaptive_quad(a, b, f, tol, M, eval_composite_trap)
simpson_result, simpson_nodes, simpson_splits = adaptive_quad(a, b, f, tol, M, eval_composite_simpsons)
gauss_result, gauss_nodes, gauss_splits = adaptive_quad(a, b, f, tol, M, eval_gauss_quad)

# Print results
print(f"Composite Trapezoidal Result: {trap_result}, Intervals: {trap_splits}")
print(f"Composite Simpson's Result: {simpson_result}, Intervals: {simpson_splits}")
print(f"Gaussian Quadrature Result: {gauss_result}, Intervals: {gauss_splits}")
