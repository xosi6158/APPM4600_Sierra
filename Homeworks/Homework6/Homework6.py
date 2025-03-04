# Homework 6

# Question 1 

import numpy as np
import math
from numpy.linalg import inv, norm



def F(x):
    F = np.zeros(2)
    F[0] = x[0]**2 + x[1]**2 - 4
    F[1] = np.exp(x[0]) + x[1] - 1
    return F

def J(x):
    J = np.array([
        [2 * x[0], 2 * x[1]],
        [np.exp(x[0]), 1]
    ])
    return J


def Newton(x0, tol, Nmax):
    for its in range(Nmax):
        Jn = J(x0)
        Jinv = inv(Jn)
        Fn = F(x0)
        x1 = x0 - Jinv.dot(Fn)
        if norm(x1 - x0) < tol:
            return x1, 0, its
        x0 = x1
    return x0, 1, Nmax


def LazyNewton(x0, tol, Nmax):
    Jn = J(x0)
    Jinv = inv(Jn)
    for its in range(Nmax):
        Fn = F(x0)
        x1 = x0 - Jinv.dot(Fn)
        if norm(x1 - x0) < tol:
            return x1, 0, its
        x0 = x1
    return x0, 1, Nmax

def Broyden(x0, tol, Nmax):
    A0 = J(x0)
    v = F(x0)
    try:
        A = inv(A0)
    except LinAlgError:
        print("Jacobian is singular at the initial guess. Skipping.")
        return x0, 1, 0

    s = -A.dot(v)
    xk = x0 + s

    for its in range(Nmax):
        w = v
        v = F(xk)
        y = v - w
        z = -A.dot(y)
        p = -np.dot(s, z)
        u = np.dot(s, A)
        tmp = s + z
        tmp2 = np.outer(tmp, u)
        A = A + (1.0 / p) * tmp2
        s = -A.dot(v)
        xk = xk + s
        if norm(s) < tol:
            return xk, 0, its
    return xk, 1, Nmax


def driver():
    tol = 1e-6
    Nmax = 100

    guesses = [
        np.array([1.0, 1.0]),
        np.array([1.0, -1.0]),
        np.array([0.0, 0.0])
    ]

    for i, guess in enumerate(guesses):
        print(f"Initial guess {i + 1}: {guess}")
        
        for method, name in [(Newton, "Newton"), (LazyNewton, "Lazy Newton"), (Broyden, "Broyden")]:
            xstar, ier, its = method(guess, tol, Nmax)
        

            print(f"{name}: Root: {xstar}")
            print(f"{name}: Iterations: {its}")


if __name__ == '__main__':
    driver()



