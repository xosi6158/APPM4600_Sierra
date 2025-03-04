
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Question 2
A = np.array([[1,1],[1+1e-10,1-1e-10]]) * 1/2
Ainv = np.array([[1-1e10, 1e10], [1+1e10,-1e10]])

norm_A= np.linalg.norm(A,2)
norm_Ainv = np.linalg.norm(Ainv,2)
condition_num = norm_A * norm_Ainv  
   
print("norm of A: ", norm_A)
print('norm of Ainv: ', norm_Ainv)
print("condition number: ", condition_num)

# Question 3

# Part C

#Algorithm 1

def alg1(x):
    return (math.e**x) - 1


x = 9.999999995000000e-10
equals = 1e-9
print("Unstable Result:", alg1(x))

# Question 3

# Part D

x = 9.999999995000000e-10
equals = 1e-9

# Taylor Series 

def taylor_fun(x, tol=10e-16):
    term = x
    sum = term
    n = 2

    while abs(term) > tol:
        term = x**n / math.factorial(n)
        sum = sum + term
        n = n + 1   
        print(n)
    return sum
taylor = taylor_fun(x)

abs_error = abs(equals - taylor)
rel_error = abs_error / equals
print("Taylor Series Result:", taylor)
print("Relative Error:", rel_error)

# Question 4

# Part A

t = np.arange(0, np.pi + np.pi/30, np.pi/30)
y = np.cos(t)
S = np.sum(t*y)
print("The sum is: S=", S)

# Question 4

# Part B

R = 1.2
delta_r = 0.1
f = 15
p = 0
theta = np.linspace(0, 2*np.pi, 1000)
x = R*(1+delta_r*np.cos(f*theta + p))*np.cos(theta)
y = R*(1+delta_r*np.cos(f*theta + p))*np.sin(theta)
plt.figure(figsize=(8,8))
plt.plot(x, y)
plt.axis('equal')
plt.title('Parametric Curves')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


for i in range(1,11):
    R = i
    delta_r = 0.05
    f = 2+i
    p = random.uniform(0,2)
    theta = np.linspace(0,2*np.pi, 1000)
    x_loop = R*(1+delta_r*np.cos(f*theta + p))*np.cos(theta)
    y_loop = R*(1+delta_r*np.cos(f*theta + p))*np.sin(theta)
    plt.plot(x_loop, y_loop)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('10 Parametric Curves through for loop')
    plt.axis('equal')

plt.figure(figsize=(8,8))
plt.show()
















