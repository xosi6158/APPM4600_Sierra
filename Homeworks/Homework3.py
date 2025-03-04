
# Problem 1 Part C

import numpy as np

def driver():

# use routines    
    f = lambda x: (2*x -1 - np.sin(x))
    a = 0
    b = 1

    tol = 1e-8

    [astar,ier, num_iterations] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('Total number of iterations:', num_iterations)

# define routines
def bisection(f,a,b,tol):
     

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, 0]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, 0]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, 0]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]
      
driver()               

# Problem 2 Part A

def driver():

# use routines    
    f = lambda x: (x-5)**9
    a = 4.82
    b = 5.2

    tol = 1e-4



    [astar,ier, num_iterations] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('Total number of iterations:', num_iterations)

# define routines
def bisection(f,a,b,tol):
     
    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, 0]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, 0]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, 0]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]
      
driver()                                

# Problem 2 Part B

def driver():

# use routines    
    f = lambda x: x**9 - 45*(x**8) + 900*(x**7) - 10500*(x**6) + 78750*(x**5) - 393750*(x**4) + 1312500*(x**3) - 2812500*(x**2) + 3515625*x - 1953125
    a = 4.82
    b = 5.2
    tol = 1e-4

    [astar,ier, num_iterations] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('Total number of iterations:', num_iterations)

# define routines
def bisection(f,a,b,tol):
     

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, 0]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, 0]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, 0]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]
      
driver()               

# Question 3 Part B
def driver():

# use routines    
    f = lambda x: x**3 + x - 4
    a = 1
    b = 4
    tol = 1e-3

    [astar,ier, count] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    print('Total number of iterations:', count)

# define routines
def bisection(f,a,b,tol):
     

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]

driver()


# Question 5 Part A
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x - 4 * np.sin(2*x) - 3

# Define x range
x = np.linspace(-10, 10, 1000)
y = f(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True)
plt.title('Plot of F(x)')
plt.xlabel('x')
plt.ylabel('f(x)')

# Count zero crossings
zero_crossings = x[np.where(np.diff(np.sign(y)))[0]]
print(f'Number of zero crossings (roots): {len(zero_crossings)}')
plt.show()

# Question 5 Part B

import numpy as np

# Define the function for fixed-point iteration
def g(x):
    return -np.sin(2*x) + (5/4)*x - 3/4  # Given function

# Fixed-point iteration routine
def fixedpt(f, x0, tol=1e-10, Nmax=1000):
    xn = x0
    for n in range(Nmax):
       xn1 = f(xn)

       if abs(xn1 - xn) < tol:
          return xn1, n
       xn = xn1
    return None, Nmax

x0 = 2
root, n = fixedpt(g, x0)

if root is None:
    print("Could not find a root")
else:
    print(f"Root: {root}")
    print(f"Number of iterations: {n}")


