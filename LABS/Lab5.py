# Lab 5

# 3.1 Exercise 2
# Build your own Bisection code which terminates when the midpoint lies in the basin of convergence for Newtons method

################################################################################
# This python script presents examples regarding the bisection method for
# 1D nonlinear root-finding, as presented in class.
################################################################################
# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;

# First, we define our example function f(x)=x+cos(x)-3.
def fun(x):
    return x + np.cos(x)-3;

def derfun(x):
    return 1-np.sin(x);

def newton_method(f, df, x0, tol=1e-10, max_iter=50):
    """Performs Newton's method to check for convergence."""
    for _ in range(max_iter):
        fx = f(x0)
        dfx = df(x0)
        if abs(dfx) < 1e-12:  # Avoid division by zero
            return False
        x1 = x0 - fx / dfx
        if abs(x1 - x0) < tol:
            return True  # Converges to a root
        x0 = x1
    return False  # Doesn't converge within max_iter

def bisect_method(f, df, a,b,tol,nmax,vrb=False):
    an, bn, n = a, b, 0
    xn = (an + bn) / 2
    rn = np.array([xn])

    if f(a) * f(b) >= 0:
        print("\n Interval is inadequate, f(a)*f(b)>=0. Try again \n")
        return None

    if vrb:
        print("\n|--n--|--an--|--bn--|----xn----|-|bn-an|--|---|f(xn)|---|")

    while n <= nmax:
        if newton_method(f, df, xn):  # Check Newton's method convergence
            break  # Stop bisection if xn is in the basin of convergence

        if vrb:
            print("|--%d--|%1.4f|%1.4f|%1.8f|%1.8f|%1.8f|" % (n, an, bn, xn, bn-an, np.abs(f(xn))))

        if (bn - an) < 2 * tol:
            break

        if f(an) * f(xn) < 0:
            bn = xn
        else:
            an = xn

        n += 1
        xn = (an + bn) / 2
        rn = np.append(rn, xn)

    return xn, rn

# Example run
root, iterations = bisect_method(fun, derfun, 3, 4, 5e-16, 100, True)
print("Estimated root:", root)
print('Iterations:', iterations)

# 3.1 Exercise 3
# Do you need to change the input of the original bisection method? If so, how did it change?
# Yes we changed the input of the original bisection method. We added the derivative of the function as an input parameter. This is required for Newton’s method to check if the midpoint lies in its basin of convergence

# 3.1 Exercise 4

# Newtons Method Code: 
################################################################################
# This python script presents examples regarding the newton method and its
# application to 1D nonlinear root-finding, as presented in class.
################################################################################
# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;

# First, we define a function we will test the Newton method with. For each
# function we define, we also define its derivative.
# Our test function from previous sections
def fun(x):
    return x + np.cos(x)-3;
def dfun(x):
    return 1 - np.sin(x);

################################################################################
# We now implement the Newton method
def newton_method(f, df, x0, tol=1e-10, nmax=50, verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0;
    rn=np.array([x0]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            print('\n derivative at initial guess is near 0, try different x0 \n');
    else:
        n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)));

            pn = - fn/dfn; #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn;

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            dfn=df(xn);
            fn=f(xn);
            nfun+=2;

        r=xn;

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)
################################################################################
# Now, we apply this method to our test function

def bisect_newton(f, df, a, b, tol, nmax, vrb=False):
    an, bn, n = a, b, 0
    xn = (an + bn) / 2
    rn = np.array([xn])

    if f(a) * f(b) >= 0:
        print("\n Interval is inadequate, f(a)*f(b)>=0. Try again \n")
        return None

    if vrb:
        print("\n|--n--|--an--|--bn--|----xn----|-|bn-an|--|---|f(xn)|---|")

    while n <= nmax:
        if newton_method(f, df, xn)[0] is not None:
            break

        if vrb:
            print("|--%d--|%1.4f|%1.4f|%1.8f|%1.8f|%1.8f|" % (n, an, bn, xn, bn - an, np.abs(f(xn))))

        if (bn - an) < 2 * tol:
            break

        if f(an) * f(xn) < 0:
            bn = xn
        else:
            an = xn

        n += 1
        xn = (an + bn) / 2
        rn = np.append(rn, xn)

    return newton_method(f, df, xn)

# Example run
root, iterations, _ = bisect_newton(fun, dfun, 3, 4, 5e-16, 100, True)
print("Estimated root:", root)
print('Iterations:', iterations)

# 3.1 Exercise 5
# Compared to bisection, this method requires fewer iterations because Newton’s method converges much faster
# If the function has discontinuities, multiple roots close together, or derivative issue, Newtons method will still struggle so that's not good

