#Homework4

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Question 1 Problem A 

# Constants
Ti = 20  
Ts = -15  
alpha = 0.138e-6  # m^2/s 
t = 5184000  # 60 days in seconds (bc of units of alpha)
# erf(x / (2 * np.sqrt(alpha * t))) - 3/7

def f(x):
    return erf(x / (2 * np.sqrt(alpha * t))) - 3/7

# Plot
x_vals = np.linspace(0, 10, 500)
y_vals = f(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label='$f(x)$')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Depth x (m)')
plt.ylabel('f(x)')
plt.title('Root Finding Problem for Freezing Depth')
plt.grid(True)
plt.show()



# Question 1 Problem B

Ti = 20
Ts = -15
alpha = 0.138e-6
t = 5184000  # 60 days in seconds

def f(x):
    return erf(x / (2 * np.sqrt(alpha * t))) - 3/7


def bisect_method(f,a,b,tol,nmax,vrb=False):
    #Bisection method applied to f between a and b

    # Initial values for interval [an,bn], midpoint xn
    an = a
    bn=b
    n=0
    xn = (an+bn)/2
    # Current guess is stored at rn[n]
    rn=np.array([xn])
    r=xn
    ier=0

    if vrb:
        print("\n Bisection method with nmax=%d and tol=%1.1e\n" % (nmax, tol))

    # The code cannot work if f(a) and f(b) have the same sign.
    # In this case, the code displays an error message, outputs empty answers and exits.
    if f(a)*f(b)>=0:
        print("\n Interval is inadequate, f(a)*f(b)>=0. Try again \n")
        print("f(a)*f(b) = %1.1f \n" % f(a)*f(b))
        r = None
        return r
    else:
        #If f(a)f(b), we proceed with the method.
        if vrb:
            print("\n|--n--|--an--|--bn--|----xn----|-|bn-an|--|---|f(xn)|---|")

            # We start two plots as subplots in the same figure.
            fig, (ax1, ax2) = plt.subplots(1, 2) #Creates figure fig and subplots
            fig.suptitle('Bisection method results') #Sets title of the figure
            ax1.set(xlabel='x',ylabel='y=f(x)') #Sets labels for axis for subplot 1
            # We plot y=f(x) on the left subplot.
            xl=np.linspace(a,b,100,endpoint=True); yl=f(xl)
            ax1.plot(xl,yl)

        while n<=nmax:
            if vrb:
                print("|--%d--|%1.4f|%1.4f|%1.8f|%1.8f|%1.8f|" % (n,an,bn,xn,bn-an,np.abs(f(xn))))

                ################################################################
                # Plot results of bisection on subplot 1 of 2 (horizontal).
                xint = np.array([an,bn])
                yint=f(xint)
                ax1.plot(xint,yint,'ko',xn,f(xn),'rs')
                ################################################################

            # Bisection method step: test subintervals [an,xn] and [xn,bn]
            # If the estimate for the error (root-xn) is less than tol, exit
            if (bn-an)<2*tol: # better test than np.abs(f(xn))<tol
                ier=1
                break

            # If f(an)*f(xn)<0, pick left interval, update bn
            if f(an)*f(xn)<0:
                bn=xn
            else:
                #else, pick right interval, update an
                an=xn

            # update midpoint xn, increase n.
            n += 1
            xn = (an+bn)/2
            rn = np.append(rn,xn)

    # Set root estimate to xn.
    r=xn

    if vrb:
        ########################################################################
        # subplot 2: approximate error log-log plot
        e = np.abs(r-rn[0:n])
        #length of interval
        ln = (b-a)*np.exp2(-np.arange(0,e.size))
        #log-log plot error vs interval length
        ax2.plot(-np.log2(ln),np.log2(e),'r--')
        ax2.set(xlabel='-log2(bn-an)',ylabel='log2(error)')
        ########################################################################

    return r, rn
################################################################################
# Now, we run this method for our example function
(r,rn)=bisect_method(f,0,2,1e-13,100)
plt.show()
print(f"Approximate root (depth x) : {r:.6f} meters")
print(f"Number of iterations : {len(rn)}")

# Question 1 Problem C

def f(x):
    return erf(x / (2 * np.sqrt(alpha * t))) - 3/7
def der_f(x):
    return (1 / np.sqrt(np.pi*alpha*t)) * np.exp(-(x/(2*np.sqrt(alpha*t)))**2)

################################################################################
# We now implement the Newton method
def newton_method(f,der_f,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0
    rn=np.array([x0])
    # function evaluations
    fn=f(xn)
    dfn=der_f(xn)
    nfun=2 #evaluation counter nfun
    dtol=1e-10 #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            print('\n derivative at initial guess is near 0, try different x0 \n')
    else:
        n=0
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|")

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)))

            pn = - fn/dfn #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn

            # Update info and loop
            n+=1
            rn=np.append(rn,xn)
            dfn=der_f(xn)
            fn=f(xn)
            nfun+=2

        r=xn

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)))
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)))

    return (r,rn,nfun)
################################################################################
# Now, we apply this method to our test function
(r,rn,nfun)=newton_method(f,der_f,0.01,1e-13,100,True)
print(f"Approximate root (depth x) : {r:.6f} meters")
print(f"Number of iterations : {len(rn)}")

# Question 1 Problem C(cont)

def newton_method(f,der_f,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0
    rn=np.array([x0])
    # function evaluations
    fn=f(xn)
    dfn=der_f(xn)
    nfun=2 #evaluation counter nfun
    dtol=1e-10 #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            print('\n derivative at initial guess is near 0, try different x0 \n')
    else:
        n=0
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|")

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)))

            pn = - fn/dfn #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn

            # Update info and loop
            n+=1
            rn=np.append(rn,xn)
            dfn=der_f(xn)
            fn=f(xn)
            nfun+=2

        r=xn

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)))
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)))

    return (r,rn,nfun)
################################################################################
# Now, we apply this method to our test function
(r,rn,nfun)=newton_method(f,der_f,2,1e-13,100,True)
print(f"Approximate root (depth x) : {r:.6f} meters")
print(f"Number of iterations : {len(rn)}")

# Question 4 Newtons Method (class)

def f(x):
    return np.exp(3*x)-27*(x**6)+27*(x**4)*np.exp(x)-9*(x**2)*np.exp(2*x)
def der_f(x):
    return 3*np.exp(3*x)-162*(x**5)+108*(x**3)*np.exp(x)+27*(x**4)*np.exp(x)-18*x*np.exp(2*x)-18*(x**2)*np.exp(2*x)

# We now implement the Newton method
def newton_method(f,der_f,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0
    rn=np.array([x0])
    # function evaluations
    fn=f(xn)
    dfn=der_f(xn)
    nfun=2 #evaluation counter nfun
    dtol=1e-10 #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            print('\n derivative at initial guess is near 0, try different x0 \n')
    else:
        n=0
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|")

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)))

            pn = - fn/dfn #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn

            # Update info and loop
            n+=1
            rn=np.append(rn,xn)
            dfn=der_f(xn)
            fn=f(xn)
            nfun+=2

        r=xn

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)))
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)))

    return (r,rn,nfun)
################################################################################
# Now, we apply this method to our test function
(r,rn,nfun)=newton_method(f,der_f,3,1e-13,100,True)
print(f"Approximate root : {r:.6f} ")
print(f"Number of iterations : {len(rn)}")

# Question 4 Secant Method (Class)
def secant_method(f, x0, x1, tol, nmax, verb=False):
    xnm = x0
    xn = x1
    rn = np.array([x0, x1])
    fnm = f(xnm)
    fn = f(xn)
    nfun = 2
    dtol = 1e-10

    if abs(xn - xnm) < dtol:
        if verb:
            print('\n slope of secant at initial guess is near 0, try different x0, x1 \n')

    else:
        n = 0
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|")

        while n <= nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|" % (n, xn, np.abs(fn)))

            if abs(fn - fnm) < dtol:
                print("Secant method: small slope detected, terminating.")
                break

            pn = -fn * (xn - xnm) / (fn - fnm)

            if np.abs(pn) < tol or np.abs(fn) < 2e-15:
                break

            xnm = xn
            xn = xn + pn

            rn = np.append(rn, xn)
            fnm = fn
            fn = f(xn)
            nfun += 1
            n += 1

        r = xn

        if n >= nmax:
            print("Secant method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n" % (n, nfun, np.abs(fn)))
        else:
            print("Secant method converged successfully, niter=%d, nfun=%d, f(r)=%1.1e" % (n, nfun, np.abs(fn)))

    return r, rn, nfun

# Now, we apply this method to our test function
(r,rn,nfun)=secant_method(f,3,5,1e-13,100,True)
print(f"Approximate root : {r:.6f} ")
print(f"Number of iterations : {len(rn)}")


# From Problem 2c

def scalar(x0, m, tol, nmax):
    for n in range(nmax):
        fx0 =f(x0)
        fprime=der_f(x0)
        if abs(fx0) < tol:
            return x0
        x1 = x0 - m * (fx0 / fprime)
    return x1
s=scalar(3,3,1e-13,100)
print(f"Approximate root : {s:.6f} ")



# Question 5 

def f(x):
    return x**6 - x - 1

def der_f(x):
    return 6 * x**5 - 1

# Newton Method 
def newton_method(f,der_f,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0
    rn=np.array([x0])
    # function evaluations
    fn=f(xn)
    dfn=der_f(xn)
    nfun=2 #evaluation counter nfun
    dtol=1e-10 #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            print('\n derivative at initial guess is near 0, try different x0 \n')
    else:
        n=0
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|")

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)))

            pn = - fn/dfn #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn

            # Update info and loop
            n+=1
            rn=np.append(rn,xn)
            dfn=der_f(xn)
            fn=f(xn)
            nfun+=2

        r=xn

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)))
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)))

    return (r,rn,nfun)
################################################################################
# Now, we apply this method to our test function
(r,rn,nfun)=newton_method(f,der_f,2,1e-13,100,True)
print(f"Approximate root : {r:.6f} ")
print(f"Number of iterations : {len(rn)}")

# Secant Method

def secant_method(f, x0, x1, tol, nmax, verb=False):
    xnm = x0
    xn = x1
    rn = np.array([x0, x1])
    fnm = f(xnm)
    fn = f(xn)
    nfun = 2
    dtol = 1e-10

    if abs(xn - xnm) < dtol:
        if verb:
            print('\n slope of secant at initial guess is near 0, try different x0, x1 \n')

    else:
        n = 0
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|")

        while n <= nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|" % (n, xn, np.abs(fn)))

            if abs(fn - fnm) < dtol:
                print("Secant method: small slope detected, terminating.")
                break

            pn = -fn * (xn - xnm) / (fn - fnm)

            if np.abs(pn) < tol or np.abs(fn) < 2e-15:
                break

            xnm = xn
            xn = xn + pn

            rn = np.append(rn, xn)
            fnm = fn
            fn = f(xn)
            nfun += 1
            n += 1

        r = xn

        if n >= nmax:
            print("Secant method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n" % (n, nfun, np.abs(fn)))
        else:
            print("Secant method converged successfully, niter=%d, nfun=%d, f(r)=%1.1e" % (n, nfun, np.abs(fn)))

    return r, rn, nfun

# Now, we apply this method to our test function
(r,rn,nfun)=secant_method(f,2,1,1e-13,100,True)
print(f"Approximate root : {r:.6f} ")
print(f"Number of iterations : {len(rn)}")

# Question 5 problem A

from scipy.optimize import brentq

# Find the root in the interval (1, 2) using a reliable method
alpha = brentq(f, 1, 2)
print(f"Exact root (alpha): {alpha:.6f}")

r_newton, rn_newton, nfun_newton = newton_method(f, der_f, 2.0, 1e-6, 100)
errors_newton = np.abs(rn_newton - alpha)
r_secant, rn_secant, nfun_secant = secant_method(f, 2.0, 1.0, 1e-6, 100)
errors_secant = np.abs(rn_secant - alpha)

print(f"{'n':<5}{'Newton Error':<20}{'Secant Error':<20}")
for i in range(max(len(errors_newton), len(errors_secant))):
    err_newton = errors_newton[i] if i < len(errors_newton) else '---'
    err_secant = errors_secant[i] if i < len(errors_secant) else '---'
    print(f"{i:<5}{str(err_newton):<20}{str(err_secant):<20}")


# Question 5 problem B

# Plot |x_{k+1} - alpha| vs |x_k - alpha| on log-log scale
plt.figure(figsize=(8, 6))

# Newton's method
plt.loglog(errors_newton[:-1], errors_newton[1:], 'bo-', label="Newton's Method")

# Secant method
plt.loglog(errors_secant[:-1], errors_secant[1:], 'ro-', label='Secant Method')

plt.xlabel(r'$|x_k - \alpha|$')
plt.ylabel(r'$|x_{k+1} - \alpha|$')
plt.title('Convergence Order Plot (Log-Log Scale)')
plt.legend()
plt.grid(True)
plt.show()

# Estimate the slope (order of convergence) for each method using the final few iterations
from scipy.stats import linregress

def estimate_slope(errors):
    log_errors_k = np.log(errors[:-1])
    log_errors_k1 = np.log(errors[1:])
    slope, _, _, _, _ = linregress(log_errors_k, log_errors_k1)
    return slope

slope_newton = estimate_slope(errors_newton[-5:])
slope_secant = estimate_slope(errors_secant[-5:])

print(f"Estimated order of convergence (Newton's Method): {slope_newton:.2f}")
print(f"Estimated order of convergence (Secant Method): {slope_secant:.2f}")

