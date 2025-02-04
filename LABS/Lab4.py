import numpy as np

# PreLab # Exercise 2.2

def fixedpt(f, x0, tol, Nmax):
    '''
    x0 = initial guess
    Nmax = max number of iterations
    tol = stopping tolerance
    Returns a vector of approximations of the fixed point at each iteration
    '''
    x_vals = np.zeros((Nmax, 1))  # Store approximations
    x_vals[0] = x0
    count = 0
    
    while count < Nmax - 1:
        count += 1
        x1 = f(x0)
        x_vals[count] = x1
        
        if abs(x1 - x0) < tol:
            return x_vals[:count+1]  # Return only computed values
        
        x0 = x1
    
    return x_vals[:count+1]  # Return all approximations if max iterations reached

def driver():
    # 2.2 Exercise 
    g1 = lambda x: (10/ (x+4)) ** 0.5
    
    
    Nmax = 100
    tol = 1e-10
    
    # Test g1
    x0 = 1.5
    approximations = fixedpt(g1, x0, tol, Nmax)
    print('Approximations for g1:', approximations.flatten())
    print(f'Converged in {len(approximations) - 1} iterations.')
    

driver()

# 3.2 Aitken's Acceleration Technique

def fixedpt(f, x0, tol, Nmax):
    '''
    x0 = initial guess
    Nmax = max number of iterations
    tol = stopping tolerance
    Returns a vector of approximations of the fixed point at each iteration
    '''
    x_vals = np.zeros((Nmax, 1))  # Store approximations
    x_vals[0] = x0
    count = 0
    
    while count < Nmax - 1:
        count += 1
        x1 = f(x0)
        x_vals[count] = x1
        
        if abs(x1 - x0) < tol:
            return x_vals[:count+1]  # Return only computed values
        
        x0 = x1
    
    return x_vals[:count+1]


def aitken_delta2(seq):
   
    aitken_seq = []
    
    for n in range(len(seq) - 2):  # Need three consecutive terms
        p_n, p_n1, p_n2 = seq[n], seq[n+1], seq[n+2]
        denominator = p_n2 - 2 * p_n1 + p_n
        if abs(denominator) < 1e-12:  # Avoid division by zero
            break
        
        p_n_prime = p_n - ((p_n1 - p_n) ** 2) / denominator
        aitken_seq.append(p_n_prime)
    
    return np.array(aitken_seq)

def driver():
    # 2.2 Exercise 
    g1 = lambda x: (10/ (x+4)) ** 0.5
    
    
    Nmax = 100
    tol = 1e-10
    
    # Test g1 again but using both to compare the methods
    x0 = 1.5
    approximations = fixedpt(g1, x0, tol, Nmax)
    aitken_approximations = aitken_delta2(approximations)
    print('Approximations for g1:', approximations.flatten())
    print(f'Converged in {len(approximations) - 1} iterations.') 
    print(f'Aitken’s Δ² method accelerates convergence in {len(aitken_approximations)} iterations.')
    print('Aitken-accelerated approximations:', aitken_approximations.flatten())

driver()

# SubRoutine Vector Approximations: 

def sub_aitken(seq, tol=1e-10, Nmax=100):
    aitken_seq = []
    count = 0  # Track number of iterations

    for n in range(len(seq) - 2):  
        if count >= Nmax:
            break  # Stop when max iterations are reached

        p_n, p_n1, p_n2 = seq[n], seq[n+1], seq[n+2]
        denominator = p_n2 - 2 * p_n1 + p_n

        if abs(denominator) < 1e-12:  
            break
        
        # Compute Aitken's accelerated approximation
        p_n_prime = p_n - ((p_n1 - p_n) ** 2) / denominator
        aitken_seq.append(p_n_prime)
        count += 1

        # Check tolerence
        if len(aitken_seq) > 1 and abs(aitken_seq[-1] - aitken_seq[-2]) < tol:
            break

    return np.array(aitken_seq)

# Example
if __name__ == "__main__":
    # Example sequence from pre lab iterations
    example_seq = np.array([1.5, 1.34839972, 1.36737637, 1.36495702, 1.36526475, 1.36522559,
 1.36523058, 1.36522994, 1.36523002, 1.36523001, 1.36523001, 1.36523001,
 1.36523001])
    # Run Aitken’s method
    accelerated_seq = sub_aitken(example_seq, tol=1e-10, Nmax=50)
    
    # Print results
    print("Original sequence:", example_seq)
    print("Aitken-accelerated sequence:", accelerated_seq)
  
