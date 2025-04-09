# Homework 10

# Problem 1 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pade

def sin_function(x):
    return np.sin(x)

# Maclaurin series for sin(x) up to x^6
maclaurin_coeffs = [0, 1, 0, -1/6, 0, 1/120, 0]


# Numerator and denominator both cubic (3, 3)
pade_33_numerator, pade_33_denominator = pade(maclaurin_coeffs, 3)

# Numerator quadratic, denominator quartic (2, 4)
pade_24_numerator, pade_24_denominator = pade(maclaurin_coeffs, 2)

# Numerator quartic, denominator quadratic (4, 2)
pade_42_numerator, pade_42_denominator = pade(maclaurin_coeffs, 4)

# functions for each approximation
def pade_33(x):
    return pade_33_numerator(x) / pade_33_denominator(x)

def pade_24(x):
    return pade_24_numerator(x) / pade_24_denominator(x)

def pade_42(x):
    return pade_42_numerator(x) / pade_42_denominator(x)

# Plot
x_values = np.linspace(0, 5, 500)
true_values = sin_function(x_values)
pade_33_values = pade_33(x_values)
pade_24_values = pade_24(x_values)
pade_42_values = pade_42(x_values)

print('Padé (3,3) approximation:', pade_33_values)
print('Padé (2,4) approximation:', pade_24_values)
print('Padé (4,2) approximation:', pade_42_values)

# sixth-order Maclaurin polynomial
mac_approximation = np.polyval(maclaurin_coeffs[::-1], x_values)
print('Maclaurin approximation:', mac_approximation)

# errors
error_33 = np.abs(true_values - pade_33_values)
error_24 = np.abs(true_values - pade_24_values)
error_42 = np.abs(true_values - pade_42_values)
error_maclaurin = np.abs(true_values - mac_approximation)

# Plotting the errors
plt.figure(figsize=(10, 5))
plt.plot(x_values, error_33, label='Error Padé (3,3)', linestyle='--')
plt.plot(x_values, error_24, label='Error Padé (2,4)', linestyle='-.')
plt.plot(x_values, error_42, label='Error Padé (4,2)', linestyle=':')
plt.plot(x_values, error_maclaurin, label='Error Maclaurin (6th order)', linestyle='-')
plt.title('Error Comparison of Padé Approximations and Maclaurin Polynomial for sin(x)')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.show()



