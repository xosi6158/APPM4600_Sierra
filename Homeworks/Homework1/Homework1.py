import matplotlib.pyplot as plt
import numpy as np
# Problem 1
coeffs = [1, -18, 144, -672, 2016, -4032, 5376, -4608, 2304, -512]
x = np.arange(1.920, 2.080, 0.001)
y = np.polyval(coeffs, x)
#plt.plot(x,y)
#plt.xlabel('x')
# plt.ylabel('y')
#y_2 = (x-2)**9
#plt.plot(x,y_2)
#plt.xlabel('x')
#plt.ylabel('y')
# plt.show()

# Problem 5
#x1 = np.pi
#x2 = 10**6
#delta = np.logspace(-16,0,num=17)
#y1 = np.abs(np.cos(x1+delta)-np.cos(x1)+2*np.sin(x1+(delta/2))*np.sin(delta/2))
#y2 = np.abs(np.cos(x2+delta)-np.cos(x2)+2*np.sin(x2+(delta/2))*np.sin(delta/2))
#plt.plot(delta,y1,label=f'x={x1}')
#plt.plot(delta,y2,label=f'x={x2}')
#plt.yscale('log')
#plt.xscale('log')
#plt.ylabel('Difference')
#plt.xlabel('Delta (log scale)')
#plt.title('Cosine Expression vs Cosine Approximation')
#plt.legend()
#plt.show()

# Problem 5c
#x1 = np.pi
#x2 = 10**6
#delta = np.logspace(-16,0,num=17)
#y_1 = np.abs(-delta*np.sin(x1)-((delta**2/2)*np.cos(x1)))
#y_2 = np.abs(-delta*np.sin(x2)-((delta**2/2)*np.cos(x2)))
#plt.plot(delta,y_1,label=f'x={x1}')
#plt.plot(delta,y_2,label=f'x={x2}')
#plt.yscale('log')
#plt.xscale('log')
#plt.ylabel('Taylor Approximation')
#plt.xlabel('Delta (log scale)')
#plt.title('Taylor Approximation of Cosine')
#plt.legend()
#plt.show()

# Problem 5c
x1 = np.pi
x2 = 10**6
delta = np.logspace(-16,0,num=17)
taylor_1 = (-delta*np.sin(x1)-((delta**2/2)*np.cos(x1)))
taylor_2 = (-delta*np.sin(x2)-((delta**2/2)*np.cos(x2)))
original_1= (np.cos(x1+delta)-np.cos(x1))
original_2= (np.cos(x2+delta)-np.cos(x2))
error_1= np.abs(taylor_1-original_1) 
error_2= np.abs(taylor_2-original_2)
plt.plot(delta,error_1,label=f'x={x1}')
plt.plot(delta,error_2,label=f'x={x2}')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Error Approximation')
plt.xlabel('Delta (log scale)')
plt.title('Taylor Error Approximation of Cosine')
plt.legend()
plt.show()


