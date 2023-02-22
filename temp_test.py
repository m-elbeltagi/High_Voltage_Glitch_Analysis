from __future__ import division
from scipy.integrate import trapz
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# def step(x):
#     if x <= 1:
#         return x
#     else:
#         return (1-(abs(1-x)))
# ## to make it apply elementwise for the comparison operator
# step = np.vectorize(step)


# x = np.linspace(0,2,num=100)
# #print(x)


# y = step(x)

# plt.scatter(x, y, c = 'k')

# ## simpsons rule must have an odd number of samples! (or sometimes doesn't get correct answer)
# ## even = 'avg' takes care of this problem however
# ## simps in general doesn't seem to work for horizontal lines, like in a step function
# result = simps(y, x)

# #result = np.trapz(x,y)

# print(result)


year = [7000,7000,7500,7500,8650,]
price = [2014,2013,2011,2011,2011,]

print ('price: {}'.format(np.mean(price)))
print ('year: {}'.format(np.mean(year)))


## 2010 honda civic, ~200,000km, 6400$ (on vrm ~7800$)
## 2012 hyundai elantra, ~150,000-200,000, 6000$ (on vrm ~7500$)
## 2012 hyundai sonata, ~150,000-200,000, 7500$ (on vrm ~8500)