import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(-5, 5, 1000)
# y= [1 / (1 + np.exp(-i)) for i in x]

x = np.arange(1, 100, 20)
y = x ** 2 + 10


plt.plot(x, y)
plt.show()