import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./planetary-chem-presentation.mplstyle')

x = np.linspace(-np.pi, np.pi)
y1 = -np.sin(x)
y2 = np.cos(x)
y3 = x
y4 = np.tanh(x)

plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')
plt.plot(x, y3, label='y3')
plt.plot(x, y4, label='y4')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')

plt.savefig('plot.png')
plt.close()
# plt.show()

x = np.linspace(-2, 2)
y = np.linspace(-2, 2)
X, Y = np.meshgrid(x, y)

Z = np.exp(- (X**2 + Y**2))

plt.contourf(X, Y, Z, 200)
plt.colorbar(label='Z')
plt.xlabel('X')
plt.ylabel('Y')

plt.savefig('plot2.png')
