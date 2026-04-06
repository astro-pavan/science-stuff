import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Removed unused M1/M1_inv matrices to clean up the code

M2_inv = np.linalg.inv(np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660]
]))

LMS_to_RGB = np.linalg.inv(np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
]))

def linear_to_srgb(c):
    """Applies the standard sRGB gamma correction."""
    c = np.clip(c, 0, 1) # Clip linear values to gamut before gamma correction
    return np.where(c <= 0.0031308, 
                    12.92 * c, 
                    1.055 * (c ** (1 / 2.4)) - 0.055)

def oklch_to_rgb(L, C, h):
    a = C * np.cos(h)
    b = C * np.sin(h)

    oklab = np.array([L, a, b])
    
    # Convert Oklab to LMS
    lms = (M2_inv @ oklab) ** 3
    
    # Convert LMS to Linear RGB
    linear_rgb = LMS_to_RGB @ lms
    
    # Convert Linear RGB to standard display sRGB!
    rgb = linear_to_srgb(linear_rgb)

    return rgb

# FIX: Bump resolution to 256 to prevent Matplotlib from interpolating in RGB
n = 256 

L = np.linspace(0.0, 1.0, num=n)
C = np.linspace(0.0, 0.25, num=n)
h1 = 1.5
h = np.linspace(h1 * np.pi, (h1 + 2) * np.pi, num=n)

rgb = oklch_to_rgb(L, C, h).T

colors = rgb
my_cmap = LinearSegmentedColormap.from_list("my_custom_gradient", colors)

x = np.linspace(-2, 2)
y = np.linspace(-2, 2)
X, Y = np.meshgrid(x, y)
Z = np.exp(- (X**2 + Y**2))

plt.contourf(X, Y, Z, 200, cmap=my_cmap)
plt.colorbar()
plt.show()
