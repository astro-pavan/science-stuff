import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660]
])

RGB_to_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
])

M2_inv = np.linalg.inv(M2)

LMS_to_RGB = np.linalg.inv(RGB_to_LMS)

def srgb_to_linear(c):
    """Reverses the standard sRGB gamma correction."""
    return np.where(c <= 0.04045, 
                    c / 12.92, 
                    ((c + 0.055) / 1.055) ** 2.4)

def rgb_to_oklch(rgb):
    # 1. Reverse Gamma (sRGB -> Linear RGB)
    linear_rgb = srgb_to_linear(rgb)
    
    # 2. Linear RGB -> LMS
    lms = RGB_to_LMS @ linear_rgb
    
    # 3. Apply non-linearity (Cube root)
    # Note: Using cbrt to handle potential precision issues with negative numbers
    lms_prime = np.cbrt(lms)
    
    # 4. LMS -> Oklab (L, a, b)
    oklab = M2 @ lms_prime
    L, a, b = oklab
    
    # 5. Oklab -> Oklch (L, C, h)
    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)
    
    return L, C, h

def linear_to_srgb(c):
    """Applies the standard sRGB gamma correction."""
    c = np.clip(c, 0, 1) # Clip linear values to gamut before gamma correction
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1 / 2.4)) - 0.055)

def oklch_to_rgb(L, C, h):
    a = C * np.cos(h)
    b = C * np.sin(h)
    oklab = np.stack([L, a, b], axis=0)
    
    # Standardize multiplication
    lms = np.tensordot(M2_inv, oklab, axes=(1, 0)) ** 3
    linear_rgb = np.tensordot(LMS_to_RGB, lms, axes=(1, 0))
    rgb = linear_to_srgb(linear_rgb)
    
    return rgb

def color_gradient(rgb1, rgb2, n=256):

    L1, C1, h1 = rgb_to_oklch(rgb1)
    L2, C2, h2 = rgb_to_oklch(rgb2)

    rgb_range = color_spiral(L1, L2, C1, C2, h1, h2, n)

    return rgb_range

def color_spiral(L1, L2, C1, C2, h1, h2, n=256):
    L_range = np.linspace(L1, L2, num=n)
    C_range = np.linspace(C1, C2, num=n)
    h_range = np.linspace(h1, h2, num=n)

    # Transpose here to get (N, 3)
    rgb_range = oklch_to_rgb(L_range, C_range, h_range).T
    return rgb_range

def color_map_2d(x, y, h_offset=0):
    L = x/2 + y/2
    C = 0.2
    h = 2 * np.arctan2(x, y)+ h_offset

    rgb = oklch_to_rgb(L, C, h)
    
    # Move the color channel from index 0 to index 2: (3, H, W) -> (H, W, 3)
    if rgb.ndim == 3:
        rgb = np.moveaxis(rgb, 0, -1)
        
    return rgb

if __name__ == '__main__':

    offset = 5
    spiral = color_spiral(0.2, 0.7, 0, 0.2, 0 + offset, 2*np.pi + offset)

    lilac = np.array([0x60, 0x64, 0xBF]) / 255
    lilac_lch = rgb_to_oklch(lilac)
    print(lilac_lch)

    colors = color_spiral(0.2, 0.54, 0.0, 0.3, -1.417 + 2*np.pi, -1.417)

    my_cmap = LinearSegmentedColormap.from_list("my_custom_gradient", colors)

    x = np.linspace(-2, 2)
    y = np.linspace(-2, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(- (X**2 + Y**2))

    plt.contourf(X, Y, Z, 200, cmap=my_cmap)
    plt.colorbar()
    plt.show()

    X, Y = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))

    rgb = color_map_2d(X, Y)

    plt.imshow(rgb, origin='lower', extent=[0, 1, 0, 1])
    plt.show()
