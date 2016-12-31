"""Main program for testing image kernels.
"""

from PIL import Image
from kernels import Mean, Gauss, Laplace, Sobel, Sharpen
import numpy as np
from scipy import misc


if __name__ == '__main__':
    data = np.uint8(misc.lena())

    kernel = Gauss()
    new_data = kernel.convolve(data)

    image = Image.fromarray(data)
    image.show()
    new_image = Image.fromarray(new_data)
    new_image.show()
