""" A module for utility methods
"""

import numpy as np

import PIL.Image

from io import BytesIO
from IPython.display import Image, display


def display_fractal(arr, fmt='jpeg'):
    """Display an array of iteration counts as a
       colorful picture of a fractal.

    Adapted from https://www.tensorflow.org/tutorials/mandelbrot

    For use within jupyter

    Args:
      arr: iteration counts, or whatever - np.ndarray [x_dim, y_dim]
      fmt: image format - str
    """
    arr_cyclic = (6.28 * arr / 20.0).reshape(list(arr.shape) + [1])
    img = np.concatenate([10+20 * np.cos(arr_cyclic),
                          30+50 * np.sin(arr_cyclic),
                          155-80 * np.cos(arr_cyclic)], 2)
    img[arr == arr.max()] = 0
    arr = img
    arr = np.uint8(np.clip(arr, 0, 255))
    fh = BytesIO()
    PIL.Image.fromarray(arr).save(fh, fmt)
    display(Image(data=file.getvalue()))
