import numpy as np


def npsafe_divide(numerator, denominator):
    return np.where(
        np.greater(denominator, 0),
        np.divide(numerator, denominator),
        np.zeros_like(numerator))