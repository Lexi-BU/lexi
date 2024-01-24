import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt

a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], "d")
b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], "d")
mt1, mt2, disparity = procrustes(a, b)

a_mean_value = a - np.mean(a, 0)
a_mean_norm = np.linalg.norm(a_mean_value)

b_mean_value = b - np.mean(b, 0)
b_mean_norm = np.linalg.norm(b_mean_value)
b_norm = b_mean_value / b_mean_norm

new_a = a_mean_norm * mt1 + np.mean(a, 0)
