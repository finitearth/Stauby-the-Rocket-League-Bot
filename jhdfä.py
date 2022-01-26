import numpy as np

a = np.zeros((3,8))
b = np.array([[2, 3, 4], [4, 5, 6], [7,8, 46]])
a[..., [4, 5, 6]] = b[..., [0, 1, 2]]
print(a)