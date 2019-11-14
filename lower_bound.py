import math

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def s_lower_bound(n, m, epsilon=1e-8):
    assert n > 1 and m > 1 and epsilon > 0 and epsilon < 1
    first_constraint = n / (2 ** m)
    second_constraint = n ** (1 / m)
    third_constraint = (n - (n ** 2) * (epsilon ** (1 / (m - 1)))) / (1 - epsilon ** (1 / (m - 1)))
    return math.ceil(max(first_constraint, second_constraint, third_constraint))


def m_lower_bound(n, s, epsilon=1e-8):
    assert n > 1 and s > 1 and epsilon > 0 and epsilon < 1
    first_constraint = math.log2(n - s)
    second_constraint = math.log(n, s)
    third_constraint = math.log(epsilon, ((n ** 2 - s) / (n - s))) + 1
    return math.ceil(max(first_constraint, second_constraint, third_constraint))


X_s_list, X_m_list, Y_s_list, Y_m_list, Z_s_list, Z_m_list = [], [], [], [], [], []
for i in range(1000, 20000, 50):
    for j in range(15, 100, 5):
        X_s_list.append(i)
        Y_s_list.append(j)
        Z_s_list.append((s_lower_bound(i, j)))
        if i > j:
            X_m_list.append(i)
            Y_m_list.append(j)
            Z_m_list.append(m_lower_bound(i, j))

X_s_list, X_m_list = np.array(X_s_list), np.array(X_m_list)
Y_s_list, Y_m_list = np.array(Y_s_list), np.array(Y_m_list)
Z_s_list, Z_m_list = np.array(Z_s_list), np.array(Z_m_list)

fig = plt.figure()
ax = Axes3D(fig)

# ax.scatter3D(X_s_list, Y_s_list, Z_s_list, cmap='Blues')
ax.scatter3D(X_m_list, Y_m_list, Z_m_list, cmap='Blues')
ax.set_xlabel('n value')
ax.set_ylabel('k value')
ax.set_zlabel('m value')
plt.savefig('m_value.pdf')
plt.show()
