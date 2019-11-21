import math

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def s_lower_bound(n, m, epsilon=1e-8):
    assert n > 1 and m > 1 and epsilon > 0 and epsilon < 1
    first_constraint = n ** (1 / m)
    second_constraint = 4 / (epsilon ** (1 / (m - 1)))
    return math.ceil(max(first_constraint, second_constraint))


def m_lower_bound(n, s, epsilon=1e-8):
    assert n > 1 and s > 1 and epsilon > 0 and epsilon < 1
    first_constraint = math.log(n, s)
    second_constraint = math.log(epsilon, s / 4) + 1
    return math.ceil(max(first_constraint, second_constraint))


def get_index_s_lower_bound(n, m, epsilon=1e-8):
    assert n > 1 and m > 1 and epsilon > 0 and epsilon < 1
    first_constraint = n ** (1 / m)
    second_constraint = 4 / (epsilon ** (1 / (m - 1)))
    if first_constraint >= second_constraint:
        return 1
    else:
        return 2


def get_index_m_lower_bound(n, s, epsilon=1e-8):
    assert n > 1 and s > 1 and epsilon > 0 and epsilon < 1
    first_constraint = math.log(n, s)
    second_constraint = math.log(epsilon, s / 4) + 1
    if first_constraint >= second_constraint:
        return 1
    else:
        return 2


X_s_list, X_m_list, Y_s_list, Y_m_list, Z_s_list, Z_m_list, Z_s_index_list, Z_m_index_list = [], [], [], [], [], [], [], []
for i in range(1000, 20000, 500):
    for j in range(5, 1000, 5):
        X_s_list.append(i)
        Y_s_list.append(j)
        Z_s_list.append((s_lower_bound(i, j)))
        Z_s_index_list.append(get_index_s_lower_bound(i, j))
        if i > j:
            X_m_list.append(i)
            Y_m_list.append(j)
            Z_m_list.append(m_lower_bound(i, j))
            Z_m_index_list.append(get_index_m_lower_bound(i, j))

X_s_list, X_m_list = np.array(X_s_list), np.array(X_m_list)
Y_s_list, Y_m_list = np.array(Y_s_list), np.array(Y_m_list)
Z_s_list, Z_m_list = np.array(Z_s_list), np.array(Z_m_list)

fig = plt.figure()
ax = Axes3D(fig)

# ax.scatter3D(X_m_list, Y_m_list, Z_m_list)
# ax.set_xlabel('n value')
# ax.set_ylabel('k value')
# ax.set_zlabel('m value')
# plt.savefig('m_value.pdf')
# plt.show()

ax.scatter3D(X_s_list, Y_s_list, Z_s_list)
ax.set_xlabel('n value')
ax.set_ylabel('m value')
ax.set_zlabel('k value')
plt.savefig('k_value.pdf')
plt.show()

# ax.scatter3D(X_s_list, Y_s_list, Z_m_index_list)
# ax.set_xlabel('n value')
# ax.set_ylabel('k value')
# ax.set_zlabel('constraint number')
# plt.savefig('m_index.pdf')
# plt.show()

# ax.scatter3D(X_s_list, Y_s_list, Z_s_index_list)
# ax.set_xlabel('n value')
# ax.set_ylabel('m value')
# ax.set_zlabel('constraint number')
# plt.savefig('k_index.pdf')
# plt.show()
