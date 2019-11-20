import math

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_prob(n, k):
    assert 1 < k < n
    a, b = math.floor(n / k), math.ceil(n / k)
    i = k - (n % k)
    return (i * a * (a - 1) + (k - i) * b * (b - 1)) / (n * (n - 1))


N, K, Prob = [], [], []
for n in range(10, 30, 1):
    for k in range(2, 10, 1):
        if n > k:
            N.append(n)
            K.append(k)
            Prob.append((get_prob(n, k)))

N, K, Prob = np.array(N), np.array(K), np.array(Prob)

fig = plt.figure()
ax = Axes3D(fig, azim=-225)

ax.scatter3D(N, K, Prob)
ax.set_xlabel('n value')
ax.set_ylabel('k value')
ax.set_zlabel('prob of invalid')
plt.savefig('prob_invalid.pdf')
plt.show()
