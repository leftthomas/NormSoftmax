import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N, K, Prob = [], [], []
for n in range(10, 30, 1):
    for k in range(2, 10, 1):
        if n > k:
            N.append(n)
            K.append(k)

N, K = np.array(N), np.array(K)
N, K = np.meshgrid(N, K)
a, b = np.floor(N / K), np.ceil(N / K)
i = K - (N % K)
Prob = (i * a * (a - 1) + (K - i) * b * (b - 1)) / (N * (N - 1))

fig = plt.figure()
ax = Axes3D(fig, azim=-225)

ax.plot_surface(N, K, Prob, cmap='rainbow')
ax.set_xlabel('n value')
ax.set_ylabel('k value')
ax.set_zlabel('prob of invalid')
plt.savefig('prob_invalid.pdf')
plt.show()
