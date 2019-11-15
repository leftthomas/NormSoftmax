import matplotlib.pyplot as plt
import numpy as np

# for k
# -0.001923 x^2 + 0.154 x + 62.47
x = np.array([6, 10, 12, 18, 24, 36, 48, 56, 60, 80])
y = np.array([61.6, 63.8, 65.3, 65.2, 65.9, 65.5, 65.0, 64.1, 64.8, 63.0])

z1 = np.polyfit(x, y, 2)  # 用2次多项式拟合
p1 = np.poly1d(z1)
print(p1)  # 在屏幕上打印拟合多项式
plot1 = plt.plot(x, y, '*', label='original values')

x2 = np.linspace(1, 100, 100)
yvals = p1(x2)
plot2 = plt.plot(x2, yvals, 'r', label='polyfit values')

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.savefig('k_value.pdf')
plt.show()
