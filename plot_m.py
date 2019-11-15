import matplotlib.pyplot as plt
import numpy as np

# for m
#
x = np.array([2, 6, 10, 12, 18, 24, 30, 36, 48, 64, 80])
y = np.array([35.1, 52.2, 56.5, 58.1, 60.7, 63.5, 63.2, 64.6, 65.3, 65.9, 66.1])

z1 = np.polyfit(np.log(x), y, 1)  # 用log拟合
p1 = np.poly1d(z1)
print(p1)  # 在屏幕上打印拟合多项式
plot1 = plt.plot(x, y, '*', label='original values')

x2 = np.linspace(1, 100, 100)
yvals = p1(np.log(x2))
plot2 = plt.plot(x2, yvals, 'r', label='polyfit values')

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.savefig('m_value.pdf')
plt.show()
