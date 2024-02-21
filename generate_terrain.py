"""

get numerical terrain

author: Tianlong Zhang (SWJTU)

See article (https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12350)

"""
import numpy as np
from matplotlib import pyplot as plt

s1 = [1, 2.5, 3.8, 1.75, 2.5]
s2 = [1.5, 3.1, 1.3, -0.25, 2]
l = [1, 1, 1, 0.75, 1]
m = [0.5, 1, 2, 1, 10]
n = [1, 1.5, 0.5, 1, 10]
d = [0.3, 0.4, 0.3, 0.5, 0.8]
qmin = [0, 0, 0, 0, 0.7]


def qxy(x, y, i):
    out = d[i] * (l[i] ** 2 - (((x - s1[i]) / m[i]) ** 2 + ((y - s2[i]) / n[i]) ** 2))
    return out


def hxy(x, y):
    out = 1
    for i in range(5):
        out += (max(qxy(x, y, i), qmin[i])) ** 2
    return out


def get_terrain(res=0.01):
    x = np.arange(start=0, stop=5, step=res)
    y = np.arange(start=0, stop=5, step=res)
    X, Y = np.meshgrid(x, y)
    vfun = np.vectorize(hxy)
    Z = vfun(X, Y)
    return Z.T


if __name__ == "__main__":
    terrain = get_terrain(0.001)
    np.savetxt('terrain.txt', terrain, fmt='%.5f')
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = np.arange(start=-1, stop=6, step=.1)
    y = np.arange(start=-2, stop=6, step=.1)
    X, Y = np.meshgrid(x, y)
    vfun = np.vectorize(hxy)
    Z = vfun(X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.9, cstride=1, rstride=1, cmap="rainbow")
    plt.show()