import numpy as np
import matplotlib.pyplot as plt

from map_feature import map_feature


def plot_decision_boundary(theta, X, degree):
    """
    Строит границу решения на графике
    """

    if X.shape[1] <= 3:
        plot_x = np.array([X[:, 1].min() - 2, X[:, 1].max() + 2])
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        plt.plot(plot_x, plot_y, label='Граница решения')
        plt.xlim([X[:, 1].min() - 2, X[:, 1].max() + 2])
        plt.ylim([X[:, 2].min() - 2, X[:, 2].max() + 2])
    else:
        u = np.linspace(-2.5, 2.5, 500)
        v = np.linspace(-2.5, 2.5, 500)

        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = map_feature(np.array([u[i]]), np.array([v[j]]), degree) @ theta

        z = z.T
        plt.contour(u, v, z, levels=[0], linewidths=1)
