import matplotlib.pyplot as plt

from plot_decision_boundary import plot_decision_boundary
from utils.plot_features import plot_features


def plot_decision_model(theta, X, y, degree):
    """
    Создает побочный эффект, отображающий модель на графике
    """

    plot_features(X, y)
    plot_decision_boundary(theta, X, degree)
    plt.xlabel('Тест 1')
    plt.ylabel('Тест 2')
