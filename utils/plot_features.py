from matplotlib import pyplot as plt


def plot_features(X, y):
    """
    Создает побочный эффект, отображающий примеры на графике
    """
    pos = y == 1
    neg = y == 0

    # Выборка признаков для положительных и отрицательных примеров
    X_pos = X[pos, 1:3]
    X_neg = X[neg, 1:3]

    # Построение графика для положительных примеров
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c='b', marker='+', label='y = 1')

    # Построение графика для отрицательных примеров
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c='r', marker='o', label='y = 0')
